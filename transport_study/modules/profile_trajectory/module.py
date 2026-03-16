import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule
from popsim.math_utils import soft_clip
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from transport_study.modules.profile_trajectory.profile_predictor.module import (
    Inputs as ProfilePredictorInputs,
)
from transport_study.modules.profile_trajectory.profile_predictor.module import (
    Outputs as ProfilePredictorOutputs,
)
from transport_study.modules.profile_trajectory.profile_predictor.module import (
    ProfilePredictor,
)


class ProfileTrajectoryOptimizer(TimeDepModule):
    """Module for optimizing a desired trajectory of plasma profiles"""

    config: "Config"

    profile_predictor: eqx.Module
    psigrid: tuple = eqx.field(static=True)

    # These are the things that we can control over time
    gapin: Array  # Inner gap [m]
    R0: Array  # Major radius [m]
    rxpt1: Array  # X-point 1 R [m]
    zxpt1: Array  # X-point 1 Z [m]
    rxpt2: Array  # X-point 2 R [m]
    zxpt2: Array  # X-point 2 Z [m]

    @chex.dataclass
    class Config:
        shape_times: (
            Array  # The times at which we specify the desired profile shapes [s]
        )
        # I think that including these ranges should prevent the autodiff from finding a gradient that pushes it out of range
        # because if you put it into a softclip it kinda makes a wall that the input can't get nudged into
        input_ranges: dict[
            str, tuple[float, float]
        ]  # The ranges for the input parameters during the trajectory

    @chex.dataclass
    class State:
        # Just a dummy variable to make it a dataclass, since we need to return something
        stateless: float = 0.0

    @chex.dataclass
    class Inputs:
        traj_time: float  # Current time, to know which point on the trajectory we're at
        # These are all the inputs that DIII-D has real-time feedback for
        # Plug in the waveforms for these in advance, and we expect them to be reasonably accurate
        Ip_MA: float  # [MA]
        B0: float  # On axis magnetic field [T]
        ne20: float  # Electron density [10^20 m^-3]
        beta: float  # Plasma beta [%]

    @chex.dataclass
    class Output:
        # Things that are necessary for computing the loss function
        profile_predictor_output: ProfilePredictorOutputs
        psi: Array

    def __init__(
        self,
        config: Config,
        profile_predictor: eqx.Module,
        psigrid: tuple,
        trajectory: dict[str, Array] | None = None,
    ):
        self.config = config
        self.profile_predictor = profile_predictor
        self.psigrid = psigrid

        num_times = len(config.shape_times)

        if trajectory is None:
            # Initialize input trajectories at the center of the input ranges
            self.gapin = (
                jnp.ones(num_times)
                * (config.input_ranges["gapin"][0] + config.input_ranges["gapin"][1])
                / 2
            )
            self.R0 = (
                jnp.ones(num_times)
                * (config.input_ranges["R0"][0] + config.input_ranges["R0"][1])
                / 2
            )
            self.rxpt1 = (
                jnp.ones(num_times)
                * (config.input_ranges["rxpt1"][0] + config.input_ranges["rxpt1"][1])
                / 2
            )
            self.zxpt1 = (
                jnp.ones(num_times)
                * (config.input_ranges["zxpt1"][0] + config.input_ranges["zxpt1"][1])
                / 2
            )
            self.rxpt2 = (
                jnp.ones(num_times)
                * (config.input_ranges["rxpt2"][0] + config.input_ranges["rxpt2"][1])
                / 2
            )
            self.zxpt2 = (
                jnp.ones(num_times)
                * (config.input_ranges["zxpt2"][0] + config.input_ranges["zxpt2"][1])
                / 2
            )
        else:
            # Load trajectories from the provided dictionary
            self.gapin = trajectory["gapin"]
            self.R0 = trajectory["R0"]
            self.rxpt1 = trajectory["rxpt1"]
            self.zxpt1 = trajectory["zxpt1"]
            self.rxpt2 = trajectory["rxpt2"]
            self.zxpt2 = trajectory["zxpt2"]

    def resolve_shapes(
        self, time: float, clip_sharpness: float = 10.0
    ) -> dict[str, float]:
        """Output the shape parameters at a given time"""
        idx = jnp.searchsorted(self.config.shape_times, time, side="right") - 1
        idx = jnp.clip(
            idx, 0, len(self.config.shape_times) - 1
        )  # Ensure idx is within bounds

        shape_dict = {
            "gapin": soft_clip(
                self.gapin[idx],
                self.config.input_ranges["gapin"][0],
                self.config.input_ranges["gapin"][1],
                sharpness=clip_sharpness,
            ),
            "R0": soft_clip(
                self.R0[idx],
                self.config.input_ranges["R0"][0],
                self.config.input_ranges["R0"][1],
                sharpness=clip_sharpness,
            ),
            "rxpt1": soft_clip(
                self.rxpt1[idx],
                self.config.input_ranges["rxpt1"][0],
                self.config.input_ranges["rxpt1"][1],
                sharpness=clip_sharpness,
            ),
            "zxpt1": soft_clip(
                self.zxpt1[idx],
                self.config.input_ranges["zxpt1"][0],
                self.config.input_ranges["zxpt1"][1],
                sharpness=clip_sharpness,
            ),
            "rxpt2": soft_clip(
                self.rxpt2[idx],
                self.config.input_ranges["rxpt2"][0],
                self.config.input_ranges["rxpt2"][1],
                sharpness=clip_sharpness,
            ),
            "zxpt2": soft_clip(
                self.zxpt2[idx],
                self.config.input_ranges["zxpt2"][0],
                self.config.input_ranges["zxpt2"][1],
                sharpness=clip_sharpness,
            ),
        }

        return shape_dict

    def __call__(self, state: "State", inputs: "Inputs") -> tuple[State, Output]:
        # Get the shape at this point in the trajectory
        shape_dict = self.resolve_shapes(inputs.traj_time)

        # Create the input for the profile predictor
        profile_predictor_input = ProfilePredictorInputs(
            Ip=inputs.Ip_MA,
            B0=inputs.B0,
            ne20=inputs.ne20,
            beta=inputs.beta,
            gapin=shape_dict["gapin"],
            R0=shape_dict["R0"],
            rxpt1=shape_dict["rxpt1"],
            zxpt1=shape_dict["zxpt1"],
            rxpt2=shape_dict["rxpt2"],
            zxpt2=shape_dict["zxpt2"],
            psi=jnp.array(self.psigrid),
        )

        # Get the output from the profile predictor
        profile_predictor_output = self.profile_predictor(profile_predictor_input)

        # Create the output for this module
        output = ProfileTrajectoryOptimizer.Output(
            profile_predictor_output=profile_predictor_output,
            psi=self.psigrid,
        )

        # New dummy state
        new_state = ProfileTrajectoryOptimizer.State()

        return new_state, output

    @classmethod
    def init(
        cls,
        config: Config,
        profile_predictor: ProfilePredictor,
        psigrid: Array,
    ):
        return cls(
            config=config,
            profile_predictor=profile_predictor,
            psigrid=tuple(psigrid.tolist()),
        )


class ProfileTrajectoryOptimizerEnv(ModuleTrainingEnv):
    module: ProfileTrajectoryOptimizer
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    @staticmethod
    def create_state(
        observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]
    ) -> ProfileTrajectoryOptimizer.State:
        return ProfileTrajectoryOptimizer.State()

    @staticmethod
    def create_inputs(
        inputs: dict[str, ArrayLike],
    ) -> ProfileTrajectoryOptimizer.Inputs:
        return ProfileTrajectoryOptimizer.Inputs(
            traj_time=inputs["traj_time"].data,
            Ip_MA=inputs["iptipp_MA"].data,
            B0=inputs["B0"].data,
            ne20=inputs["dstdenp"].data / 10,
            beta=inputs["beta"].data,
        )

    def get_trainable(self):
        # Get only the time-dependent controllable parameters
        return {
            "gapin": self.module.gapin,
            "R0": self.module.R0,
            "rxpt1": self.module.rxpt1,
            "zxpt1": self.module.zxpt1,
            "rxpt2": self.module.rxpt2,
            "zxpt2": self.module.zxpt2,
        }
