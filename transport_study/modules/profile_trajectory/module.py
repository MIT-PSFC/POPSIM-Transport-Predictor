import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule
from popsim.math_utils import soft_clip
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from transport_study.datasets.d3d.d3d_dataset import INNER_WALL
from transport_study.modules.profile_predictor.module import (
    Inputs as ProfilePredictorInputs,
)
from transport_study.modules.profile_predictor.module import (
    Outputs as ProfilePredictorOutputs,
)
from transport_study.modules.profile_predictor.module import (
    ProfilePredictor,
)


@chex.dataclass
class PCSInputMapper:
    """Goes from the things we can control to the inputs the profile predictor needs
    NOTE: The DIII-D PCS ABSOLUTELY MUST BE IN THE PROPER CONTROL MODE
    otherwise this mapping from inputs to what the plasma does will be entirely different!

    See transport_study/tests/datasets/test_d3d_dataset_prof.py for an example of verifying that these line up
    """

    R0: Array  # Geometric major radius [m]
    gapin: Array  # Inner gap [m]
    rxpt1: Array  # Lower X-point R [m]
    zxpt1: Array  # Lower X-point Z [m]
    rxpt2: Array  # Upper X-point R [m]
    zxpt2: Array  # Upper X-point Z [m]

    @property
    def a_minor(self):
        """Compute the minor radius from the inputs"""
        return self.R0 - self.gapin - INNER_WALL

    @property
    def kappa(self):
        """Assuming in X-point control!"""
        return jnp.abs(self.zxpt2 - self.zxpt1) / (self.a_minor * 2)

    @property
    def delta_bot(self):
        """The bottom X point is on the LCFS, and we can get the triangularity from how far in it is radially"""
        return (self.R0 - self.rxpt1) / self.a_minor

    @property
    def delta_top(self):
        """The top X point is on the LCFS, and we can get the triangularity from how far in it is radially"""
        return (self.R0 - self.rxpt2) / self.a_minor


class ProfileTrajectoryOptimizer(TimeDepModule):
    """Module for optimizing a desired trajectory of plasma profiles"""

    config: "Config"

    profile_predictor: eqx.Module
    psigrid: tuple = eqx.field(static=True)

    # These are the things that we can control over time
    # Note that the PCS needs to be in the proper control mode for these to actually line up
    # in shots 201927 and 206364, they do
    R0: Array  # Major radius [m]
    gapin: Array  # Inner gap [m]
    rxpt1: Array  # Bottom X-point R [m]
    zxpt1: Array  # Bottom X-point Z [m]
    rxpt2: Array  # Top X-point R [m]
    zxpt2: Array  # Top X-point Z [m]
    ne20_edge: Array  # Edge density [10^20 m^-3]

    @chex.dataclass
    class Config:
        traj_times: (
            Array  # The times at which we specify the our desired trajectory [s]
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
        betan: float  # Plasma beta [%]

    @chex.dataclass
    class Output:
        # Things that are necessary for computing the loss function
        profile_predictor_output: ProfilePredictorOutputs
        psi: Array
        q_star: float  # Edge safety factor proxy for q_min [~]
        fGW: float  # Greenwald density fraction [~]
        R0: float  # Major radius [m], needed for effective collisionality

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

        num_times = len(config.traj_times)

        if trajectory is None:
            trajectory = {}

        for input_name in [
            "R0",
            "gapin",
            "rxpt1",
            "zxpt1",
            "rxpt2",
            "zxpt2",
            "ne20_edge",
        ]:
            if input_name not in trajectory:
                trajectory[input_name] = (
                    jnp.ones(num_times)
                    * (
                        config.input_ranges[input_name][0]
                        + config.input_ranges[input_name][1]
                    )
                    / 2
                )

        self.R0 = trajectory["R0"]
        self.gapin = trajectory["gapin"]
        self.rxpt1 = trajectory["rxpt1"]
        self.zxpt1 = trajectory["zxpt1"]
        self.rxpt2 = trajectory["rxpt2"]
        self.zxpt2 = trajectory["zxpt2"]
        self.ne20_edge = trajectory["ne20_edge"]

    def resolve_targets(
        self, time: float, clip_sharpness: float = 10.0
    ) -> dict[str, float]:
        """Output the target parameters at a given time"""
        idx = jnp.searchsorted(self.config.traj_times, time, side="right") - 1
        idx = jnp.clip(
            idx, 0, len(self.config.traj_times) - 1
        )  # Ensure idx is within bounds

        targ_dict = {
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
            "ne20_edge": soft_clip(
                self.ne20_edge[idx],
                self.config.input_ranges["ne20_edge"][0],
                self.config.input_ranges["ne20_edge"][1],
                sharpness=clip_sharpness,
            ),
        }

        return targ_dict

    def __call__(self, state: "State", inputs: "Inputs") -> tuple[State, Output]:
        # Get the target parameters at this point in the trajectory
        targ_dict = self.resolve_targets(inputs.traj_time)

        # Map PCS control inputs to shape parameters
        pcs_inputs = PCSInputMapper(
            R0=targ_dict["R0"],
            gapin=targ_dict["gapin"],
            rxpt1=targ_dict["rxpt1"],
            zxpt1=targ_dict["zxpt1"],
            rxpt2=targ_dict["rxpt2"],
            zxpt2=targ_dict["zxpt2"],
        )

        # Create the input for the profile predictor
        profile_predictor_input = ProfilePredictorInputs(
            Ip=inputs.Ip_MA,
            B0=inputs.B0,
            betan=inputs.betan,
            ne20=targ_dict["ne20_edge"],
            R0=targ_dict["R0"],
            a_minor=pcs_inputs.a_minor,
            kappa=pcs_inputs.kappa,
            delta_top=pcs_inputs.delta_top,
            delta_bot=pcs_inputs.delta_bot,
            psi=jnp.array(self.psigrid),
        )

        # Get the output from the profile predictor
        profile_predictor_output = self.profile_predictor(profile_predictor_input)

        # Create the output for this module
        output = ProfileTrajectoryOptimizer.Output(
            profile_predictor_output=profile_predictor_output,
            psi=self.psigrid,
            q_star=profile_predictor_input.q_star,
            fGW=profile_predictor_input.fGW,
            R0=targ_dict["R0"],
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
        trajectory: dict[str, Array] | None = None,
    ):
        if not isinstance(psigrid, tuple):
            psigrid = tuple(psigrid.tolist())

        return cls(
            config=config,
            profile_predictor=profile_predictor,
            psigrid=psigrid,
            trajectory=trajectory,
        )


class ProfileTrajectoryOptimizerEnv(ModuleTrainingEnv):
    module: ProfileTrajectoryOptimizer
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)
    optimize_density: bool = eqx.field(static=True, default=False)

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
            Ip_MA=inputs["Ip_MA_prog"].data,
            B0=inputs["B0_prog"].data,
            betan=inputs["betan_prog"].data,
        )

    def get_trainable(self):
        # Get only the time-dependent controllable parameters
        trainable = {
            "R0": self.module.R0,
            "gapin": self.module.gapin,
            "rxpt1": self.module.rxpt1,
            "zxpt1": self.module.zxpt1,
            "rxpt2": self.module.rxpt2,
            "zxpt2": self.module.zxpt2,
        }
        if self.optimize_density:
            trainable["ne20_edge"] = self.module.ne20_edge
        return trainable
