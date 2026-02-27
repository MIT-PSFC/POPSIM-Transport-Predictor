import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule
from popsim.math_utils import soft_clip
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.module import (
    Inputs as ProfilePredictorInputs,
)
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.module import (
    Outputs as ProfilePredictorOutputs,
)
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.module import (
    ProfilePredictor,
)


class ProfileTrajectoryOptimizer(TimeDepModule):
    """Module for optimizing a desired trajectory of plasma profiles"""

    config: "Config"

    profile_predictor: eqx.Module
    psigrid: tuple = eqx.field(static=True)

    # These are the things that we can control over time
    # TODO(ZanderKeith) might be worthwhile to have a small translation layer to be more DIII-D-like
    # What we are really controlling is R0, GapIn, RxTop, RxBot, ZxTop, ZxBot
    R0: Array  # Major radius [m]
    a_minor: Array  # Minor radius [m]
    kappa: Array  # Elongation [-]
    delta_top: Array  # Upper triangularity [-]
    delta_bottom: Array  # Lower triangularity [-]

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
        ne20_edge: float  # Electron density [10^20 m^-3]
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
            self.R0 = (
                jnp.ones(num_times)
                * (config.input_ranges["R0"][0] + config.input_ranges["R0"][1])
                / 2
            )
            self.a_minor = (
                jnp.ones(num_times)
                * (
                    config.input_ranges["a_minor"][0]
                    + config.input_ranges["a_minor"][1]
                )
                / 2
            )
            self.kappa = (
                jnp.ones(num_times)
                * (config.input_ranges["kappa"][0] + config.input_ranges["kappa"][1])
                / 2
            )
            self.delta_top = (
                jnp.ones(num_times)
                * (
                    config.input_ranges["delta_top"][0]
                    + config.input_ranges["delta_top"][1]
                )
                / 2
            )
            self.delta_bottom = (
                jnp.ones(num_times)
                * (
                    config.input_ranges["delta_bottom"][0]
                    + config.input_ranges["delta_bottom"][1]
                )
                / 2
            )
        else:
            # Load trajectories from the provided dictionary
            self.R0 = trajectory["R0"]
            self.a_minor = trajectory["a_minor"]
            self.kappa = trajectory["kappa"]
            self.delta_top = trajectory["delta_top"]
            self.delta_bottom = trajectory["delta_bottom"]

    def resolve_shapes(
        self, time: float, clip_sharpness: float = 10.0
    ) -> dict[str, float]:
        """Output the shape parameters at a given time"""
        idx = jnp.searchsorted(self.config.shape_times, time, side="right") - 1
        idx = jnp.clip(
            idx, 0, len(self.config.shape_times) - 1
        )  # Ensure idx is within bounds

        shape_dict = {
            "R0": soft_clip(
                self.R0[idx],
                self.config.input_ranges["R0"][0],
                self.config.input_ranges["R0"][1],
                sharpness=clip_sharpness,
            ),
            "a_minor": soft_clip(
                self.a_minor[idx],
                self.config.input_ranges["a_minor"][0],
                self.config.input_ranges["a_minor"][1],
                sharpness=clip_sharpness,
            ),
            "kappa": soft_clip(
                self.kappa[idx],
                self.config.input_ranges["kappa"][0],
                self.config.input_ranges["kappa"][1],
                sharpness=clip_sharpness,
            ),
            "delta_top": soft_clip(
                self.delta_top[idx],
                self.config.input_ranges["delta_top"][0],
                self.config.input_ranges["delta_top"][1],
                sharpness=clip_sharpness,
            ),
            "delta_bottom": soft_clip(
                self.delta_bottom[idx],
                self.config.input_ranges["delta_bottom"][0],
                self.config.input_ranges["delta_bottom"][1],
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
            ne20_edge=inputs.ne20_edge,
            beta=inputs.beta,
            R0=shape_dict["R0"],
            a_minor=shape_dict["a_minor"],
            kappa=shape_dict["kappa"],
            delta_top=shape_dict["delta_top"],
            delta_bottom=shape_dict["delta_bottom"],
            psi=jnp.array(self.psigrid),
        )

        # Get the output from the profile predictor
        profile_predictor_output = self.profile_predictor(profile_predictor_input)

        # Create the output for this module
        output = ProfileTrajectoryOptimizer.Output(
            profile_predictor_output=profile_predictor_output,
            psi=self.psigrid,
        )

        # Update the state (in this case, just increment the time)
        # This is so dumb but I want to get something to try out the other stuff for right now
        # There *should* be a way to get the time from the coords of the inputs and just use that instead of having a separate time state
        # this is so weird, a stateless time-dependent module. You really gotta try using a thing to understand it
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
            Ip_MA=inputs["Ip_MA"].data,
            B0=inputs["B0"].data,
            ne20_edge=inputs["ne20_edge"].data,
            beta=inputs["beta"].data,
        )

    def get_trainable(self):
        # Get only the time-dependent controllable parameters
        return {
            "R0": self.module.R0,
            "a_minor": self.module.a_minor,
            "kappa": self.module.kappa,
            "delta_top": self.module.delta_top,
            "delta_bottom": self.module.delta_bottom,
        }
