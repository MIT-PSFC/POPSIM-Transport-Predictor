import chex
import equinox as eqx
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from popsim_transport_predictor.modules.profile_predictor import (
    ProfilePredictorInputs,
    ProfilePredictorOutputs,
)


class ProfileTrajectoryOptimizer(TimeDepModule):
    """Module for optimizing a desired trajectory of plasma profiles"""

    profile_predictor: eqx.Module
    psigrid: tuple = eqx.field(static=True)

    # These are the things that we can control over time
    # TODO(ZanderKeith) might be worthwhile to have a small transition to be more DIII-D-like
    R0: Array[float]  # Major radius [m]
    a_minor: Array[float]  # Minor radius [m]
    kappa: Array[float]  # Elongation [-]
    delta_top: Array[float]  # Upper triangularity [-]
    delta_bottom: Array[float]  # Lower triangularity [-]

    @chex.dataclass
    class Config:
        shape_times: (
            ArrayLike  # The times at which we specify the desired profile shapes [s]
        )
        input_ranges: dict[
            str, tuple[float, float]
        ]  # The ranges for the input parameters during the trajectory
        # TODO(ZanderKeith) talk to Allen about the best way to implement this
        # I think that including these ranges should prevent the autodiff from finding a gradient that pushes it out of range
        # because if you put it into a softclip it kinda makes a wall that the input can't get nudged into

    @chex.dataclass
    class State:
        # There aren't any state variables for this module, the profile predictor is stateless
        # All we need to care about is the present time and go to the right place in the input trajectories
        time_state: float  # This is just a dummy thing

    @chex.dataclass
    class Inputs:
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
        profile_predictor: eqx.Module,
        psigrid: tuple,
    ):
        self.profile_predictor = profile_predictor
        self.psigrid = psigrid

    def __call__(self, state: "State", inputs: "Inputs") -> tuple[State, Output]:
        # Get the current time
        t = state.time_state

        # Get the current input parameters based on the time and the input trajectories
        R0_t = self.R0(t)
        a_minor_t = self.a_minor(t)
        kappa_t = self.kappa(t)
        delta_top_t = self.delta_top(t)
        delta_bottom_t = self.delta_bottom(t)

        # Create the input for the profile predictor
        profile_predictor_input = ProfilePredictorInputs(
            Ip_MA=inputs.Ip_MA,
            B0=inputs.B0,
            ne20_edge=inputs.ne20_edge,
            beta=inputs.beta,
            R0=R0_t,
            a_minor=a_minor_t,
            kappa=kappa_t,
            delta_top=delta_top_t,
            delta_bottom=delta_bottom_t,
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
        new_state = ProfileTrajectoryOptimizer.State(time_state=t + 0.001)

        return new_state, output


class ProfileTrajectoryOptimizerEnv(ModuleTrainingEnv):
    module: ProfileTrajectoryOptimizer
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    def get_trainable(self):
        # Get only the time-dependent controllable parameters
        return {
            "R0": self.module.R0,
            "a_minor": self.module.a_minor,
            "kappa": self.module.kappa,
            "delta": self.module.delta,
        }
