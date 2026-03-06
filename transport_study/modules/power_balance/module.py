import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import ArrayLike
from popsim import TimeDepModule
from popsim.math_utils import soft_clip
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower

MIN_TAUE = 0.001  # Default inimum reasonable value for tau_e [s]
MAX_TAUE = 0.3  # Maximum reasonable value for tau_e [s]
MIN_POWER = -20  # Minimum reasonable value for power (dW/dt) [MW]
MAX_POWER = 20  # Maximum reasonable value for power (dW/dt) [MW]


@chex.dataclass
class TauePredictorOutputs:
    taue_pred: float  # [s]
    debug_info: dict | None = None


class BoundedNNPredictor(eqx.Module):
    """Predict tau_e with a bounded NN"""

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    @chex.dataclass
    class Inputs:
        """Inputs for NN predictor"""

        Ip_MA: float
        B0: float
        R0: float
        a_minor: float
        kappa: float
        ne20: float
        P_aux_MW: float

    def __call__(self, inp: "Inputs") -> TauePredictorOutputs:
        arr = jnp.array(
            [inp.Ip_MA, inp.B0, inp.R0, inp.a_minor, inp.kappa, inp.ne20, inp.P_aux_MW]
        )
        nn_out = self.nn(arr)
        bounded_out = soft_clip(
            nn_out, self.min_val, self.max_val, sharpness=2
        ).squeeze()

        output = TauePredictorOutputs(
            taue_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),  # Squeeze to match dimensions with taue_pred
            },
        )

        return output


class ScalingLawPredictor(eqx.Module):
    """Predict tau_e with a scaling law"""

    scaling_lmode: dict[str, float] = eqx.field(static=True)
    scaling_hmode: dict[str, float] = eqx.field(static=True)
    scaling_lh_transition: dict[str, float] = eqx.field(static=True)

    min_taue: float = eqx.field(static=True)
    max_taue: float = eqx.field(static=True)
    isotope_mass: float = eqx.field(
        static=True, default=2
    )  # Assuming DD operation for now (TODO(ZanderKeith) does this signal exist for source devices?)

    @chex.dataclass
    class Inputs:
        """Inputs for scaling law predictor, MUST BE IN REAL UNITS"""

        Ip_MA: float  # [MA]
        B0: float  # On axis magnetic field [T]
        R0: float  # Major radius [m]
        a_minor: float  # Minor radius [m]
        kappa: float  # Elongation [-]
        ne20: float  # Electron density [10^20 m^-3]
        P_aux_MW: float
        # Only ever used for the scaling law to calculate P_abs
        P_oh_MW: float

        @property
        def epsilon(self):
            return self.a_minor / self.R0

        @property
        def P_abs_MW(self):
            return self.P_aux_MW + self.P_oh_MW

    def __init__(
        self,
        scaling_lmode: dict[str, float] | None = None,
        scaling_hmode: dict[str, float] | None = None,
        scaling_lh_transition: dict[str, float] | None = None,
        min_taue: float | None = None,
        max_taue: float | None = None,
    ):
        self.scaling_lmode = (
            scaling_lmode if scaling_lmode is not None else self.create_ipb98()
        )
        self.scaling_hmode = (
            scaling_hmode if scaling_hmode is not None else self.create_ibp89()
        )
        self.scaling_lh_transition = (
            scaling_lh_transition
            if scaling_lh_transition is not None
            else self.create_iter1996()
        )
        self.min_taue = MIN_TAUE if min_taue is None else min_taue
        self.max_taue = MAX_TAUE if max_taue is None else max_taue

    def __call__(self, inp: Inputs) -> TauePredictorOutputs:
        # Ensure each of the input values is strictly greater than 0.001 to avoid numerical instability.
        Ip_MA = jnp.clip(inp.Ip_MA, 0.001, None)
        B0 = jnp.clip(inp.B0, 0.001, None)
        ne19 = jnp.clip(inp.ne20 / 10, 0.001, None)
        ne20 = jnp.clip(inp.ne20, 0.001, None)
        P_abs_MW = jnp.clip(inp.P_abs_MW, 0.001, None)
        kappa = jnp.clip(inp.kappa, 0.001, None)
        epsilon = jnp.clip(inp.epsilon, 0.001, None)
        R0 = jnp.clip(inp.R0, 0.001, None)

        p_thresh = self.scaling_lh_transition["coeff"] * (
            (ne20 ** self.scaling_lh_transition["alpha_N"])
            * (B0 ** self.scaling_lh_transition["alpha_B"])
            * (R0 ** self.scaling_lh_transition["alpha_R"])
        )

        taue_lmode = self.scaling_lmode["coeff"] * (
            (Ip_MA ** self.scaling_lmode["alpha_I"])
            * (B0 ** self.scaling_lmode["alpha_B"])
            * (ne19 ** self.scaling_lmode["alpha_N"])
            * (P_abs_MW ** self.scaling_lmode["alpha_P"])
            * (R0 ** self.scaling_lmode["alpha_R"])
            * (kappa ** self.scaling_lmode["alpha_kappa"])
            * (epsilon ** self.scaling_lmode["alpha_epsilon"])
            * (self.isotope_mass ** self.scaling_lmode["alpha_mass"])
        )

        taue_hmode = self.scaling_hmode["coeff"] * (
            (Ip_MA ** self.scaling_hmode["alpha_I"])
            * (B0 ** self.scaling_hmode["alpha_B"])
            * (ne19 ** self.scaling_hmode["alpha_N"])
            * (P_abs_MW ** self.scaling_hmode["alpha_P"])
            * (R0 ** self.scaling_hmode["alpha_R"])
            * (kappa ** self.scaling_hmode["alpha_kappa"])
            * (epsilon ** self.scaling_hmode["alpha_epsilon"])
            * (self.isotope_mass ** self.scaling_hmode["alpha_mass"])
        )

        taue = jnp.where(P_abs_MW < p_thresh, taue_lmode, taue_hmode)

        # Softmax output
        bounded = soft_clip(taue, self.min_taue, self.max_taue, sharpness=10)
        taue_pred = bounded.squeeze()

        out = TauePredictorOutputs(
            taue_pred=taue_pred,
            debug_info={
                "p_thresh": p_thresh,
                "taue_lmode": taue_lmode,
                "taue_hmode": taue_hmode,
            },
        )

        return out

    @classmethod
    def create_ipb98(cls) -> dict[str, float]:
        scaling = {
            "coeff": jnp.array(56.2 * 10**-3),
            "alpha_I": jnp.array(0.93),
            "alpha_B": jnp.array(0.15),
            "alpha_N": jnp.array(0.41),
            "alpha_P": jnp.array(-0.69),
            "alpha_R": jnp.array(1.97),
            "alpha_kappa": jnp.array(0.78),
            "alpha_epsilon": jnp.array(0.58),
            "alpha_mass": jnp.array(0.19),
        }
        return scaling

    @classmethod
    def create_ibp89(cls):
        scaling = {
            "coeff": jnp.array(38 * 10**-3),
            "alpha_I": jnp.array(0.85),
            "alpha_B": jnp.array(0.2),
            "alpha_N": jnp.array(0.1),
            "alpha_P": jnp.array(-0.5),
            "alpha_R": jnp.array(1.5),
            "alpha_kappa": jnp.array(0.5),
            "alpha_epsilon": jnp.array(0.3),
            "alpha_mass": jnp.array(0.5),
        }
        return scaling

    @classmethod
    def create_iter1996(cls):
        # https://ukaea.github.io/PROCESS/physics-models/plasma_h_mode/
        # NOTE: This scaling law uses ne20 instead of ne19
        scaling = {
            "coeff": 0.45,
            "alpha_N": 0.75,
            "alpha_B": 1,
            "alpha_R": 2,
        }
        return scaling


class PowerBalance(TimeDepModule):
    @chex.dataclass
    class State:
        Wtot_MJ: float

    @chex.dataclass
    class Inputs:
        # Real-valued inputs in physical units
        Ip_MA: float
        B0: float
        R0: float
        a_minor: float
        kappa: float
        ne20: float
        P_aux_MW: float
        # Things that might get put into a neural network
        Ip_MA_nn: float
        B0_nn: float
        R0_nn: float
        a_minor_nn: float
        kappa_nn: float
        ne20_nn: float
        P_aux_nn: float

    @chex.dataclass
    class Output:
        Wtot_MJ_pred: float
        P_cond_MW: float
        taue_predictor_output: TauePredictorOutputs


class PowerBalanceScalingLaw(PowerBalance):
    taue_predictor: ScalingLawPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower

    @classmethod
    def init(
        cls,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        scaling_lmode: dict[str, float] | None = None,
        scaling_hmode: dict[str, float] | None = None,
        scaling_lh_transition: dict[str, float] | None = None,
        min_taue: float = MIN_TAUE,
        max_taue: float = MAX_TAUE,
    ) -> "PowerBalanceScalingLaw":
        taue_predictor = ScalingLawPredictor(
            scaling_lmode=scaling_lmode,
            scaling_hmode=scaling_hmode,
            scaling_lh_transition=scaling_lh_transition,
            min_taue=min_taue,
            max_taue=max_taue,
        )
        return cls(
            taue_predictor=taue_predictor,
            p_oh_predictor=p_oh_predictor,
            p_rad_predictor=p_rad_predictor,
        )

    def __call__(
        self, state: PowerBalance.State, inputs: PowerBalance.Inputs
    ) -> tuple[PowerBalance.State, PowerBalance.Output]:
        p_oh_predictor_inputs = OhmicPower.Inputs(
            Ip_MA_real=inputs.Ip_MA,
            Ip_MA_nn=inputs.Ip_MA_nn,
            B0_nn=inputs.B0_nn,
            R0_nn=inputs.R0_nn,
            a_minor_nn=inputs.a_minor_nn,
            kappa_nn=inputs.kappa_nn,
            ne20_nn=inputs.ne20_nn,
            P_aux_nn=inputs.P_aux_nn,
        )
        p_oh_predictor_output = self.p_oh_predictor(p_oh_predictor_inputs)

        p_rad_predictor_inputs = RadiatedPower.Inputs(
            ne20_real=inputs.ne20,
            Ip_MA_nn=inputs.Ip_MA_nn,
            B0_nn=inputs.B0_nn,
            R0_nn=inputs.R0_nn,
            a_minor_nn=inputs.a_minor_nn,
            kappa_nn=inputs.kappa_nn,
            ne20_nn=inputs.ne20_nn,
            P_aux_nn=inputs.P_aux_nn,
        )
        p_rad_predictor_output = self.p_rad_predictor(p_rad_predictor_inputs)

        taue_predictor_inputs = ScalingLawPredictor.Inputs(
            Ip_MA=inputs.Ip_MA_nn,
            B0=inputs.B0_nn,
            R0=inputs.R0_nn,
            a_minor=inputs.a_minor_nn,
            kappa=inputs.kappa_nn,
            ne20=inputs.ne20_nn,
            P_aux_MW=inputs.P_aux_nn,
            P_oh_MW=p_oh_predictor_output.P_oh_MW_pred,
        )
        taue_predictor_output = self.taue_predictor(taue_predictor_inputs)

        taue_pred = taue_predictor_output.taue_pred
        P_cond_MW = state.Wtot_MJ / taue_pred
        P_rad_MW = p_rad_predictor_output.P_rad_MW_pred
        P_oh_MW = p_oh_predictor_output.P_oh_MW_pred

        P_abs_MW = inputs.P_aux_MW + P_oh_MW

        Wtot_MJ_dot = P_abs_MW - P_cond_MW - P_rad_MW

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=state.Wtot_MJ,
            P_cond_MW=P_cond_MW,
            taue_predictor_output=taue_predictor_output,
        )
        return state_dot, output


class PowerBalanceSciML(PowerBalance):
    taue_predictor: BoundedNNPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        min_taue: float | None = None,
        max_taue: float | None = None,
        prng_seed: int = 42,
    ) -> "PowerBalanceSciML":
        min_taue = MIN_TAUE if min_taue is None else min_taue
        max_taue = MAX_TAUE if max_taue is None else max_taue

        taue_predictor = BoundedNNPredictor(
            nn=eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=nn_width,
                depth=nn_depth,
                key=jax.random.PRNGKey(prng_seed),
            ),
            min_val=min_taue,
            max_val=max_taue,
        )
        return cls(
            taue_predictor=taue_predictor,
            p_oh_predictor=p_oh_predictor,
            p_rad_predictor=p_rad_predictor,
        )

    def __call__(
        self, state: PowerBalance.State, inputs: PowerBalance.Inputs
    ) -> tuple[PowerBalance.State, PowerBalance.Output]:
        p_oh_predictor_inputs = OhmicPower.Inputs(
            Ip_MA_real=inputs.Ip_MA,
            Ip_MA_nn=inputs.Ip_MA_nn,
            B0_nn=inputs.B0_nn,
            R0_nn=inputs.R0_nn,
            a_minor_nn=inputs.a_minor_nn,
            kappa_nn=inputs.kappa_nn,
            ne20_nn=inputs.ne20_nn,
            P_aux_nn=inputs.P_aux_nn,
        )
        p_oh_predictor_output = self.p_oh_predictor(p_oh_predictor_inputs)

        p_rad_predictor_inputs = RadiatedPower.Inputs(
            ne20_real=inputs.ne20,
            Ip_MA_nn=inputs.Ip_MA_nn,
            B0_nn=inputs.B0_nn,
            R0_nn=inputs.R0_nn,
            a_minor_nn=inputs.a_minor_nn,
            kappa_nn=inputs.kappa_nn,
            ne20_nn=inputs.ne20_nn,
            P_aux_nn=inputs.P_aux_nn,
        )
        p_rad_predictor_output = self.p_rad_predictor(p_rad_predictor_inputs)

        taue_predictor_inputs = BoundedNNPredictor.Inputs(
            Ip_MA=inputs.Ip_MA_nn,
            B0=inputs.B0_nn,
            R0=inputs.R0_nn,
            a_minor=inputs.a_minor_nn,
            kappa=inputs.kappa_nn,
            ne20=inputs.ne20_nn,
            P_aux_MW=inputs.P_aux_nn,
        )
        taue_predictor_output = self.taue_predictor(taue_predictor_inputs)

        taue_pred = taue_predictor_output.taue_pred
        P_cond_MW = state.Wtot_MJ / taue_pred
        P_rad_MW = p_rad_predictor_output.P_rad_MW_pred
        P_oh_MW = p_oh_predictor_output.P_oh_MW_pred

        P_abs_MW = inputs.P_aux_MW + P_oh_MW

        Wtot_MJ_dot = P_abs_MW - P_cond_MW - P_rad_MW

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=state.Wtot_MJ,
            P_cond_MW=P_cond_MW,
            taue_predictor_output=taue_predictor_output,
        )
        return state_dot, output


class PowerBalanceUnstructuredNN(PowerBalance):
    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    def __call__(
        self, state: PowerBalance.State, inputs: PowerBalance.Inputs
    ) -> tuple[PowerBalance.State, PowerBalance.Output]:
        arr = jnp.array(
            [
                inputs.Ip_MA_nn,
                inputs.B0_nn,
                inputs.R0_nn,
                inputs.a_minor_nn,
                inputs.kappa_nn,
                inputs.ne20_nn,
                inputs.P_aux_nn,
            ]
        )
        nn_out = self.nn(arr)
        # TODO(ZanderKeith) this should be predicting in beta or something
        # at least all the inputs are in the same range...
        Wtot_MJ_dot = soft_clip(
            nn_out, self.min_val, self.max_val, sharpness=6
        ).squeeze()

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=state.Wtot_MJ,
            P_cond_MW=jnp.nan,  # Not predicted in this model
            taue_predictor_output=TauePredictorOutputs(
                taue_pred=jnp.nan, debug_info={"nn_out": nn_out.squeeze()}
            ),
        )

        return state_dot, output

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        min_val: float | None = None,
        max_val: float | None = None,
        prng_seed: int = 42,
    ) -> "PowerBalanceUnstructuredNN":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        min_val = MIN_POWER if min_val is None else min_val
        max_val = MAX_POWER if max_val is None else max_val
        return cls(nn=nn, min_val=min_val, max_val=max_val)


class PowerBalanceEnv(ModuleTrainingEnv):
    module: PowerBalance
    normalization_method: str = eqx.field(static=True, default="unset")
    freeze_submodules: list[str] = eqx.field(static=True, default_factory=list)
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    @staticmethod
    def create_state(observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        return PowerBalance.State(Wtot_MJ=observations["Wtot_MJ"].data)

    def create_inputs(self, inputs: dict[str, ArrayLike]):
        # TODO(ZanderKeith) this needs to be replaced with a thing where we initialize the module with a transform
        # The inputs to a top-level module should likely ALWAYS be in physical units
        if isinstance(inputs, xr.Dataset):
            inputs = {var: inputs[var].data for var in inputs.data_vars}

        if self.normalization_method == "raw":
            inputs = PowerBalance.Inputs(
                Ip_MA=inputs["Ip_MA"],
                B0=inputs["B0"],
                R0=inputs["R0"],
                a_minor=inputs["a_minor"],
                kappa=inputs["kappa"],
                ne20=inputs["ne20_line_avg"],
                P_aux_MW=inputs["P_aux_MW"],
                Ip_MA_nn=inputs["Ip_MA"],
                B0_nn=inputs["B0"],
                R0_nn=inputs["R0"],
                a_minor_nn=inputs["a_minor"],
                kappa_nn=inputs["kappa"],
                ne20_nn=inputs["ne20_line_avg"],
                P_aux_nn=inputs["P_aux_MW"],
            )
        elif self.normalization_method == "physics":
            inputs = PowerBalance.Inputs(
                Ip_MA=inputs["Ip_MA"],
                B0=inputs["B0"],
                R0=inputs["R0"],
                a_minor=inputs["a_minor"],
                kappa=inputs["kappa"],
                ne20=inputs["ne20_line_avg"],
                P_aux_MW=inputs["P_aux_MW"],
                Ip_MA_nn=inputs["Ip_MA"],
                B0_nn=inputs["q_star"],
                R0_nn=inputs["epsilon"],
                a_minor_nn=inputs["aB0"],
                kappa_nn=inputs["kappa"],
                ne20_nn=inputs["f_G"],
                P_aux_nn=inputs["surface_power_density"],
            )
        elif self.normalization_method == "z_score":
            inputs = PowerBalance.Inputs(
                Ip_MA=inputs["Ip_MA"],
                B0=inputs["B0"],
                R0=inputs["R0"],
                a_minor=inputs["a_minor"],
                kappa=inputs["kappa"],
                ne20=inputs["ne20_line_avg"],
                P_aux_MW=inputs["P_aux_MW"],
                Ip_MA_nn=inputs["Ip_MA_z"],
                B0_nn=inputs["B0_z"],
                R0_nn=inputs["R0_z"],
                a_minor_nn=inputs["a_minor_z"],
                kappa_nn=inputs["kappa_z"],
                ne20_nn=inputs["ne20_line_avg_z"],
                P_aux_nn=inputs["P_aux_MW_z"],
            )
        elif self.normalization_method == "coral":
            inputs = PowerBalance.Inputs(
                Ip_MA=inputs["Ip_MA"],
                B0=inputs["B0"],
                R0=inputs["R0"],
                a_minor=inputs["a_minor"],
                kappa=inputs["kappa"],
                ne20=inputs["ne20_line_avg"],
                P_aux_MW=inputs["P_aux_MW"],
                Ip_MA_nn=inputs["Ip_MA_coral"],
                B0_nn=inputs["B0_coral"],
                R0_nn=inputs["R0_coral"],
                a_minor_nn=inputs["a_minor_coral"],
                kappa_nn=inputs["kappa_coral"],
                ne20_nn=inputs["ne20_line_avg_coral"],
                P_aux_nn=inputs["P_aux_MW_coral"],
            )
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}"
            )
        return inputs

    def get_trainable(self):
        trainable_leaves = {}
        if isinstance(self.module, PowerBalanceScalingLaw) or isinstance(
            self.module, PowerBalanceSciML
        ):
            if "p_oh_predictor" not in self.freeze_submodules:
                trainable_leaves["p_oh_predictor"] = eqx.filter(
                    self.module.p_oh_predictor, eqx.is_inexact_array
                )
            if "p_rad_predictor" not in self.freeze_submodules:
                trainable_leaves["p_rad_predictor"] = eqx.filter(
                    self.module.p_rad_predictor, eqx.is_inexact_array
                )

        if isinstance(self.module, PowerBalanceSciML):
            trainable_leaves["taue_predictor"] = eqx.filter(
                self.module.taue_predictor, eqx.is_inexact_array
            )
        elif isinstance(self.module, PowerBalanceUnstructuredNN):
            trainable_leaves["nn"] = eqx.filter(self.module.nn, eqx.is_inexact_array)

        return trainable_leaves
