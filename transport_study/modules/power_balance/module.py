import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule, discrete_no_save_field
from popsim.math_utils import smooth_clamp
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from transport_study.modules.normalization import InputNormalizer
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower

MIN_TAUE = 0.001  # Default minimum reasonable value for tau_e [s]
MAX_TAUE = 0.8  # Maximum reasonable value for tau_e [s]

# Floor smoothing width of the scaling-law tau_e clamp [s], kept narrow so
# physically common low tau_e (20-50 ms) passes through undistorted
TAUE_CLAMP_MIN_WIDTH = 0.001

# Sharpness of the sigmoid L-H mode blend in the scaling law
# a hard jnp.where switch would zero the gradient of the P_LH threshold coefficients
LH_BLEND_SHARPNESS = 8.0

MIN_POWER = -32  # Minimum reasonable value for conducted power (dW/dt) [MW]
MAX_POWER = 32  # Maximum reasonable value for conducted power (dW/dt) [MW]
MIN_WTOT_MJ = 0.001  # Floor for predicted stored energy [MJ], keeps Wtot strictly positive

# Smoothing width of every smooth_clamp bound, as a fraction of the bound range.
# smooth_clamp is identity inside (min+width, max-width) and its gradient tail
# decays as exp(-overshoot/width), staying alive ~10x further out than tanh soft_clip
BOUND_CLAMP_WIDTH_FRAC = 0.1


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
        arr = jnp.array([inp.Ip_MA, inp.B0, inp.R0, inp.a_minor, inp.kappa, inp.ne20, inp.P_aux_MW])
        nn_out = self.nn(arr)
        width = BOUND_CLAMP_WIDTH_FRAC * (self.max_val - self.min_val)
        bounded_out = smooth_clamp(nn_out, self.min_val, self.max_val, width, width).squeeze()

        output = TauePredictorOutputs(
            taue_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),
            },
        )

        return output


class ScalingLawPredictor(eqx.Module):
    """Predict tau_e with a scaling law"""

    # Trainable coefficient leaves, NOT static
    # get_trainable selects them so the scaling law fits its coefficients to data
    scaling_lmode: dict[str, Array]
    scaling_hmode: dict[str, Array]
    scaling_lh_transition: dict[str, Array]

    min_taue: float = eqx.field(static=True)
    max_taue: float = eqx.field(static=True)
    isotope_mass: float = eqx.field(static=True, default=2)  # Assume DD operation, the source devices carry no per-shot isotope signal

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

        # Needed for the scaling law to calculate P_abs
        # This MUST come from the P_oh submodule and NOT from the training dataset!
        # otherwise that's giving this model more information and that's cheating!
        P_oh_MW: float

        @property
        def epsilon(self):
            return self.a_minor / self.R0

        @property
        def P_abs_MW(self):
            return self.P_aux_MW + self.P_oh_MW

    def __init__(
        self,
        scaling_lmode: dict[str, ArrayLike] | None = None,
        scaling_hmode: dict[str, ArrayLike] | None = None,
        scaling_lh_transition: dict[str, ArrayLike] | None = None,
        min_taue: float | None = None,
        max_taue: float | None = None,
    ):
        # Cast to float arrays so every coefficient is an inexact-array leaf
        # (python float or int values would be dropped by eqx.is_inexact_array)
        scaling_lmode = scaling_lmode if scaling_lmode is not None else self.create_iter89()
        scaling_hmode = scaling_hmode if scaling_hmode is not None else self.create_ipb98()
        scaling_lh_transition = scaling_lh_transition if scaling_lh_transition is not None else self.create_iter1996()
        self.scaling_lmode = {k: jnp.asarray(v, dtype=float) for k, v in scaling_lmode.items()}
        self.scaling_hmode = {k: jnp.asarray(v, dtype=float) for k, v in scaling_hmode.items()}
        self.scaling_lh_transition = {k: jnp.asarray(v, dtype=float) for k, v in scaling_lh_transition.items()}
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

        # Smooth blend so gradient reaches the scaling_lh_transition coefficients
        # (jnp.where is piecewise constant in p_thresh, threshold could never train)
        lh_weight = jax.nn.sigmoid(LH_BLEND_SHARPNESS * (P_abs_MW / p_thresh - 1.0))
        taue = (1.0 - lh_weight) * taue_lmode + lh_weight * taue_hmode

        # The clamp input here is the PHYSICAL scaling-law tau_e, so the floor
        # smoothing must stay narrow: device tau_e commonly sits at 20-50 ms,
        # which a range-fraction min width would distort by tens of percent.
        # (BoundedNNPredictor keeps symmetric widths, its clamp input is a raw
        # NN output with no physical meaning)
        width_max = BOUND_CLAMP_WIDTH_FRAC * (self.max_taue - self.min_taue)
        bounded = smooth_clamp(taue, self.min_taue, self.max_taue, TAUE_CLAMP_MIN_WIDTH, width_max)
        taue_pred = bounded.squeeze()

        out = TauePredictorOutputs(
            taue_pred=taue_pred,
            debug_info={
                "p_thresh": p_thresh,
                "taue_lmode": taue_lmode,
                "taue_hmode": taue_hmode,
                "lh_weight": lh_weight,
            },
        )

        return out

    @classmethod
    def create_ipb98(cls) -> dict[str, ArrayLike]:
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
    def create_iter89(cls):
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
    """Base for the time-dependent stored-energy models.

    Inputs are always in PHYSICAL units plus the device index
    Each concrete model normalizes for its neural networks internally
    (see transport_study.modules.normalization)
    physics pieces like the scaling laws and the Wtot/tau_e power balance consume the physical values.
    """

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
        ne20_line_avg: float
        P_aux_MW: float
        # Device index selecting per-device normalization statistics
        ds_source_idx: float

        def to_normalizer_inputs(self) -> InputNormalizer.Inputs:
            return InputNormalizer.Inputs(
                Ip_MA=self.Ip_MA,
                B0=self.B0,
                R0=self.R0,
                a_minor=self.a_minor,
                kappa=self.kappa,
                ne20_line_avg=self.ne20_line_avg,
                P_aux_MW=self.P_aux_MW,
                ds_source_idx=self.ds_source_idx,
            )

    @chex.dataclass
    class Output:
        Wtot_MJ_pred: float
        P_cond_MW: float
        taue_predictor_output: TauePredictorOutputs
        # Submodule predictions, exposed so the training loss can anchor them
        # to the measured signals. NaN for model types without submodules.
        P_oh_MW_pred: float = float("nan")
        P_rad_MW_pred: float = float("nan")

    @staticmethod
    def positive_wtot(wtot_mj: ArrayLike) -> ArrayLike:
        """Stored energy floored to MIN_WTOT_MJ, used for outputs and physics terms.

        The Euler stepper can integrate the raw state below zero
        (nothing bounds dW/dt against it), so every consumer of the
        state goes through this floor instead of reading it raw.
        """
        return jnp.maximum(wtot_mj, MIN_WTOT_MJ)

    @staticmethod
    def bound_wtot_dot(wtot_mj: ArrayLike, wtot_mj_dot: ArrayLike) -> ArrayLike:
        """The single dW/dt path every model type routes through.

        Smooth-clamps the raw dW/dt to [MIN_POWER, MAX_POWER]
        (identity in the interior, only the edges smooth),
        then blocks further decrease once the integrated state is
        at the Wtot floor so the raw state cannot run away to large negative values.
        The clamp is the same physical prior for all model types:
        the structured models can otherwise emit unbounded dW/dt
        (P_cond at the tau_e floor, the unbounded softplus P_oh/P_rad submodules).
        """
        width = BOUND_CLAMP_WIDTH_FRAC * (MAX_POWER - MIN_POWER)
        bounded = smooth_clamp(wtot_mj_dot, MIN_POWER, MAX_POWER, width, width)
        return jnp.where(wtot_mj <= MIN_WTOT_MJ, jnp.maximum(bounded, 0.0), bounded)


class PowerBalanceScalingLaw(PowerBalance):
    taue_predictor: ScalingLawPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower

    @classmethod
    def init(
        cls,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        scaling_lmode: dict[str, ArrayLike] | None = None,
        scaling_hmode: dict[str, ArrayLike] | None = None,
        scaling_lh_transition: dict[str, ArrayLike] | None = None,
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

    def __call__(self, state: PowerBalance.State, inputs: PowerBalance.Inputs) -> tuple[PowerBalance.State, PowerBalance.Output]:
        # Submodules take physical inputs and normalize internally with their own stats
        normalizer_inputs = inputs.to_normalizer_inputs()
        p_oh_predictor_output = self.p_oh_predictor(normalizer_inputs)
        p_rad_predictor_output = self.p_rad_predictor(normalizer_inputs)

        # The scaling law is dimensional physics, it MUST see physical units.
        taue_predictor_inputs = ScalingLawPredictor.Inputs(
            Ip_MA=inputs.Ip_MA,
            B0=inputs.B0,
            R0=inputs.R0,
            a_minor=inputs.a_minor,
            kappa=inputs.kappa,
            ne20=inputs.ne20_line_avg,
            P_aux_MW=inputs.P_aux_MW,
            P_oh_MW=p_oh_predictor_output.P_oh_MW_pred,
        )
        taue_predictor_output = self.taue_predictor(taue_predictor_inputs)

        taue_pred = taue_predictor_output.taue_pred
        Wtot_MJ = self.positive_wtot(state.Wtot_MJ)
        P_cond_MW = Wtot_MJ / taue_pred
        P_rad_MW = p_rad_predictor_output.P_rad_MW_pred
        P_oh_MW = p_oh_predictor_output.P_oh_MW_pred

        P_abs_MW = inputs.P_aux_MW + P_oh_MW

        Wtot_MJ_dot = self.bound_wtot_dot(state.Wtot_MJ, P_abs_MW - P_cond_MW - P_rad_MW)

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=Wtot_MJ,
            P_cond_MW=P_cond_MW,
            taue_predictor_output=taue_predictor_output,
            P_oh_MW_pred=P_oh_MW,
            P_rad_MW_pred=P_rad_MW,
        )
        return state_dot, output


class PowerBalanceSciML(PowerBalance):
    taue_predictor: BoundedNNPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower
    normalizer: InputNormalizer

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        normalizer: InputNormalizer,
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
            normalizer=normalizer,
        )

    def __call__(self, state: PowerBalance.State, inputs: PowerBalance.Inputs) -> tuple[PowerBalance.State, PowerBalance.Output]:
        # Submodules take physical inputs and normalize internally with their own stats
        normalizer_inputs = inputs.to_normalizer_inputs()
        p_oh_predictor_output = self.p_oh_predictor(normalizer_inputs)
        p_rad_predictor_output = self.p_rad_predictor(normalizer_inputs)

        # The tau_e NN sees this model's own normalized features
        features = self.normalizer(normalizer_inputs)
        taue_predictor_inputs = BoundedNNPredictor.Inputs(
            Ip_MA=features.Ip_MA,
            B0=features.B0,
            R0=features.R0,
            a_minor=features.a_minor,
            kappa=features.kappa,
            ne20=features.ne20_line_avg,
            P_aux_MW=features.P_aux_MW,
        )
        taue_predictor_output = self.taue_predictor(taue_predictor_inputs)

        taue_pred = taue_predictor_output.taue_pred
        Wtot_MJ = self.positive_wtot(state.Wtot_MJ)
        P_cond_MW = Wtot_MJ / taue_pred
        P_rad_MW = p_rad_predictor_output.P_rad_MW_pred
        P_oh_MW = p_oh_predictor_output.P_oh_MW_pred

        P_abs_MW = inputs.P_aux_MW + P_oh_MW

        Wtot_MJ_dot = self.bound_wtot_dot(state.Wtot_MJ, P_abs_MW - P_cond_MW - P_rad_MW)

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=Wtot_MJ,
            P_cond_MW=P_cond_MW,
            taue_predictor_output=taue_predictor_output,
            P_oh_MW_pred=P_oh_MW,
            P_rad_MW_pred=P_rad_MW,
        )
        return state_dot, output


class PowerBalanceUnstructuredNN(PowerBalance):
    nn: eqx.Module
    normalizer: InputNormalizer

    def __call__(self, state: PowerBalance.State, inputs: PowerBalance.Inputs) -> tuple[PowerBalance.State, PowerBalance.Output]:
        features = self.normalizer(inputs.to_normalizer_inputs())
        nn_out = self.nn(features.to_vec())
        Wtot_MJ_dot = self.bound_wtot_dot(state.Wtot_MJ, nn_out.squeeze())

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        output = PowerBalance.Output(
            Wtot_MJ_pred=self.positive_wtot(state.Wtot_MJ),
            P_cond_MW=jnp.nan,  # Not predicted in this model
            taue_predictor_output=TauePredictorOutputs(taue_pred=jnp.nan, debug_info={"nn_out": nn_out.squeeze()}),
        )

        return state_dot, output

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        normalizer: InputNormalizer,
        prng_seed: int = 42,
    ) -> "PowerBalanceUnstructuredNN":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, normalizer=normalizer)


class PowerBalanceTransformer(PowerBalance):
    """Purely data-driven dW/dt predictor with recurrent causal attention.

    A rolling buffer of the last history_len predicted Wtot values
    is carried in the module State as a DISCRETE field
    The simple-Euler stepper integrates only continuous state (Wtot_MJ)
    and passes discrete fields through as the next state directly
    (see popsim.simulate._single_step and popsim.modules.delay.DelayBuffer for the pattern),
    so the buffer update is an exact discrete shift
    no_save keeps the (history_len,) buffer out of the recorded simulation output.

    Each step: shift the current (floored) Wtot state into the buffer,
    embed the normalized 7-vector to a query token,
    embed each buffered Wtot value plus a learned per-slot position embedding to key/value tokens,
    attend (causal by construction, the buffer only ever contains current and past predictions),
    then a residual connection and an MLP head produce a bounded Wtot_MJ_dot.
    Scalar Wtot tokens are indistinguishable beyond their value,
    so without the position embedding attention would be
    permutation-invariant over the history and unable to read trends
    """

    normalizer: InputNormalizer
    feature_embed: eqx.nn.Linear
    wtot_embed: eqx.nn.Linear
    pos_embed: Array
    attention: eqx.nn.MultiheadAttention
    head: eqx.nn.MLP
    history_len: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    @chex.dataclass
    class State:
        Wtot_MJ: float
        # (history_len,) past predicted Wtot_MJ values, most recent last
        history: Array = discrete_no_save_field(default=None)

    def __call__(self, state: "PowerBalanceTransformer.State", inputs: PowerBalance.Inputs) -> tuple:
        features = self.normalizer(inputs.to_normalizer_inputs())
        query = self.feature_embed(features.to_vec())

        # Shift the buffer by one and insert the current predicted Wtot at the end
        wtot_now = self.positive_wtot(state.Wtot_MJ)
        new_history = jnp.concatenate([state.history[1:], wtot_now[None]])

        tokens = jax.vmap(self.wtot_embed)(new_history[:, None]) + self.pos_embed
        attn_out = self.attention(query[None, :], tokens, tokens)[0]
        latent = query + attn_out
        nn_out = self.head(latent)
        Wtot_MJ_dot = self.bound_wtot_dot(state.Wtot_MJ, nn_out.squeeze())

        state_out = PowerBalanceTransformer.State(Wtot_MJ=Wtot_MJ_dot, history=new_history)
        output = PowerBalance.Output(
            Wtot_MJ_pred=wtot_now,
            P_cond_MW=jnp.nan,  # Not predicted in this model
            taue_predictor_output=TauePredictorOutputs(taue_pred=jnp.nan, debug_info={"nn_out": nn_out.squeeze()}),
        )
        return state_out, output

    @classmethod
    def init(
        cls,
        d_model: int,
        num_heads: int,
        history_len: int,
        nn_width: int,
        nn_depth: int,
        normalizer: InputNormalizer,
        prng_seed: int = 42,
    ) -> "PowerBalanceTransformer":
        key_embed, key_wtot, key_pos, key_attn, key_head = jax.random.split(jax.random.PRNGKey(prng_seed), 5)
        feature_embed = eqx.nn.Linear(7, d_model, key=key_embed)
        wtot_embed = eqx.nn.Linear(1, d_model, key=key_wtot)
        # Small random init breaks slot symmetry when the buffer holds a constant history
        # (the seeded state at t0)
        pos_embed = 0.02 * jax.random.normal(key_pos, (history_len, d_model))
        attention = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=d_model, key=key_attn)
        head = eqx.nn.MLP(
            in_size=d_model,
            out_size=1,
            width_size=nn_width,
            depth=nn_depth,
            key=key_head,
        )
        return cls(
            normalizer=normalizer,
            feature_embed=feature_embed,
            wtot_embed=wtot_embed,
            pos_embed=pos_embed,
            attention=attention,
            head=head,
            history_len=history_len,
            d_model=d_model,
        )


class PowerBalanceEnv(ModuleTrainingEnv):
    module: PowerBalance
    domain_adaptation: str = eqx.field(static=True, default="unset")
    freeze_submodules: list[str] = eqx.field(static=True, default_factory=list)
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    def create_state(self, observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        Wtot_MJ = observations["Wtot_MJ"].data
        if isinstance(self.module, PowerBalanceTransformer):
            # Seed the Wtot history with the measured t0 value tiled,
            # a constant history rather than a fake all-zero one
            Wtot_MJ = jnp.asarray(Wtot_MJ)
            history = jnp.broadcast_to(Wtot_MJ[..., None], (*Wtot_MJ.shape, self.module.history_len))
            return PowerBalanceTransformer.State(Wtot_MJ=Wtot_MJ, history=history)
        return PowerBalance.State(Wtot_MJ=Wtot_MJ)

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        # Top-level module inputs are always in physical units,
        # normalization happens inside the modules
        if isinstance(inputs, xr.Dataset):
            inputs = {var: inputs[var].data for var in inputs.data_vars}
        return PowerBalance.Inputs(
            Ip_MA=inputs["Ip_MA"],
            B0=inputs["B0"],
            R0=inputs["R0"],
            a_minor=inputs["a_minor"],
            kappa=inputs["kappa"],
            ne20_line_avg=inputs["ne20_line_avg"],
            P_aux_MW=inputs["P_aux_MW"],
            ds_source_idx=inputs["ds_source_idx"],
        )

    def get_trainable(self):
        """Trainable leaves for the optimizer partition.

        Selects NN leaves explicitly, never whole modules
        normalizer statistics are ordinary array leaves on every module and must stay frozen
        (a broad eqx.filter over a module would silently train them)
        """
        if self.domain_adaptation == "transfer":
            last_layer_leaves = []
            if isinstance(self.module, PowerBalanceUnstructuredNN):
                last_layer_leaves += [
                    self.module.nn.layers[-1].weight,
                    self.module.nn.layers[-1].bias,
                ]
            if isinstance(self.module, PowerBalanceTransformer):
                last_layer_leaves += [
                    self.module.head.layers[-1].weight,
                    self.module.head.layers[-1].bias,
                ]
            if isinstance(self.module, PowerBalanceSciML):
                last_layer_leaves += [
                    self.module.taue_predictor.nn.layers[-1].weight,
                    self.module.taue_predictor.nn.layers[-1].bias,
                ]
            if isinstance(self.module, PowerBalanceScalingLaw):
                # No last-layer analog, fine-tune all three coefficient dicts
                taue = self.module.taue_predictor
                for scaling in (taue.scaling_lmode, taue.scaling_hmode, taue.scaling_lh_transition):
                    last_layer_leaves += list(scaling.values())
            if isinstance(self.module, (PowerBalanceSciML, PowerBalanceScalingLaw)):
                if "p_oh_predictor" not in self.freeze_submodules:
                    last_layer_leaves += [
                        self.module.p_oh_predictor.nn.layers[-1].weight,
                        self.module.p_oh_predictor.nn.layers[-1].bias,
                    ]
                if "p_rad_predictor" not in self.freeze_submodules:
                    last_layer_leaves += [
                        self.module.p_rad_predictor.nn.layers[-1].weight,
                        self.module.p_rad_predictor.nn.layers[-1].bias,
                    ]
            return last_layer_leaves

        trainable_leaves = {}
        if isinstance(self.module, (PowerBalanceScalingLaw, PowerBalanceSciML)):
            if "p_oh_predictor" not in self.freeze_submodules:
                trainable_leaves["p_oh_predictor"] = eqx.filter(self.module.p_oh_predictor.nn, eqx.is_inexact_array)
            if "p_rad_predictor" not in self.freeze_submodules:
                trainable_leaves["p_rad_predictor"] = eqx.filter(self.module.p_rad_predictor.nn, eqx.is_inexact_array)

        if isinstance(self.module, PowerBalanceSciML):
            trainable_leaves["taue_predictor"] = eqx.filter(self.module.taue_predictor.nn, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceScalingLaw):
            trainable_leaves["taue_predictor"] = eqx.filter(self.module.taue_predictor, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceUnstructuredNN):
            trainable_leaves["nn"] = eqx.filter(self.module.nn, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceTransformer):
            trainable_leaves["feature_embed"] = eqx.filter(self.module.feature_embed, eqx.is_inexact_array)
            trainable_leaves["wtot_embed"] = eqx.filter(self.module.wtot_embed, eqx.is_inexact_array)
            trainable_leaves["pos_embed"] = self.module.pos_embed
            trainable_leaves["attention"] = eqx.filter(self.module.attention, eqx.is_inexact_array)
            trainable_leaves["head"] = eqx.filter(self.module.head, eqx.is_inexact_array)

        return trainable_leaves
