import abc
from collections.abc import Mapping

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from popsim import TimeDepModule, discrete_no_save_field
from popsim.math_utils import smooth_clamp
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

from transport_study.modules.normalization import N_FEATURES, InputNormalizer
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower

# Model types with p_oh / p_rad submodule predictors, each trained as its own prereq case
MODEL_TYPES_WITH_SUBMODULES = ("sciml-taue-scalinglaw", "sciml-taue-nn")
# The submodule pseudo-model-types of those prereq cases
SUBMODULE_MODEL_TYPES = ("p_oh", "p_rad")
# Model types the multiobjective case axis applies to: the transformer's head also predicts P_oh and P_rad for the loss to anchor
MULTIOBJECTIVE_MODEL_TYPES = ("transformer",)
# Model types fed their own predicted Wtot, their normalizer is built with_energy
MODEL_TYPES_WITH_ENERGY_INPUT = ("mlp", "transformer")

MIN_TAUE = 0.001  # Minimum reasonable value for tau_e [s]
MAX_TAUE = 0.8  # Maximum reasonable value for tau_e [s]

# Floor smoothing width of the scaling-law tau_e clamp [s], kept narrow so
# physically common low tau_e (20-50 ms) passes through undistorted
TAUE_CLAMP_MIN_WIDTH = 0.001

# Sharpness of the sigmoid L-H mode blend in the scaling law
# a hard jnp.where switch would zero the gradient of the P_LH threshold coefficients
LH_BLEND_SHARPNESS = 8.0

MIN_POWER = -40  # Lower bound on the predicted dW/dt [MW]
MAX_POWER = 40  # Upper bound on the predicted dW/dt [MW]
MIN_WTOT_MJ = 0.001  # Floor for predicted stored energy [MJ], keeps Wtot strictly positive

# Smoothing width of every smooth_clamp bound, as a fraction of the bound range.
# smooth_clamp is identity inside (min+width, max-width) and its gradient tail
# decays as exp(-overshoot/width), staying alive ~10x further out than tanh soft_clip
BOUND_CLAMP_WIDTH_FRAC = 0.1

# Starting coefficients of the scaling-law tau_e model, a prefactor and the exponent of each factor.
# ITER89-P L-mode [s], its a^0.3 folded into R^1.5 epsilon^0.3,
# density in 1e19 m^-3 (the published 0.048 is for 1e20 m^-3, times 10^-0.1)
ITER89P_LMODE = {
    "coeff": 0.038,
    "alpha_I": 0.85,
    "alpha_B": 0.2,
    "alpha_N": 0.1,
    "alpha_P": -0.5,
    "alpha_R": 1.5,
    "alpha_kappa": 0.5,
    "alpha_epsilon": 0.3,
    "alpha_mass": 0.5,
}
# IPB98(y,2) H-mode [s], density in 1e19 m^-3
IPB98Y2_HMODE = {
    "coeff": 0.0562,
    "alpha_I": 0.93,
    "alpha_B": 0.15,
    "alpha_N": 0.41,
    "alpha_P": -0.69,
    "alpha_R": 1.97,
    "alpha_kappa": 0.78,
    "alpha_epsilon": 0.58,
    "alpha_mass": 0.19,
}
# ITER 1996 L-H power threshold [MW], density in 1e20 m^-3
# (https://ukaea.github.io/PROCESS/physics-models/plasma_h_mode/)
ITER1996_LH_THRESHOLD = {
    "coeff": 0.45,
    "alpha_N": 0.75,
    "alpha_B": 1.0,
    "alpha_R": 2.0,
}
# The ScalingLawPredictor fields holding those coefficients.
# They train but take no weight decay, a pull toward zero is no prior for a physics exponent
SCALING_LAW_FIELDS = ("scaling_lmode", "scaling_hmode", "scaling_lh_transition")


def trainable_scaling(scaling: dict[str, float]) -> dict[str, Array]:
    """The trainable leaves of a scaling, its prefactor stored as log_coeff.

    A log prefactor stays positive and trains in relative steps like the exponents.
    A linear one sits at ~0.05, where an Adam step of ~lr could flip its sign.
    """
    leaves = {name: jnp.asarray(value, dtype=float) for name, value in scaling.items() if name != "coeff"}
    coeff = jnp.asarray(scaling["coeff"], dtype=float)
    leaves["log_coeff"] = jnp.log(coeff)
    return leaves


def power_law(scaling: Mapping[str, ArrayLike], factors: Mapping[str, ArrayLike]) -> Array:
    """exp(log_coeff) * prod(factor^alpha) over the exponents of a scaling, factors keyed by exponent name."""
    log_terms = [alpha * jnp.log(factors[name]) for name, alpha in scaling.items() if name != "log_coeff"]
    return jnp.exp(scaling["log_coeff"] + sum(log_terms))


@chex.dataclass
class TauePredictorOutputs:
    taue_pred: float  # [s]
    debug_info: dict | None = None


class BoundedNNPredictor(eqx.Module):
    """tau_e from an NN over the normalized input features, smooth-clamped to [min_val, max_val]."""

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    def __call__(self, feature_vec: Array) -> TauePredictorOutputs:
        nn_out = self.nn(feature_vec)
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

    isotope_mass: float = eqx.field(static=True, default=2)  # Assume DD operation, the source devices carry no per-shot isotope signal

    @chex.dataclass
    class Inputs:
        """Inputs for scaling law predictor, MUST BE IN REAL UNITS"""

        ip_MA: float  # [MA]
        b_geo: float  # Vacuum toroidal field at the geometric axis [T]
        geometric_axis_r: float  # Major radius [m]
        minor_radius: float  # Minor radius [m]
        elongation: float  # Elongation [-]
        n_e_line_average_1e20: float  # Electron density [10^20 m^-3]
        power_additional_MW: float

        # Needed for the scaling law to calculate P_abs
        # This MUST come from the P_oh submodule and NOT from the training dataset!
        # otherwise that's giving this model more information and that's cheating!
        power_ohm_MW: float

        @property
        def epsilon(self):
            return self.minor_radius / self.geometric_axis_r

        @property
        def P_abs_MW(self):
            return self.power_additional_MW + self.power_ohm_MW

    def __init__(self):
        """ITER89-P L-mode, IPB98(y,2) H-mode and the 1996 L-H threshold as the starting coefficients."""
        self.scaling_lmode = trainable_scaling(ITER89P_LMODE)
        self.scaling_hmode = trainable_scaling(IPB98Y2_HMODE)
        self.scaling_lh_transition = trainable_scaling(ITER1996_LH_THRESHOLD)

    def __call__(self, inp: Inputs) -> TauePredictorOutputs:
        # Every factor floored at 0.001 so its log stays finite
        ip_MA = jnp.clip(inp.ip_MA, 0.001, None)
        b_geo = jnp.clip(inp.b_geo, 0.001, None)
        ne19 = jnp.clip(inp.n_e_line_average_1e20 * 10, 0.001, None)
        ne20 = jnp.clip(inp.n_e_line_average_1e20, 0.001, None)
        P_abs_MW = jnp.clip(inp.P_abs_MW, 0.001, None)
        elongation = jnp.clip(inp.elongation, 0.001, None)
        epsilon = jnp.clip(inp.epsilon, 0.001, None)
        geometric_axis_r = jnp.clip(inp.geometric_axis_r, 0.001, None)

        confinement_factors: dict[str, ArrayLike] = {
            "alpha_I": ip_MA,
            "alpha_B": b_geo,
            "alpha_N": ne19,
            "alpha_P": P_abs_MW,
            "alpha_R": geometric_axis_r,
            "alpha_kappa": elongation,
            "alpha_epsilon": epsilon,
            "alpha_mass": self.isotope_mass,
        }
        taue_lmode = power_law(self.scaling_lmode, confinement_factors)
        taue_hmode = power_law(self.scaling_hmode, confinement_factors)
        threshold_factors: dict[str, ArrayLike] = {"alpha_N": ne20, "alpha_B": b_geo, "alpha_R": geometric_axis_r}
        p_thresh = power_law(self.scaling_lh_transition, threshold_factors)

        # Smooth blend so gradient reaches the scaling_lh_transition coefficients
        lh_weight = jax.nn.sigmoid(LH_BLEND_SHARPNESS * (P_abs_MW / p_thresh - 1.0))
        taue = (1.0 - lh_weight) * taue_lmode + lh_weight * taue_hmode

        # The clamp input here is the PHYSICAL scaling-law tau_e, so the floor
        # smoothing must stay narrow: device tau_e commonly sits at 20-50 ms,
        # which a range-fraction min width would distort by tens of percent.
        # (BoundedNNPredictor keeps symmetric widths, its clamp input is a raw NN output with no physical meaning)
        width_max = BOUND_CLAMP_WIDTH_FRAC * (MAX_TAUE - MIN_TAUE)
        bounded = smooth_clamp(taue, MIN_TAUE, MAX_TAUE, TAUE_CLAMP_MIN_WIDTH, width_max)
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


class PowerBalance(TimeDepModule):
    """Base for the time-dependent stored-energy models.

    Inputs are always in PHYSICAL units plus the device index
    Each concrete model normalizes for its neural networks internally
    (see transport_study.modules.normalization)
    physics pieces like the scaling laws and the Wtot/tau_e power balance consume the physical values.
    """

    @chex.dataclass
    class State:
        energy_mhd_MJ: float

    # The 7 physical inputs plus the device index selecting per-device normalization statistics
    Inputs = InputNormalizer.Inputs

    @chex.dataclass
    class Output:
        energy_mhd_MJ_pred: float
        P_cond_MW: float
        taue_predictor_output: TauePredictorOutputs
        # Submodule predictions, exposed so the training loss can anchor them
        # to the measured signals. NaN for model types without submodules.
        power_ohm_MW_pred: float = float("nan")
        power_radiated_MW_pred: float = float("nan")

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


class PowerBalanceTaue(PowerBalance):
    """Base of the tau_e models, dW/dt = P_aux + P_oh - P_rad - Wtot / tau_e with predicted P_oh and P_rad.

    Subclasses declare the submodule fields and implement predict_taue.
    Note that while we are including most of the same terms,
    this tau_E is not quite the same as in H89 and H98 scaling laws.
    Here we say the tau_E is based on total energy for only the conducted power, while
    ITER89-P and IPB98(y,2) define tau_E with the loss power P_heat - dW/dt,
    radiation not subtracted, and for only the thermal energy.
    """

    p_oh_predictor: eqx.AbstractVar[OhmicPower]
    p_rad_predictor: eqx.AbstractVar[RadiatedPower]

    @abc.abstractmethod
    def predict_taue(self, inputs: InputNormalizer.Inputs, power_ohm_MW: ArrayLike) -> TauePredictorOutputs:
        """tau_e from the physical inputs and the predicted ohmic power."""

    def __call__(self, state: PowerBalance.State, inputs: InputNormalizer.Inputs) -> tuple[PowerBalance.State, PowerBalance.Output]:
        # Submodules take physical inputs and normalize internally with their own stats
        p_oh_predictor_output = self.p_oh_predictor(inputs)
        p_rad_predictor_output = self.p_rad_predictor(inputs)
        power_ohm_MW = p_oh_predictor_output.power_ohm_MW_pred
        power_radiated_MW = p_rad_predictor_output.power_radiated_MW_pred

        taue_predictor_output = self.predict_taue(inputs, power_ohm_MW)
        energy_mhd_MJ = self.positive_wtot(state.energy_mhd_MJ)
        P_cond_MW = energy_mhd_MJ / taue_predictor_output.taue_pred
        P_abs_MW = inputs.power_additional_MW + power_ohm_MW
        energy_mhd_MJ_dot = self.bound_wtot_dot(state.energy_mhd_MJ, P_abs_MW - P_cond_MW - power_radiated_MW)

        state_dot = PowerBalance.State(energy_mhd_MJ=energy_mhd_MJ_dot)
        output = PowerBalance.Output(
            energy_mhd_MJ_pred=energy_mhd_MJ,
            P_cond_MW=P_cond_MW,
            taue_predictor_output=taue_predictor_output,
            power_ohm_MW_pred=power_ohm_MW,
            power_radiated_MW_pred=power_radiated_MW,
        )
        return state_dot, output


class PowerBalanceScalingLaw(PowerBalanceTaue):
    taue_predictor: ScalingLawPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower

    @classmethod
    def init(cls, p_oh_predictor: OhmicPower, p_rad_predictor: RadiatedPower) -> "PowerBalanceScalingLaw":
        return cls(
            taue_predictor=ScalingLawPredictor(),
            p_oh_predictor=p_oh_predictor,
            p_rad_predictor=p_rad_predictor,
        )

    def predict_taue(self, inputs: InputNormalizer.Inputs, power_ohm_MW: ArrayLike) -> TauePredictorOutputs:
        # The scaling law is dimensional physics, it MUST see physical units
        taue_predictor_inputs = ScalingLawPredictor.Inputs(
            ip_MA=inputs.ip_MA,
            b_geo=inputs.b_geo,
            geometric_axis_r=inputs.geometric_axis_r,
            minor_radius=inputs.minor_radius,
            elongation=inputs.elongation,
            n_e_line_average_1e20=inputs.n_e_line_average_1e20,
            power_additional_MW=inputs.power_additional_MW,
            power_ohm_MW=power_ohm_MW,
        )
        return self.taue_predictor(taue_predictor_inputs)


class PowerBalanceSciML(PowerBalanceTaue):
    taue_predictor: BoundedNNPredictor
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower
    normalizer: InputNormalizer

    @classmethod
    def init(
        cls,
        nn_width: int,
        nn_depth: int,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        normalizer: InputNormalizer,
        prng_seed: int,
    ) -> "PowerBalanceSciML":
        taue_predictor = BoundedNNPredictor(
            nn=eqx.nn.MLP(
                in_size=N_FEATURES,
                out_size=1,
                width_size=nn_width,
                depth=nn_depth,
                key=jax.random.PRNGKey(prng_seed),
            ),
            min_val=MIN_TAUE,
            max_val=MAX_TAUE,
        )
        return cls(
            taue_predictor=taue_predictor,
            p_oh_predictor=p_oh_predictor,
            p_rad_predictor=p_rad_predictor,
            normalizer=normalizer,
        )

    def predict_taue(self, inputs: InputNormalizer.Inputs, power_ohm_MW: ArrayLike) -> TauePredictorOutputs:
        # The tau_e NN sees this model's own normalized features
        features = self.normalizer(inputs)
        feature_vec = features.to_vec()
        return self.taue_predictor(feature_vec)


class PowerBalanceMLP(PowerBalance):
    """dW/dt from an MLP over the normalized inputs and the current predicted Wtot.

    The predicted Wtot is the rollout's only feedback, the model's own state as in the tau_e models' Wtot / tau_e.
    Without it dW/dt is a function of the inputs alone,
    an open-loop integrator that accumulates every error over the shot.
    The normalizer (built with_energy) scales Wtot into the NN input
    and maps the NN output back to dW/dt (normalize_with_energy, energy_rate_scale),
    so both sides sit near 1 whatever the device's energy scale.
    """

    nn: eqx.Module
    normalizer: InputNormalizer

    def __call__(self, state: PowerBalance.State, inputs: InputNormalizer.Inputs) -> tuple[PowerBalance.State, PowerBalance.Output]:
        energy_mhd_MJ = self.positive_wtot(state.energy_mhd_MJ)
        nn_input = self.normalizer.normalize_with_energy(inputs, energy_mhd_MJ)
        nn_out = self.nn(nn_input)
        energy_rate_scale = self.normalizer.energy_rate_scale(inputs)
        energy_mhd_MJ_dot_unbounded = energy_rate_scale * nn_out.squeeze()
        energy_mhd_MJ_dot = self.bound_wtot_dot(state.energy_mhd_MJ, energy_mhd_MJ_dot_unbounded)

        state_dot = PowerBalance.State(energy_mhd_MJ=energy_mhd_MJ_dot)
        output = PowerBalance.Output(
            energy_mhd_MJ_pred=energy_mhd_MJ,
            P_cond_MW=jnp.nan,  # Not predicted in this model
            taue_predictor_output=TauePredictorOutputs(taue_pred=jnp.nan, debug_info={"nn_out": nn_out.squeeze()}),
        )

        return state_dot, output

    @classmethod
    def init(
        cls,
        nn_width: int,
        nn_depth: int,
        normalizer: InputNormalizer,
        prng_seed: int = 42,
    ) -> "PowerBalanceMLP":
        nn = eqx.nn.MLP(
            in_size=N_FEATURES + 1,
            out_size=1,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, normalizer=normalizer)


class PowerBalanceTransformer(PowerBalance):
    """Purely data-driven dW/dt predictor with recurrent causal attention.

    A rolling buffer of the last history_len predicted Wtot values
    is carried in the module State as a DISCRETE field
    The simple-Euler stepper integrates only continuous state (energy_mhd_MJ)
    and passes discrete fields through as the next state directly
    (see popsim.simulate._single_step and popsim.modules.delay.DelayBuffer for the pattern),
    so the buffer update is an exact discrete shift
    no_save keeps the (history_len,) buffer out of the recorded simulation output.

    Each step: shift the current (floored) Wtot state into the buffer,
    embed the normalized 7 inputs to a query token,
    embed each buffered Wtot value plus a learned per-slot position embedding to key/value tokens,
    attend (causal by construction, the buffer only ever contains current and past predictions),
    then a residual connection and an MLP head produce a bounded energy_mhd_MJ_dot.
    Scalar Wtot tokens are indistinguishable beyond their value,
    so without the position embedding attention would be
    permutation-invariant over the history and unable to read trends.
    As in the mlp, the normalizer (built with_energy) scales every buffered Wtot at the current inputs
    and maps the head output back to dW/dt (normalize_with_energy, energy_rate_scale).

    With predicts_powers (a multiobjective case) the head has two more outputs,
    P_oh and P_rad on the same normalized power scale as dW/dt.
    They do not enter the Wtot update, they only give the training loss extra targets to anchor.
    """

    normalizer: InputNormalizer
    feature_embed: eqx.nn.Linear
    wtot_embed: eqx.nn.Linear
    pos_embed: Array
    attention: eqx.nn.MultiheadAttention
    head: eqx.nn.MLP
    history_len: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    predicts_powers: bool = eqx.field(static=True)

    @chex.dataclass
    class State:
        energy_mhd_MJ: float
        # (history_len,) past predicted energy_mhd_MJ values, most recent last
        history: Array = discrete_no_save_field(default=None)

    def __call__(self, state: "PowerBalanceTransformer.State", inputs: InputNormalizer.Inputs) -> tuple:
        # Shift the buffer by one and insert the current predicted Wtot at the end
        wtot_now = self.positive_wtot(state.energy_mhd_MJ)
        new_history = jnp.concatenate([state.history[1:], wtot_now[None]])

        # Every buffered Wtot normalized at the current inputs, the last row holds the current state
        normalized_vecs = jax.vmap(lambda energy_mhd_MJ: self.normalizer.normalize_with_energy(inputs, energy_mhd_MJ))(new_history)
        query = self.feature_embed(normalized_vecs[-1, :N_FEATURES])
        normalized_history = normalized_vecs[:, N_FEATURES]

        tokens = jax.vmap(self.wtot_embed)(normalized_history[:, None]) + self.pos_embed
        attn_out = self.attention(query[None, :], tokens, tokens)[0]
        latent = query + attn_out
        # [dW/dt] or, with predicts_powers, [dW/dt, P_oh, P_rad], all on the normalized power scale
        nn_out = self.head(latent)
        energy_rate_scale = self.normalizer.energy_rate_scale(inputs)
        energy_mhd_MJ_dot_unbounded = energy_rate_scale * nn_out[0]
        energy_mhd_MJ_dot = self.bound_wtot_dot(state.energy_mhd_MJ, energy_mhd_MJ_dot_unbounded)
        state_out = PowerBalanceTransformer.State(energy_mhd_MJ=energy_mhd_MJ_dot, history=new_history)
        power_ohm_MW_pred = energy_rate_scale * nn_out[1] if self.predicts_powers else jnp.nan
        power_radiated_MW_pred = energy_rate_scale * nn_out[2] if self.predicts_powers else jnp.nan
        output = PowerBalance.Output(
            energy_mhd_MJ_pred=wtot_now,
            P_cond_MW=jnp.nan,  # Not predicted in this model
            taue_predictor_output=TauePredictorOutputs(taue_pred=jnp.nan, debug_info={"nn_out": nn_out[0]}),
            power_ohm_MW_pred=power_ohm_MW_pred,
            power_radiated_MW_pred=power_radiated_MW_pred,
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
        predicts_powers: bool,
        prng_seed: int = 42,
    ) -> "PowerBalanceTransformer":
        key_embed, key_wtot, key_pos, key_attn, key_head = jax.random.split(jax.random.PRNGKey(prng_seed), 5)
        feature_embed = eqx.nn.Linear(N_FEATURES, d_model, key=key_embed)
        wtot_embed = eqx.nn.Linear(1, d_model, key=key_wtot)
        # Small random init breaks slot symmetry when the buffer holds a constant history
        # (the seeded state at t0)
        pos_embed = 0.02 * jax.random.normal(key_pos, (history_len, d_model))
        attention = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=d_model, key=key_attn)
        head = eqx.nn.MLP(
            in_size=d_model,
            out_size=3 if predicts_powers else 1,
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
            predicts_powers=predicts_powers,
        )


class PowerBalanceEnv(ModuleTrainingEnv):
    module: PowerBalance
    domain_adaptation: str = eqx.field(static=True, default="unset")
    freeze_submodules: list[str] = eqx.field(static=True, default_factory=list)
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    def create_state(self, observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        energy_mhd_MJ = observations["energy_mhd_MJ"].data
        if isinstance(self.module, PowerBalanceTransformer):
            # Seed the Wtot history with the measured t0 value tiled,
            # a constant history rather than a fake all-zero one
            energy_mhd_MJ = jnp.asarray(energy_mhd_MJ)
            history = jnp.broadcast_to(energy_mhd_MJ[..., None], (*energy_mhd_MJ.shape, self.module.history_len))
            return PowerBalanceTransformer.State(energy_mhd_MJ=energy_mhd_MJ, history=history)
        return PowerBalance.State(energy_mhd_MJ=energy_mhd_MJ)

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        # Top-level module inputs are always in physical units,
        # normalization happens inside the modules
        return InputNormalizer.inputs_from_dict(inputs)

    def get_trainable(self):
        """Trainable leaves for the optimizer partition.

        Selects NN leaves explicitly, never whole modules
        normalizer statistics are ordinary array leaves on every module and must stay frozen
        (a broad eqx.filter over a module would silently train them)
        """
        if self.domain_adaptation == "transfer":
            last_layer_leaves = []
            if isinstance(self.module, PowerBalanceMLP):
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
                for name in SCALING_LAW_FIELDS:
                    scaling = getattr(self.module.taue_predictor, name)
                    last_layer_leaves += list(scaling.values())
            if isinstance(self.module, PowerBalanceTaue):
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
        if isinstance(self.module, PowerBalanceTaue):
            if "p_oh_predictor" not in self.freeze_submodules:
                trainable_leaves["p_oh_predictor"] = eqx.filter(self.module.p_oh_predictor.nn, eqx.is_inexact_array)
            if "p_rad_predictor" not in self.freeze_submodules:
                trainable_leaves["p_rad_predictor"] = eqx.filter(self.module.p_rad_predictor.nn, eqx.is_inexact_array)

        if isinstance(self.module, PowerBalanceSciML):
            trainable_leaves["taue_predictor"] = eqx.filter(self.module.taue_predictor.nn, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceScalingLaw):
            trainable_leaves["taue_predictor"] = eqx.filter(self.module.taue_predictor, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceMLP):
            trainable_leaves["nn"] = eqx.filter(self.module.nn, eqx.is_inexact_array)
        elif isinstance(self.module, PowerBalanceTransformer):
            trainable_leaves["feature_embed"] = eqx.filter(self.module.feature_embed, eqx.is_inexact_array)
            trainable_leaves["wtot_embed"] = eqx.filter(self.module.wtot_embed, eqx.is_inexact_array)
            trainable_leaves["pos_embed"] = self.module.pos_embed
            trainable_leaves["attention"] = eqx.filter(self.module.attention, eqx.is_inexact_array)
            trainable_leaves["head"] = eqx.filter(self.module.head, eqx.is_inexact_array)

        return trainable_leaves
