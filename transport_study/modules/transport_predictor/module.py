import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, ArrayLike, PyTree
from popsim import TimeDepModule, discrete_no_save_field
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_volume
from popsim.math_utils import safe_log
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.rtd_mlp import Activation, RtdMLP
from popsim.simulate import StepperType
from scipy.constants import epsilon_0, eV, mu_0
from torax import ToraxConfig
from torax import experimental as torax_experimental
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic import torax_pydantic

from transport_study.modules.normalization import (
    FeatureNormalizer,
    feature_fit_arrays,
    make_feature_normalizer,
)
from transport_study.modules.power_balance.module import PowerBalance, PowerBalanceEnv

# Namespace import: the profile predictor also names its input dataclass
# Inputs, and a from-import alias makes ruff and isort fight over sorting
from transport_study.modules.profile_predictor import module as profile_predictor_module
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    NN_INPUT_NAMES,
    ProfilePredictor,
)
from transport_study.modules.profile_predictor.torax_module import (
    TAU_REF_S,
    TORAX_TRANSPORT_MODEL_NAMES,
    TRANSPORT_COEFFICIENT_NAMES,
    build_circular_geometry_jax,
    build_miller_geometry_jax,
    clamp_core_profiles,
)

# Positivity floor for the profile outputs, ne [1e20 m^-3] and te [keV]
MIN_PROFILE = 1e-3

# Floors applied to profiles used to seed a TORAX state. The GP-fit profiles
# clamp to exactly 0 over several edge points on rampdown timeslices, and
# zero-temperature cells NaN the TORAX solve regardless of the predicted
# coefficients (2026-07-24 attribution probe). 0.02 keV still left one NaN
# sample, these values gave 0/256
TE_SEED_FLOOR_KEV = 0.05
NE_SEED_FLOOR_20 = 0.02

# Floor on the stored energy when used as a denominator or feature scale
MIN_W_MJ = 1e-3

# The 10 profile predictor feature slots plus one aux power feature
N_TRANSPORT_NN_INPUTS = N_NN_INPUTS + 1

# Names of the transport_nn_inputs slots, in order. The beta-derived slots keep
# the profile predictor's names even though they come from Wtot here
TRANSPORT_NN_INPUT_NAMES = (*NN_INPUT_NAMES, "paux_norm")

# Dataset variables transport_nn_inputs is derived from (no betan - the
# beta-derived slots come from the stored energy)
TRANSPORT_NN_INPUT_SOURCE_VARS = (
    "Ip_MA",
    "B0",
    "ne20_line_avg",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
    "P_aux_MW",
    "Wtot_MJ",
)

# Coefficients predicted by the transport predictor sources network, in the
# order of the network outputs. Unlike ProfilePredictorTorax there is no
# P_aux_total entry: the auxiliary heating magnitude is a measured input here,
# the NN only predicts the deposition shape and the particle fueling.
SOURCE_SHAPE_COEFFICIENT_NAMES = (
    "S_total",
    "gaussian_location",
    "gaussian_width",
    "electron_heat_fraction",
)


@chex.dataclass
class Inputs:
    """Inputs to the transport predictor module.

    Union of the profile predictor and power balance input sets, minus betan:
    the stored energy is part of the predicted state, so every beta-derived
    quantity is computed from the state Wtot via the *_from_wtot methods
    instead of a measurement.
    """

    Ip_MA: float  # Plasma current [MA]
    B0: float  # On-axis toroidal field [T]
    ne20_line_avg: float  # Line-averaged electron density [10^20 m^-3]
    R0: float  # Geometric major radius [m]
    a_minor: float  # Minor radius [m]
    kappa: float  # Elongation
    delta_top: float  # Upper triangularity
    delta_bot: float  # Bottom triangularity
    P_aux_MW: float  # Auxiliary heating power [MW]
    ds_source_idx: float  # Device index selecting per-device normalization statistics

    @property
    def epsilon(self):
        return self.a_minor / self.R0

    @property
    def q_star(self):
        delta = (self.delta_top + self.delta_bot) / 2
        f_shaping = calc_f_shaping(
            self.epsilon,
            self.kappa,
            delta,
        )
        q_star = calc_q_star(self.B0, self.R0, self.epsilon, self.Ip_MA, f_shaping)
        return q_star

    @property
    def fGW(self):
        greenwald_limit = self.Ip_MA / (jnp.pi * self.a_minor**2)
        return self.ne20_line_avg / greenwald_limit

    @property
    def aB0(self):
        return self.a_minor * self.B0

    @property
    def volume_approx(self):
        return calc_plasma_volume(
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            areal_elongation=self.kappa,
        )

    def beta_from_wtot(self, Wtot_MJ: ArrayLike) -> ArrayLike:
        # Inverse of W = (3/2) * beta * B0^2 / (2 mu_0) * V / 1e6
        pressure_Pa = Wtot_MJ * 1e6 / (1.5 * self.volume_approx)
        return 2.0 * mu_0 * pressure_Pa / self.B0**2

    def betan_from_wtot(self, Wtot_MJ: ArrayLike) -> ArrayLike:
        # betan follows the percent Troyon convention (beta[%] * a*B0/Ip)
        return self.beta_from_wtot(Wtot_MJ) * 100.0 * self.a_minor * self.B0 / self.Ip_MA

    def te_approx_from_wtot(self, Wtot_MJ: ArrayLike) -> ArrayLike:
        # Single-fluid pressure p = ne * Te, so Te = p / ne
        pressure_Pa = Wtot_MJ * 1e6 / (1.5 * self.volume_approx)
        pressure_keV20 = pressure_Pa / eV / 1e3 / 1e20
        return pressure_keV20 / self.ne20_line_avg

    def nu_star_from_wtot(self, Wtot_MJ: ArrayLike) -> ArrayLike:
        # characteristic collisionality, from https://arxiv.org/pdf/2406.18442 eqn 2
        # SI formula with temperature in joules, rearranged so the physical
        # constants and unit conversions fold into python-float coefficients
        # before touching the arrays: float32 array intermediates would
        # otherwise overflow (ne_m3 / te_J^2 ~ 1e49) or underflow (eV^4 ~ 6.6e-76)
        # and produce inf * 0 = nan
        te_eV = self.te_approx_from_wtot(Wtot_MJ) * 1e3
        # coulomb logarithm of debye_length over b90, which expands to
        # log of 4 pi eps0^1.5 te_J^1.5 / (e^3 ne_m3^0.5) with te_J = te_eV * e
        lambda_coeff = 4 * jnp.pi * epsilon_0**1.5 / (eV**1.5 * 1e10)
        ln_lambda = safe_log(lambda_coeff * te_eV**1.5 / jnp.sqrt(self.ne20_line_avg))
        # e^4 / (2 pi eps0^2) * ne_m3 / te_J^2
        collision_coeff = eV**2 / (2 * jnp.pi * epsilon_0**2) * 1e20
        collision_term = collision_coeff * self.ne20_line_avg / te_eV**2
        geometry_term = self.q_star * self.R0 / (self.epsilon**1.5)
        return collision_term * geometry_term * ln_lambda

    def transport_nn_inputs(self, Wtot_MJ: ArrayLike) -> Array:
        # The 10 profile predictor feature slots in the same order, with every
        # beta-derived entry computed from the state Wtot instead of a measured
        # betan, plus a dimensionless aux power feature: P_aux over the
        # Wtot / TAU_REF_S power scale. No log on the power feature so
        # P_aux = 0 ohmic phases stay finite.
        W_safe = jnp.maximum(Wtot_MJ, MIN_W_MJ)
        paux_norm = TAU_REF_S * self.P_aux_MW / W_safe
        inp_array = jnp.array(
            [
                self.beta_from_wtot(W_safe),
                self.q_star,
                self.epsilon,
                self.fGW,
                self.aB0,
                self.betan_from_wtot(W_safe),
                self.kappa,
                self.delta_top,
                self.delta_bot,
                safe_log(self.nu_star_from_wtot(W_safe)),
                paux_norm,
            ]
        )
        return inp_array

    def to_power_balance_inputs(self) -> PowerBalance.Inputs:
        return PowerBalance.Inputs(
            Ip_MA=self.Ip_MA,
            B0=self.B0,
            R0=self.R0,
            a_minor=self.a_minor,
            kappa=self.kappa,
            ne20_line_avg=self.ne20_line_avg,
            P_aux_MW=self.P_aux_MW,
            ds_source_idx=self.ds_source_idx,
        )

    def to_profile_predictor_inputs(self, rho: Array, betan: ArrayLike) -> "profile_predictor_module.Inputs":
        """Profile predictor inputs. betan is required because it is not a
        measured input here, callers derive it from the state Wtot."""
        return profile_predictor_module.Inputs(
            Ip=self.Ip_MA,
            B0=self.B0,
            betan=betan,
            ne20_line_avg=self.ne20_line_avg,
            R0=self.R0,
            a_minor=self.a_minor,
            kappa=self.kappa,
            delta_top=self.delta_top,
            delta_bot=self.delta_bot,
            ds_source_idx=self.ds_source_idx,
            rho=rho,
        )


def transport_nn_input_matrix(ds: xr.Dataset) -> np.ndarray:
    """(N, N_TRANSPORT_NN_INPUTS) matrix of the transport features over a flattened dataset.

    Every column is broadcast against Ip_MA first, so per-shot variables line up
    with the per-timeslice ones. The beta-derived entries use the MEASURED
    stored energy (at runtime the modules use the state-implied Wtot instead).
    The profile predictor's counterpart is module.nn_input_matrix.
    """
    reference = ds["Ip_MA"]

    def col(var: str) -> np.ndarray:
        return np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel()

    inputs = Inputs(
        Ip_MA=col("Ip_MA"),
        B0=col("B0"),
        ne20_line_avg=col("ne20_line_avg"),
        R0=col("R0"),
        a_minor=col("a_minor"),
        kappa=col("kappa"),
        delta_top=col("delta_top"),
        delta_bot=col("delta_bot"),
        P_aux_MW=col("P_aux_MW"),
        ds_source_idx=np.zeros(reference.size),  # Unused by transport_nn_inputs
    )
    return np.asarray(inputs.transport_nn_inputs(col("Wtot_MJ"))).T


def make_transport_nn_input_normalizer(method: str, fit_ds: xr.Dataset | None, n_devices: int) -> FeatureNormalizer:
    """Build the per-device stat stage over the 11 transport features.

    Thin wrapper around normalization.make_feature_normalizer (which holds the
    method dispatch shared with the profile predictor), so fit_ds None yields
    identity statistics with the correct pytree structure for callers about to
    overwrite the buffers from a checkpoint.
    """
    fit_data = None if fit_ds is None else feature_fit_arrays(fit_ds, transport_nn_input_matrix(fit_ds))
    return make_feature_normalizer(method, fit_data, n_devices, N_TRANSPORT_NN_INPUTS)


def wtot_from_profiles(ne20: Array, te_keV: Array, rho: Array, volume_m3: ArrayLike) -> ArrayLike:
    """Stored energy [MJ] implied by ne/te profiles on rho.

    Same single-fluid p = ne * Te convention as the Inputs beta conversions,
    with dV = V d(rho^2) so a flat profile recovers W = (3/2) p V exactly.
    """
    pressure_Pa = ne20 * 1e20 * te_keV * 1e3 * eV
    return 1.5 * jnp.trapezoid(pressure_Pa * 2.0 * rho, rho) * volume_m3 / 1e6


@chex.dataclass
class Output:
    ne: Array  # Electron density profile on rho [1e20 m^-3]
    te: Array  # Electron temperature profile on rho [keV]
    rho: Array  # The rho grid the profiles are evaluated on
    debug_info: dict | None = None


class TransportPredictor(TimeDepModule):
    """Base for the time-dependent profile predictors.

    Inputs are always in physical units plus the device index. Subclasses
    define their own State: the three architectures carry genuinely different
    state (stored energy, profile history buffer, TORAX internals).
    """

    # Class attribute aliases satisfy the TimeDepModule required-inner-class
    # check while keeping the shared dataclasses at module level
    Inputs = Inputs
    Output = Output

    @staticmethod
    def positive_profiles(ne: Array, te: Array):
        """Enforce positivity of the profiles, and return a debug dict"""
        ne_pos = jnp.clip(ne, MIN_PROFILE)
        te_pos = jnp.clip(te, MIN_PROFILE)
        debug_info = {
            "ne_orig": ne,
            "te_orig": te,
        }
        return ne_pos, te_pos, debug_info


class TransportPredictorTransformer(TransportPredictor):
    """Transport predictor using a fully data-driven transformer architecture.

    A rolling buffer of the last history_len profiles is carried in the module
    State as a DISCRETE field: the simple-Euler stepper integrates only
    continuous state and passes discrete fields through as the next state
    directly (see popsim.simulate._single_step and popsim.modules.delay.DelayBuffer),
    so the buffer update is an exact discrete shift and the model MUST run
    under a SIMPLE_EULER stepper. no_save keeps the buffer out of the recorded
    simulation output.

    Each step
    1: derive the stored energy implied by the current buffered profile (there is no measured betan)
    2: embed the normalized transport features of the CURRENT timestep to a query token
    (only predicted profiles are kept as history, never past input features)
    3: embed each buffered profile plus a learned per-slot position embedding to key/value tokens
    4: attend (causal by construction, the buffer only ever contains current and past profiles)
    5: then a residual connection and an MLP head produce the next profile
    6: which is rolled into the buffer
    Without the position embedding attention is permutation-invariant over the history,
    so the model could not tell the most recent profile from the least recent
    (and the t0-seeded buffer holds identical rows, where values alone carry no ordering at all)
    The reported Output is the profile currently stored for time t
    """

    # Per-device stat stage (CORAL or z-score) over the 11 transport_nn_inputs
    # Frozen like every normalizer, the trainable getters never include it
    normalizer: FeatureNormalizer
    feature_embed: eqx.nn.Linear
    profile_embed: eqx.nn.Linear
    pos_embed: Array
    attention: eqx.nn.MultiheadAttention
    head: eqx.nn.MLP
    rhogrid: tuple = eqx.field(static=True)
    history_len: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    @chex.dataclass
    class State:
        # (history_len, 2 * n_rho) raw physical profiles [ne | te], most recent last
        profiles: Array = discrete_no_save_field(default=None)

    def __call__(self, state: "TransportPredictorTransformer.State", inputs: Inputs) -> tuple:
        n_rho = len(self.rhogrid)
        rho = jnp.array(self.rhogrid)
        ne_now = state.profiles[-1, :n_rho]
        te_now = state.profiles[-1, n_rho:]

        # Stored energy implied by the current profile state drives every
        # beta-derived feature and scale
        Wtot_MJ = wtot_from_profiles(ne_now, te_now, rho, inputs.volume_approx)
        features = self.normalizer(inputs.transport_nn_inputs(Wtot_MJ), inputs.ds_source_idx)
        query = self.feature_embed(features)

        # Buffer rows are raw physical profiles: scale to order one with the
        # same data-derived scales the profile predictors use, floored where
        # the state or inputs are unreliable
        ne_scale = jnp.maximum(inputs.ne20_line_avg, 1e-2)
        te_scale = jnp.clip(inputs.te_approx_from_wtot(Wtot_MJ), 0.05, 5.0)
        row_scale = jnp.concatenate(
            [
                jnp.broadcast_to(ne_scale, (n_rho,)),
                jnp.broadcast_to(te_scale, (n_rho,)),
            ]
        )
        tokens = jax.vmap(self.profile_embed)(state.profiles / row_scale) + self.pos_embed

        attn_out = self.attention(query[jnp.newaxis, :], tokens, tokens)[0]
        latent = query + attn_out
        nn_out = self.head(latent)

        ne_next = nn_out[:n_rho] * ne_scale
        te_next = nn_out[n_rho:] * te_scale
        # Clipped values enter the buffer so the carried state stays physical
        ne_next, te_next, debug_info = self.positive_profiles(ne_next, te_next)
        debug_info.update(Wtot_MJ_state=Wtot_MJ)

        # Shift the buffer by one and insert the newest profile at the end
        new_row = jnp.concatenate([ne_next, te_next])
        new_profiles = jnp.concatenate([state.profiles[1:], new_row[jnp.newaxis, :]], axis=0)

        state_out = TransportPredictorTransformer.State(profiles=new_profiles)
        output = Output(
            ne=ne_now,
            te=te_now,
            rho=rho,
            debug_info=debug_info,
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
        rhogrid: Array,
        normalizer: FeatureNormalizer,
        prng_seed: int = 42,
    ) -> "TransportPredictorTransformer":
        key_feat, key_prof, key_pos, key_attn, key_head = jax.random.split(jax.random.PRNGKey(prng_seed), 5)
        # Coerce to tuple: arrays in static fields break pytree metadata equality
        rhogrid_tuple = tuple(np.asarray(rhogrid).tolist())
        n_rho = len(rhogrid_tuple)
        feature_embed = eqx.nn.Linear(N_TRANSPORT_NN_INPUTS, d_model, key=key_feat)
        profile_embed = eqx.nn.Linear(2 * n_rho, d_model, key=key_prof)
        # Small random init breaks slot symmetry when the buffer holds a
        # constant history (the seeded state at t0)
        pos_embed = 0.02 * jax.random.normal(key_pos, (history_len, d_model))
        attention = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=d_model, key=key_attn)
        head = eqx.nn.MLP(
            in_size=d_model,
            out_size=2 * n_rho,
            width_size=nn_width,
            depth=nn_depth,
            key=key_head,
        )
        return cls(
            normalizer=normalizer,
            feature_embed=feature_embed,
            profile_embed=profile_embed,
            pos_embed=pos_embed,
            attention=attention,
            head=head,
            rhogrid=rhogrid_tuple,
            history_len=history_len,
            d_model=d_model,
        )


class TransportPredictorSciML(TransportPredictor):
    """Transport predictor using a time-dependent power balance and a
    time-independent profile predictor.

    The power balance evolves the stored energy state. Each step the evolving
    Wtot is converted to a normalized beta (the exact inverse of the beta to
    stored-energy estimate) and fed to the profile predictor, so profile-loss
    gradients flow into the power balance networks and the profiles follow
    the predicted dynamics.
    """

    power_balance: PowerBalance
    profile_predictor: ProfilePredictor

    State = PowerBalance.State

    @classmethod
    def init(
        cls,
        power_balance: PowerBalance,
        profile_predictor: ProfilePredictor,
    ) -> "TransportPredictorSciML":
        return cls(power_balance=power_balance, profile_predictor=profile_predictor)

    def __call__(self, state: PowerBalance.State, inputs: Inputs) -> tuple[PowerBalance.State, Output]:
        # The power balance model normalizes internally with its own stats
        pb_state_dot, pb_output = self.power_balance(state=state, inputs=inputs.to_power_balance_inputs())

        # Wtot_MJ_pred is the floored current-state estimate (positive_wtot)
        Wtot_MJ = pb_output.Wtot_MJ_pred
        betan_used = inputs.betan_from_wtot(Wtot_MJ)

        rho = jnp.array(self.profile_predictor.rhogrid)
        pp_output = self.profile_predictor(inputs.to_profile_predictor_inputs(rho=rho, betan=betan_used))

        # Unwrap the xr.DataArray profiles: xarray leaves do not survive the
        # stepper output stacking
        ne, te, debug_info = self.positive_profiles(pp_output.ne.data, pp_output.te.data)
        debug_info.update(
            Wtot_MJ_pred=Wtot_MJ,
            P_cond_MW=pb_output.P_cond_MW,
            taue_pred=pb_output.taue_predictor_output.taue_pred,
            betan_used=betan_used,
        )

        output = Output(ne=ne, te=te, rho=rho, debug_info=debug_info)
        return pb_state_dot, output


class TransportPredictorToraxBase(TransportPredictor):
    """Shared machinery for the TORAX-backed transport predictors.

    Standalone from ProfilePredictorTorax on purpose: that module is a
    steady-state relaxation estimator which has to infer the auxiliary
    heating magnitude with a NN, while here P_aux is an input fed
    straight to the generic_heat source, and the sources network only
    predicts the deposition shape and the particle fueling.
    Every beta-derived feature comes from the stored energy implied
    by the profile state, there is no input betan.

    One __call__ advances TORAX by exactly one solver step of sim_dt, so the
    torax config numerics must satisfy t_final - t_initial == fixed_dt ==
    sim_dt. rhogrid must span [0, 1] inclusive, it doubles as the
    initial-condition grid.

    Subclasses define State and __call__, both carry state as DISCRETE
    fields, so a SIMPLE_EULER stepper is required (see the
    TransportPredictorTransformer docstring).
    """

    rhogrid: tuple = eqx.field(static=True)
    # Which TORAX transport model the transport network parameterizes:
    # "constant", "cgm", "gyrobohm", or "qlknn"
    transport_model: str = eqx.field(static=True)
    # Which per-sample geometry builder to use: "circular" or "miller"
    geometry_builder: str = eqx.field(static=True)
    # Radial exponent p in delta(rho_norm) = delta_edge * rho_norm**p,
    # only used by the miller builder
    delta_exponent: float = eqx.field(static=True)

    nn_transport: RtdMLP
    nn_sources: RtdMLP
    nn_edge: RtdMLP
    # Per-device stat stage (CORAL or z-score) over the 11 transport_nn_inputs
    normalizer: FeatureNormalizer

    step_fn: SimulationStepFn = eqx.field(static=True)

    # Static mesh info for JAX-differentiable geometry construction
    _face_centers: tuple = eqx.field(static=True)
    _rho_hires_norm: tuple = eqx.field(static=True)
    # Outer simulation time step [s] the torax config window must match
    sim_dt: float = eqx.field(static=True)

    def __init__(
        self,
        nn_width: int,
        nn_depth: int,
        rhogrid: tuple,
        torax_config: ToraxConfig | dict,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
        sim_dt: float,
        transport_model: str = "cgm",
        geometry_builder: str = "circular",
        delta_exponent: float = 2.0,
    ):
        if transport_model not in TRANSPORT_COEFFICIENT_NAMES:
            raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TRANSPORT_COEFFICIENT_NAMES)}")
        self.transport_model = transport_model
        self.normalizer = normalizer
        if geometry_builder not in ("circular", "miller"):
            raise ValueError(f"Unknown geometry builder '{geometry_builder}', valid: ('circular', 'miller')")
        self.geometry_builder = geometry_builder
        self.delta_exponent = float(delta_exponent)

        key, subkey_transport, subkey_sources, subkey_edge = jax.random.split(key, 4)
        self.nn_transport = RtdMLP(
            in_size=N_TRANSPORT_NN_INPUTS,
            out_size=len(TRANSPORT_COEFFICIENT_NAMES[transport_model]),
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_transport,
        )
        self.nn_sources = RtdMLP(
            in_size=N_TRANSPORT_NN_INPUTS,
            # One output per SOURCE_SHAPE_COEFFICIENT_NAMES entry:
            # S_total, gaussian_location, gaussian_width, electron_heat_fraction
            out_size=len(SOURCE_SHAPE_COEFFICIENT_NAMES),
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_sources,
        )
        self.nn_edge = RtdMLP(
            in_size=N_TRANSPORT_NN_INPUTS,
            out_size=2,  # edge density fraction, edge temperature fraction
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_edge,
        )

        if isinstance(torax_config, dict):
            torax_config = ToraxConfig.from_dict(torax_config)

        expected_model_name = TORAX_TRANSPORT_MODEL_NAMES[transport_model]
        if torax_config.transport.model_name != expected_model_name:
            raise ValueError(
                f"transport_model '{transport_model}' requires torax_config transport.model_name "
                f"'{expected_model_name}', got '{torax_config.transport.model_name}'"
            )

        self.step_fn = torax_experimental.make_step_fn(torax_config)
        # Coerce to tuple: arrays in static fields break pytree metadata
        # equality (ambiguous truth value) when two module instances coexist
        self.rhogrid = tuple(np.asarray(rhogrid).tolist())

        static_geo = self.step_fn.geometry_provider(0.0)
        self._face_centers = tuple(static_geo.torax_mesh.face_centers.tolist())
        self._rho_hires_norm = tuple(np.array(static_geo.rho_hires_norm).tolist())

        # One solver step per __call__ by construction: the config window and
        # the fixed solver dt must both equal the outer simulation dt
        numerics = self.step_fn.runtime_params_provider.numerics
        fixed_dt = float(numerics.fixed_dt.get_value(0.0))
        t_span = float(numerics.t_final - numerics.t_initial)
        if abs(t_span - sim_dt) > 1e-9 or abs(fixed_dt - sim_dt) > 1e-9:
            raise ValueError(
                f"torax config numerics span t_final - t_initial = {t_span} s "
                f"with fixed_dt = {fixed_dt} s, but the outer simulation dt is "
                f"{sim_dt} s, one __call__ must advance exactly one solver step of sim_dt"
            )
        self.sim_dt = float(sim_dt)

    @classmethod
    def init(
        cls,
        rhogrid: Array,
        torax_config: ToraxConfig | dict,
        nn_width: int,
        nn_depth: int,
        prng_seed: int,
        normalizer: FeatureNormalizer,
        sim_dt: float,
        transport_model: str = "cgm",
        geometry_builder: str = "circular",
        delta_exponent: float = 2.0,
    ):
        return cls(
            nn_width=nn_width,
            nn_depth=nn_depth,
            rhogrid=rhogrid,
            torax_config=torax_config,
            key=jax.random.PRNGKey(prng_seed),
            normalizer=normalizer,
            sim_dt=sim_dt,
            transport_model=transport_model,
            geometry_builder=geometry_builder,
            delta_exponent=delta_exponent,
        )

    @property
    def rho_norm_grid(self) -> np.ndarray:
        """Cell-center grid the TORAX core profiles live on.

        This is rho_norm, the normalized toroidal flux radius. For the
        circular geometry used here, minor radius = a * rho_norm, so rho_norm
        is exactly the normalized minor radius rho the datasets use.
        """
        face_centers = np.array(self._face_centers)
        return (face_centers[:-1] + face_centers[1:]) / 2.0

    def _transport_coefficients(self, nn_transport_out: jax.Array) -> dict:
        """Bound the raw transport-network outputs to physical ranges for the configured model.

        The bounds keep the TORAX solver stable during training for any network
        output. Keys and order match TRANSPORT_COEFFICIENT_NAMES[self.transport_model].
        Same bounds as ProfilePredictorTorax, see the rationale comments there.
        """
        if self.transport_model == "constant":
            #   chi_i: 0.1 - 5 m^2/s
            #   chi_e: 0.1 - 10 m^2/s
            #   D_e:   0.1 - 2 m^2/s   (nonzero floor prevents advection-only blowup)
            #   V_e:   -5 - 5 m/s      (signed pinch)
            return {
                "chi_i": 0.1 + 4.9 * jax.nn.sigmoid(nn_transport_out[0:1]),
                "chi_e": 0.1 + 9.9 * jax.nn.sigmoid(nn_transport_out[1:2]),
                "D_e": 0.1 + 1.9 * jax.nn.sigmoid(nn_transport_out[2:3]),
                "V_e": 5.0 * jnp.tanh(nn_transport_out[3:4]),
            }
        elif self.transport_model == "cgm":
            # Bounds: chi_e_i_ratio 0.2 to 5, chi_D_ratio 1 to 20,
            # VR_D_ratio -5 to 5, alpha 1.8 to 2.2, chi_stiff 0.5 to 3
            return {
                "chi_e_i_ratio": 0.2 + 4.8 * jax.nn.sigmoid(nn_transport_out[0:1]),
                "chi_D_ratio": 1.0 + 19.0 * jax.nn.sigmoid(nn_transport_out[1:2]),
                "VR_D_ratio": 5.0 * jnp.tanh(nn_transport_out[2:3]),
                "alpha": 1.8 + 0.4 * jax.nn.sigmoid(nn_transport_out[3:4]),
                "chi_stiff": 0.5 + 2.5 * jax.nn.sigmoid(nn_transport_out[4:5]),
            }
        elif self.transport_model == "gyrobohm":
            # Bounds: bohm and gyrobohm multipliers exp(-3) to exp(3),
            # D_face_c1 and D_face_c2 0.01 to 5, V_face_coeff -4 to 2 with the
            # sigmoid bias putting a random init at the TORAX default -0.1
            return {
                "chi_bohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[0:1])),
                "chi_gyrobohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[1:2])),
                "D_face_c1": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[2:3]),
                "D_face_c2": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[3:4]),
                "V_face_coeff": 2.0 - 6.0 * jax.nn.sigmoid(nn_transport_out[4:5] - 0.62),
            }
        else:  # qlknn
            #   ITG_flux_ratio_correction: ~0.08 - 12 around 1
            #   ETG_correction_factor: ~0.03 - 4 around the default 1/3
            #   collisionality_multiplier: ~0.08 - 12 around 1
            return {
                "ITG_flux_ratio_correction": jnp.exp(2.5 * jnp.tanh(nn_transport_out[0:1])),
                "ETG_correction_factor": (1.0 / 3.0) * jnp.exp(2.5 * jnp.tanh(nn_transport_out[1:2])),
                "collisionality_multiplier": jnp.exp(2.5 * jnp.tanh(nn_transport_out[2:3])),
            }

    def _nn_coefficients(self, inputs: Inputs, Wtot_MJ: ArrayLike) -> dict:
        # Transport model free parameters, source shape and edge boundary
        # conditions from neural networks, bounded to physical ranges so the
        # TORAX solver stays stable during training. The heating MAGNITUDE is
        # NOT here: it is the measured P_aux_MW input, wired directly to
        # generic_heat.P_total in _build_provider_and_geo.
        # Source network outputs, per SOURCE_SHAPE_COEFFICIENT_NAMES:
        #   S_total: 0 - inf, softplus multiple of the device fueling scale
        #     particle_inventory / TAU_REF_S (x 1e21 in the provider), same
        #     device-transferable reparameterization as ProfilePredictorTorax
        #   gaussian_location: 0 - 0.8 (deposition center in rho_norm)
        #   gaussian_width: 0.02 - 0.4 (deposition width in rho_norm)
        #   electron_heat_fraction: 0.2 - 0.95 (ceiling raised with the
        #     profile predictor, ST NBI heating is electron-dominated. The
        #     sigmoid bias keeps the random init balanced at 0.5)
        nn_inputs = self.normalizer(inputs.transport_nn_inputs(Wtot_MJ), inputs.ds_source_idx)
        coeffs = self._transport_coefficients(self.nn_transport(nn_inputs))
        nn_sources_out = self.nn_sources(nn_inputs)
        # Particle inventory in 1e21 electrons: ne20_line_avg * volume * 0.1
        inventory = 0.1 * inputs.ne20_line_avg * inputs.volume_approx
        coeffs["S_total"] = jax.nn.softplus(nn_sources_out[0:1]) * inventory / TAU_REF_S
        coeffs["gaussian_location"] = 0.8 * jax.nn.sigmoid(nn_sources_out[1:2])
        coeffs["gaussian_width"] = 0.02 + 0.38 * jax.nn.sigmoid(nn_sources_out[2:3])
        coeffs["electron_heat_fraction"] = 0.2 + 0.75 * jax.nn.sigmoid(nn_sources_out[3:4] - 0.4)

        # Edge boundary conditions as NN-predicted fractions, same floors and
        # negative temperature bias as ProfilePredictorTorax (see the rationale
        # comments there), with the temperature scale from the state-implied
        # stored energy instead of a measured betan
        nn_edge_out = self.nn_edge(nn_inputs)
        # Floor lowered 0.05 -> 0.01 with the profile predictor (ceiling probe
        # railed the old floor on 88% of MAST samples)
        coeffs["n_e_right_bc"] = (0.01 + 0.94 * jax.nn.sigmoid(nn_edge_out[0:1])) * inputs.ne20_line_avg  # [1e20 m^-3]
        te_scale = jnp.clip(inputs.te_approx_from_wtot(Wtot_MJ), 0.05, 5.0)
        coeffs["T_e_right_bc"] = 0.02 + jax.nn.sigmoid(nn_edge_out[1:2] - 5.0) * te_scale  # [keV]
        return coeffs

    def _build_provider_and_geo(
        self,
        inputs: Inputs,
        coeffs: dict,
        ne_ic: Array | None = None,
        te_ic: Array | None = None,
    ):
        """Runtime params provider and per-sample geometry.

        ne_ic / te_ic on rhogrid set the initial profile conditions (only
        consumed by get_initial_state, so callers that carry the TORAX state
        omit them).
        """
        ip_update = torax_experimental.TimeVaryingScalarUpdate(
            value=jnp.atleast_1d(inputs.Ip_MA * 1e6),
        )
        S_total_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["S_total"] * 1e21)
        # Measured auxiliary heating, absorption_fraction stays at its config value
        p_aux_update = torax_experimental.TimeVaryingScalarUpdate(value=jnp.atleast_1d(inputs.P_aux_MW * 1e6))
        gaussian_location_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_location"])
        gaussian_width_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_width"])
        electron_heat_fraction_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["electron_heat_fraction"])
        ne_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["n_e_right_bc"] * 1e20)
        te_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["T_e_right_bc"])

        mapping = {
            "profile_conditions.Ip": ip_update,
            "profile_conditions.n_e_right_bc": ne_right_bc_update,
            # Assume the ion edge temperature matches the electron edge temperature
            "profile_conditions.T_e_right_bc": te_right_bc_update,
            "profile_conditions.T_i_right_bc": te_right_bc_update,
            "sources.gas_puff.S_total": S_total_update,
            "sources.generic_heat.P_total": p_aux_update,
            "sources.generic_heat.gaussian_location": gaussian_location_update,
            "sources.generic_heat.gaussian_width": gaussian_width_update,
            "sources.generic_heat.electron_heat_fraction": electron_heat_fraction_update,
        }
        if ne_ic is not None and te_ic is not None:
            rho_ic = jnp.array(self.rhogrid)
            t_ic_update = torax_experimental.TimeVaryingArrayUpdate(value=te_ic[jnp.newaxis, :], rho_norm=rho_ic)
            n_ic_update = torax_experimental.TimeVaryingArrayUpdate(value=1e20 * ne_ic[jnp.newaxis, :], rho_norm=rho_ic)
            mapping["profile_conditions.T_e"] = t_ic_update
            # Assume the ion temperature matches the electron temperature
            mapping["profile_conditions.T_i"] = t_ic_update
            mapping["profile_conditions.n_e"] = n_ic_update
        mapping.update(self._transport_provider_mapping(coeffs))
        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(mapping)

        # Build JAX-differentiable geometry from per-sample inputs
        face_centers_np = np.array(self._face_centers)
        torax_mesh = torax_pydantic.Grid1D(face_centers=face_centers_np)
        rho_hires_norm_np = np.array(self._rho_hires_norm)
        if self.geometry_builder == "miller":
            geo = build_miller_geometry_jax(
                R_major=inputs.R0,
                a_minor=inputs.a_minor,
                B_0=inputs.B0,
                elongation_LCFS=inputs.kappa,
                delta_top=inputs.delta_top,
                delta_bot=inputs.delta_bot,
                torax_mesh=torax_mesh,
                rho_hires_norm_np=rho_hires_norm_np,
                delta_exponent=self.delta_exponent,
            )
        else:
            geo = build_circular_geometry_jax(
                R_major=inputs.R0,
                a_minor=inputs.a_minor,
                B_0=inputs.B0,
                elongation_LCFS=inputs.kappa,
                torax_mesh=torax_mesh,
                rho_hires_norm_np=rho_hires_norm_np,
            )
        geo_provider = geometry_provider_lib.ConstantGeometryProvider(geo=geo)
        return new_provider, geo_provider

    def _transport_provider_mapping(self, coeffs: dict) -> dict:
        """Runtime-params override entries for the configured transport model."""

        def scalar(name: str) -> torax_experimental.TimeVaryingScalarUpdate:
            return torax_experimental.TimeVaryingScalarUpdate(value=coeffs[name])

        if self.transport_model == "constant":
            # chi_i etc. are radial profiles (TimeVaryingArray) in the constant
            # model: broadcast the NN scalar to a flat profile
            rho = jnp.array([1.0])

            def flat_profile(name: str) -> torax_experimental.TimeVaryingArrayUpdate:
                return torax_experimental.TimeVaryingArrayUpdate(
                    value=jnp.broadcast_to(coeffs[name][:, jnp.newaxis], (1, 1)),
                    rho_norm=rho,
                )

            return {
                "transport_model.chi_i": flat_profile("chi_i"),
                "transport_model.chi_e": flat_profile("chi_e"),
                "transport_model.D_e": flat_profile("D_e"),
                "transport_model.V_e": flat_profile("V_e"),
            }
        elif self.transport_model == "cgm":
            return {
                "transport_model.chi_e_i_ratio": scalar("chi_e_i_ratio"),
                "transport_model.chi_D_ratio": scalar("chi_D_ratio"),
                "transport_model.VR_D_ratio": scalar("VR_D_ratio"),
                # Plain float leaves in the provider: replaced with traced scalars
                # directly rather than via TimeVaryingScalarUpdate
                "transport_model.alpha": jnp.squeeze(coeffs["alpha"]),
                "transport_model.chi_stiff": jnp.squeeze(coeffs["chi_stiff"]),
            }
        elif self.transport_model == "gyrobohm":
            # Same NN multiplier applied to both species: the BgB model already
            # fixes chi_i_B = 2 * chi_e_B and chi_i_gB = 0.5 * chi_e_gB
            return {
                "transport_model.chi_e_bohm_multiplier": scalar("chi_bohm_multiplier"),
                "transport_model.chi_i_bohm_multiplier": scalar("chi_bohm_multiplier"),
                "transport_model.chi_e_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
                "transport_model.chi_i_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
                "transport_model.D_face_c1": scalar("D_face_c1"),
                "transport_model.D_face_c2": scalar("D_face_c2"),
                "transport_model.V_face_coeff": scalar("V_face_coeff"),
            }
        else:  # qlknn
            # Plain float leaves in the provider (like cgm alpha and chi_stiff):
            # replaced with traced scalars directly
            return {
                "transport_model.ITG_flux_ratio_correction": jnp.squeeze(coeffs["ITG_flux_ratio_correction"]),
                "transport_model.ETG_correction_factor": jnp.squeeze(coeffs["ETG_correction_factor"]),
                "transport_model.collisionality_multiplier": jnp.squeeze(coeffs["collisionality_multiplier"]),
            }

    def _advance_one_step(self, sim_state, post_processed, provider, geo_provider):
        """Advance TORAX by exactly one solver step of sim_dt.

        Direct step_fn call, no relaxation loop: sim_dt is the fixed solver
        dt, enforced at init. The evolved core profiles are clamped before
        they are carried into the next call, same per-step clamp as the
        ProfilePredictorTorax relaxation loop (see clamp_core_profiles for
        the NaN rationale).

        The step is wrapped in jax.checkpoint: reverse-mode AD through an
        outer rollout scan recomputes the solver step instead of storing its
        residuals, which dominate training memory (measured 12 to 17 MB per
        step per sample at float64, so a 100-step rollout would need over
        1 GB per sample without remat, at a measured ~1.35x compute cost
        with it).
        """

        # Provider and geometry stay closure captures, matching the
        # wrap_body_in_checkpoint pattern in _run_loop_jit_with_geo: their
        # closed-over traced values are saved rather than rematerialized,
        # and they are small (NN coefficient scalars, mesh-sized geometry)
        def step(sim_state, post_processed):
            next_state, next_post = self.step_fn(
                sim_state,
                post_processed,
                runtime_params_overrides=provider,
                geo_overrides=geo_provider,
            )
            return clamp_core_profiles(next_state), next_post

        return jax.checkpoint(step, prevent_cse=False)(sim_state, post_processed)

    def _interp_profiles(self, sim_state, coeffs: dict, rho: Array) -> tuple[Array, Array]:
        """Interpolate TORAX cell-center core profiles onto rho.

        Augment the cell values with both endpoints before interpolating:
        jnp.interp flat-holds outside the data range, which would ignore the
        exact Dirichlet edge BC at rho = 1. Duplicating the innermost cell at
        rho = 0 encodes the zero-gradient axis condition.
        """
        # n_e is in m^-3 and T_e is in keV in TORAX
        ne = sim_state.core_profiles.n_e.value / 1e20
        te = sim_state.core_profiles.T_e.value
        rho_cells = jnp.asarray(self.rho_norm_grid)
        rho_full = jnp.concatenate([jnp.zeros(1), rho_cells, jnp.ones(1)])
        ne_full = jnp.concatenate([ne[:1], ne, coeffs["n_e_right_bc"]])
        te_full = jnp.concatenate([te[:1], te, coeffs["T_e_right_bc"]])
        return jnp.interp(rho, rho_full, ne_full), jnp.interp(rho, rho_full, te_full)


class TransportPredictorTorax(TransportPredictorToraxBase):
    """TORAX transport predictor carrying only the ne/te profiles.

    Each __call__ rebuilds a TORAX initial state with the stored profiles as
    initial condition, runs the simulation over one outer time step, and
    stores the evolved profiles.
    """

    @chex.dataclass
    class State:
        ne: Array = discrete_no_save_field(default=None)  # (n_rho,) [1e20 m^-3]
        te: Array = discrete_no_save_field(default=None)  # (n_rho,) [keV]

    def __call__(self, state: "TransportPredictorTorax.State", inputs: Inputs) -> tuple:
        rho = jnp.array(self.rhogrid)
        # Floor the carried profiles before they seed a TORAX state: measured
        # seeds can contain exact-zero te points (see TE_SEED_FLOOR_KEV)
        ne_state = jnp.maximum(state.ne, NE_SEED_FLOOR_20)
        te_state = jnp.maximum(state.te, TE_SEED_FLOOR_KEV)
        Wtot_MJ = wtot_from_profiles(ne_state, te_state, rho, inputs.volume_approx)
        coeffs = self._nn_coefficients(inputs, Wtot_MJ)

        # Initial condition from the stored profiles, edge point pinned to the NN Dirichlet BC
        # a discontinuity at the LCFS NaNs the solver under the critical gradient model
        te_ic = te_state.at[-1].set(jnp.squeeze(coeffs["T_e_right_bc"]))
        ne_ic = ne_state.at[-1].set(jnp.squeeze(coeffs["n_e_right_bc"]))
        provider, geo_provider = self._build_provider_and_geo(inputs, coeffs, ne_ic=ne_ic, te_ic=te_ic)

        initial_state, initial_post = torax_experimental.get_initial_state_and_post_processed_outputs(
            step_fn=self.step_fn,
            runtime_params_overrides=provider,
            geometry_overrides=geo_provider,
        )
        final_state, _post = self._advance_one_step(initial_state, initial_post, provider, geo_provider)

        ne_next, te_next = self._interp_profiles(final_state, coeffs, rho)
        ne_next, te_next, debug_info = self.positive_profiles(ne_next, te_next)
        debug_info.update(Wtot_MJ_state=Wtot_MJ)

        state_out = TransportPredictorTorax.State(ne=ne_next, te=te_next)
        # Output the profile estimate at the current time, the evolved
        # profiles become the next state
        output = Output(ne=state.ne, te=state.te, rho=rho, debug_info=debug_info)
        return state_out, output


class TransportPredictorToraxSimState(TransportPredictorToraxBase):
    """TORAX transport predictor carrying the full TORAX state.

    Alternative to TransportPredictorTorax: instead of rebuilding a TORAX
    initial state from stored ne/te each step, the whole ToraxSimState pytree
    (profiles, psi, currents) is carried as DISCRETE state, so nothing is
    lost to per-step reinitialization. The cost is TORAX internals inside the
    module state: seeding requires get_initial_state_and_post_processed_outputs
    and the state layout is tied to the torax config.
    """

    @chex.dataclass
    class State:
        sim_state: PyTree = discrete_no_save_field(default=None)  # ToraxSimState
        post_processed: PyTree = discrete_no_save_field(default=None)  # PostProcessedOutputs

    def __call__(self, state: "TransportPredictorToraxSimState.State", inputs: Inputs) -> tuple:
        rho = jnp.array(self.rhogrid)

        # Stored energy implied by the carried TORAX core profiles drives the
        # NN features, cell values suffice for the integral
        core_profiles = state.sim_state.core_profiles
        rho_cells = jnp.asarray(self.rho_norm_grid)
        Wtot_MJ = wtot_from_profiles(
            core_profiles.n_e.value / 1e20,
            core_profiles.T_e.value,
            rho_cells,
            inputs.volume_approx,
        )
        coeffs = self._nn_coefficients(inputs, Wtot_MJ)
        # No initial profile conditions: the state is carried, not rebuilt
        provider, geo_provider = self._build_provider_and_geo(inputs, coeffs)

        # step_fn evaluates time-varying params at the state time and clips
        # dt against the config t_final, and the carried state has already
        # advanced to t_final, so rewind t to t_initial before each step
        numerics = self.step_fn.runtime_params_provider.numerics
        sim_state = dataclasses.replace(
            state.sim_state,
            t=jnp.full_like(state.sim_state.t, float(numerics.t_initial)),
        )
        final_state, final_post = self._advance_one_step(sim_state, state.post_processed, provider, geo_provider)

        # Output the profile estimate at the current time from the carried state
        ne_now, te_now = self._interp_profiles(state.sim_state, coeffs, rho)
        ne_now, te_now, debug_info = self.positive_profiles(ne_now, te_now)
        debug_info.update(Wtot_MJ_state=Wtot_MJ)

        state_out = TransportPredictorToraxSimState.State(sim_state=final_state, post_processed=final_post)
        output = Output(ne=ne_now, te=te_now, rho=rho, debug_info=debug_info)
        return state_out, output


class TransportPredictorEnv(ModuleTrainingEnv):
    """Training environment shared by every transport predictor architecture.

    State seeding uses the measured signals at the segment start: the
    dataloader's state_init_vars carry the t0 slice of the profiles, the
    stored energy, and (for the TORAX sim-state variant) every scalar input,
    so create_state never has to slice the time axis itself.
    """

    module: TransportPredictor
    domain_adaptation: str = eqx.field(static=True, default="unset")
    # Sciml submodule names ("power_balance", "profile_predictor") excluded
    # from the trainable selection
    freeze_submodules: list[str] = eqx.field(static=True, default_factory=list)
    # Every architecture carries discrete state, so SIMPLE_EULER is required
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    def create_state(self, observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        # asarray: eager callers hand numpy-backed xr data, and the modules
        # index the carried state with jax-only ops (.at)
        ne0 = jnp.asarray(observations["ne20_rho"].data)
        te0 = jnp.asarray(observations["Te_keV_rho"].data)

        if isinstance(self.module, TransportPredictorTransformer):
            # Fill the whole history buffer with the measured initial profile,
            # as if the plasma had been sitting at that profile forever
            row = jnp.concatenate([ne0, te0])
            profiles = jnp.tile(row[jnp.newaxis, :], (self.module.history_len, 1))
            return TransportPredictorTransformer.State(profiles=profiles)

        if isinstance(self.module, TransportPredictorSciML):
            return PowerBalance.State(Wtot_MJ=observations["Wtot_MJ"].data)

        if isinstance(self.module, TransportPredictorTorax):
            # Seed floors: measured rampdown profiles can hold exact-zero te
            # points which NaN the TORAX solve (see TE_SEED_FLOOR_KEV)
            return TransportPredictorTorax.State(
                ne=jnp.maximum(ne0, NE_SEED_FLOOR_20),
                te=jnp.maximum(te0, TE_SEED_FLOOR_KEV),
            )

        if isinstance(self.module, TransportPredictorToraxSimState):
            # Build a full TORAX initial state from the measured profiles,
            # mirroring the per-step initial-condition flow of
            # TransportPredictorTorax.__call__ (edge points pinned to the NN
            # Dirichlet BCs so the solver never sees an LCFS discontinuity,
            # seeds floored against exact-zero te points in the measurements)
            module = self.module
            inputs0 = self.create_inputs(observations)
            rho = jnp.array(module.rhogrid)
            ne0 = jnp.maximum(ne0, NE_SEED_FLOOR_20)
            te0 = jnp.maximum(te0, TE_SEED_FLOOR_KEV)
            Wtot_MJ = wtot_from_profiles(ne0, te0, rho, inputs0.volume_approx)
            coeffs = module._nn_coefficients(inputs0, Wtot_MJ)
            te_ic = te0.at[-1].set(jnp.squeeze(coeffs["T_e_right_bc"]))
            ne_ic = ne0.at[-1].set(jnp.squeeze(coeffs["n_e_right_bc"]))
            provider, geo_provider = module._build_provider_and_geo(inputs0, coeffs, ne_ic=ne_ic, te_ic=te_ic)
            initial_state, initial_post = torax_experimental.get_initial_state_and_post_processed_outputs(
                step_fn=module.step_fn,
                runtime_params_overrides=provider,
                geometry_overrides=geo_provider,
            )
            return TransportPredictorToraxSimState.State(sim_state=initial_state, post_processed=initial_post)

        raise ValueError(f"Unknown transport predictor module type: {type(self.module)}")

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        # Top-level module inputs are always in physical units,
        # normalization happens inside the modules
        if isinstance(inputs, xr.Dataset):
            inputs = {var: inputs[var].data for var in inputs.data_vars}
        return Inputs(
            Ip_MA=inputs["Ip_MA"],
            B0=inputs["B0"],
            ne20_line_avg=inputs["ne20_line_avg"],
            R0=inputs["R0"],
            a_minor=inputs["a_minor"],
            kappa=inputs["kappa"],
            delta_top=inputs["delta_top"],
            delta_bot=inputs["delta_bot"],
            P_aux_MW=inputs["P_aux_MW"],
            ds_source_idx=inputs["ds_source_idx"],
        )

    def get_trainable(self):
        """Trainable leaves for the optimizer partition.

        Selects NN leaves explicitly, never whole modules: normalizer
        statistics are ordinary array leaves on every module and must stay
        frozen (a broad eqx.filter over a module would silently train them).
        The sciml power balance selection is delegated to PowerBalanceEnv so
        the two studies can never drift apart on what counts as trainable.
        """
        if self.domain_adaptation == "transfer":
            last_layer_leaves = []
            if isinstance(self.module, TransportPredictorTransformer):
                last_layer_leaves += [
                    self.module.head.layers[-1].weight,
                    self.module.head.layers[-1].bias,
                ]
            if isinstance(self.module, TransportPredictorToraxBase):
                for nn in (self.module.nn_transport, self.module.nn_sources, self.module.nn_edge):
                    last_layer_leaves += [nn.layers[-1].weight, nn.layers[-1].bias]
            if isinstance(self.module, TransportPredictorSciML):
                if "power_balance" not in self.freeze_submodules:
                    pb_env = PowerBalanceEnv(
                        module=self.module.power_balance,
                        domain_adaptation="transfer",
                        freeze_submodules=[],
                    )
                    last_layer_leaves += pb_env.get_trainable()
                if "profile_predictor" not in self.freeze_submodules:
                    nn = self.module.profile_predictor.nn
                    last_layer_leaves += [nn.layers[-1].weight, nn.layers[-1].bias]
            return last_layer_leaves

        trainable_leaves = {}
        if isinstance(self.module, TransportPredictorTransformer):
            for name in ("feature_embed", "profile_embed", "pos_embed", "attention", "head"):
                trainable_leaves[name] = eqx.filter(getattr(self.module, name), eqx.is_inexact_array)
        elif isinstance(self.module, TransportPredictorToraxBase):
            for name in ("nn_transport", "nn_sources", "nn_edge"):
                trainable_leaves[name] = eqx.filter(getattr(self.module, name), eqx.is_inexact_array)
        elif isinstance(self.module, TransportPredictorSciML):
            if "power_balance" not in self.freeze_submodules:
                pb_env = PowerBalanceEnv(
                    module=self.module.power_balance,
                    domain_adaptation=self.domain_adaptation,
                    freeze_submodules=[],
                )
                trainable_leaves["power_balance"] = pb_env.get_trainable()
            if "profile_predictor" not in self.freeze_submodules:
                trainable_leaves["profile_predictor"] = eqx.filter(self.module.profile_predictor.nn, eqx.is_inexact_array)

        return trainable_leaves
