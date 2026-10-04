"""Literal case-name regression tests for the transport transfer study.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb
projects, and SLURM job names, so its format must never drift. Mirror
tests/power_balance_transfer/test_power_balance_case_naming.py: a loaded
TransportStudy.Config on the sample datasets (cmod-low1 as source, cmod-high
as target) and a _case helper with defaults model_type="sciml",
training_data="cmod-low1", domain_adaptation=None, freeze_submodules=True,
num_target_shots=0, geometry_builder="circular", torax_state="rebuild".

Test implementations are deliberately blocked out as stubs, see the
repository convention.
"""


def test_source_trained_case_name():
    """str of the default sciml case is exactly
    "case.sciml.td_cmod-low1.freeze_True": no norm_ token (normalization is
    not a case axis in this study), and the suppressed geometry_builder /
    torax_state defaults produce no geom_ / tstate_ tokens."""


def test_exnihilo_case_name():
    """str of a transformer exnihilo case with num_target_shots=5 is exactly
    "case.transformer.td_exnihilo.freeze_True.targ_5"."""


def test_domain_adaptation_case_name():
    """str of the default case with domain_adaptation="transfer" and
    num_target_shots=3 is exactly
    "case.sciml.td_cmod-low1.freeze_True.targ_3.da_transfer"."""


def test_torax_miller_case_name():
    """str of a torax-gyrobohm case with geometry_builder="miller" is exactly
    "case.torax-gyrobohm.td_cmod-low1.freeze_True.geom_miller": the non-default
    geometry gets its token, the default torax_state stays suppressed."""


def test_torax_carry_case_name():
    """str of a torax-gyrobohm case with torax_state="carry" is exactly
    "case.torax-gyrobohm.td_cmod-low1.freeze_True.tstate_carry": the non-default
    state carry gets its token, the default geometry stays suppressed."""


def test_torax_miller_carry_token_order():
    """A torax case with both geometry_builder="miller" and torax_state="carry"
    emits the tokens in STR_TOKEN_FIELDS order:
    "case.torax-gyrobohm.td_cmod-low1.freeze_True.geom_miller.tstate_carry"."""


def test_submodule_prereq_case_names():
    """The sciml case's prereqs contain a "case.power_balance.td_..." and a
    "case.profile.td_..." case, and the power_balance prereq's own prereqs
    contain "case.p_oh.td_..." and "case.p_rad.td_..." cases (chained
    prereq depth of two)."""


def test_non_torax_rejects_torax_axes():
    """Constructing a transformer case with geometry_builder="miller" or
    torax_state="carry" raises ValueError: those axes only apply to torax-*
    model types."""


def test_cases_are_hashable_and_set_stable():
    """The default case is in a set containing an identically constructed
    case, and both hash equal (the __hash__ = Study.Case.__hash__ alias
    survived the dataclass decorator)."""


def test_compatible_configs():
    """is_compatible is True for a config differing only in case-grid axes
    (model_types), and False when any COMPAT_HYPERPARAM_FIELDS entry differs
    (data_normalization, power_balance_model_type, profile_model_type,
    power_balance_data_normalization, hyperparam_* fields)."""


def test_base_config_rejects_subclass_fields():
    """The base StudyConfig raises a validation error when given
    TransportStudy.Config-only fields such as torax_state_options (extra
    fields are forbidden)."""
