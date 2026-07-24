"""End-to-end and wiring tests for the transport transfer study.

Mirror tests/power_balance_transfer/test_power_balance_transfer.py: a loaded
TransportStudy.Config on the sample datasets (cmod-low1 as source, cmod-high
as target, small target_test_set_size), tiny epoch counts.

Test implementations are deliberately blocked out as stubs, see the
repository convention.
"""


def test_get_ds_transport_transfer():
    """get_ds(source, "transport_transfer") returns a dataset that keeps the
    profile signals interpolated onto the shared RHO_GRID, computes P_aux_MW
    from the zero-filled per-system aux power signals, retains Wtot_MJ,
    delta_top/delta_bot, fresh_profiles, and the Te_shape/ne_shape variables,
    clips the profile error signals at zero, and is reindexed to the uniform
    1 kHz timebase (mid-shot gaps become NaN slices). Also belongs as a
    parametrization of TestGetDs in tests/orchestration/test_organize_data.py."""


def test_transport_transfer_cases():
    """make_cases enumerates the expected grid: every (model_type,
    training_data, domain_adaptation, num_target_shots) combination subject to
    the skip rules (freeze_submodules only varies for sciml, geometry_builder
    and torax_state only vary for torax-*), plus the deduped prereq chain:
    each sciml axis combination pulls in exactly one power_balance, one
    profile, one p_oh, and one p_rad case, and hyperparam prereq cases are
    shared across the grid."""


def test_make_train_config_sciml_submodule_wiring():
    """make_train_config for a sciml case nests full TrainConfigs for the
    power_balance and profile submodules (model_init_config["submodules"]),
    each with checkpoint_dir pointing at that prereq case's trained model dir,
    restore_submodules True, and the power_balance submodule config itself
    nesting p_oh/p_rad submodule configs (recursion through
    _make_submodule_config reaches depth two)."""


def test_make_train_config_transfer_lr_scaling():
    """make_train_config for a transfer case points model_init at the
    transfer_pretrain twin's checkpoint (physics-coral is a stat
    normalization) and scales lr0/lrf down by TRANSFER_LR_FACTOR after the
    tuned merge."""


def test_env_create_state_per_model_type():
    """TransportPredictorEnv.create_state seeds the right state per module:
    transformer gets a (history_len, 2 n_rho) buffer tiled from the measured
    t0 profiles, sciml gets PowerBalance.State with the measured Wtot_MJ,
    torax rebuild gets ne/te from the measured profiles, and torax carry gets
    a full ToraxSimState built from the measured profiles with the edge points
    pinned to the NN boundary conditions."""


def test_env_get_trainable_selections():
    """get_trainable never includes normalizer statistics; for transfer it
    returns only last-layer leaves (transformer head, the three torax MLPs,
    sciml taue/profile last layers); for sciml with freeze_submodules both
    submodules drop out of the selection entirely."""


def test_loss_fn_shapes_and_weighting():
    """The transport loss peak-normalizes each channel per timeslice, applies
    the huber delta on the normalized residual, weights samples by device via
    ds_source_idx, and returns a scalar; the validation variant is delta-free
    absolute error and both handle a (time, rho) prediction against xr-backed
    targets."""


def test_transformer_training_smoke():
    """A transformer case trains end to end on the sample datasets for a
    couple of epochs and writes a result file containing ne20_rho_pred /
    Te_keV_rho_pred plus the error_abs_ts / error_rel_ts / error_abs_shot /
    error_rel_shot variables the base _summarize_case_errors reads."""


def test_torax_rebuild_and_carry_smoke():
    """A torax-cgm case with torax_state rebuild and one with carry both
    advance a few training steps without NaN loss (the carry variant
    exercises the TORAX initial-state construction in the env)."""


def test_collect_results_schema():
    """collect_results returns one row per finished case along case_idx with
    the err_E_D_S summary variables and the case coords model_type /
    training_data / domain_adaptation / freeze_submodules / geometry_builder /
    torax_state / num_target_shots."""


def test_normalizer_fit_features():
    """_fit_transport_input_normalizer builds an (N, 11) feature matrix whose
    columns match Inputs.transport_nn_inputs evaluated with the measured
    Wtot_MJ, drops rows with NaN device index, and returns identity stats for
    devices with too few samples."""
