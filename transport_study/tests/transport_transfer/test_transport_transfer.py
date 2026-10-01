"""End-to-end and wiring tests for the transport transfer study.

Mirror tests/power_balance_transfer/test_power_balance_transfer.py: a loaded
TransportStudy.Config on the sample datasets (cmod-low1 as source, cmod-high
as target, small target_test_set_size), tiny epoch counts.

Test implementations are deliberately blocked out as stubs, see the
repository convention.
"""


def test_get_ds_transport_transfer():
    """get_ds(source, "transport_transfer") returns a dataset that keeps the
    profile signals interpolated onto the shared RHO_GRID, computes power_additional_MW
    from the zero-filled per-system aux power signals, retains energy_mhd_MJ,
    power_ohm_MW/power_radiated_MW (anchor targets for the sciml training loss),
    triangularity_upper/triangularity_lower, fresh_profile, and the t_e_shape/n_e_shape variables,
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
    _make_submodule_config reaches depth two). Every transport model type's
    target_vars carry the profile error-bar companions and fresh_profile
    (TRANSPORT_TARGET_VARS, read by the loss), the sciml target_vars
    additionally carry energy_mhd_MJ/power_ohm_MW/power_radiated_MW for the anchor terms, the
    loss config carries the anchor_weight_energy_mhd/_p_oh/_p_rad keys, and the
    optimizer config carries submodule_lr_factors for the power_balance
    subtree."""


def test_loss_fn_sciml_anchor_terms():
    """The training loss adds plain absolute-error anchor terms pulling the
    Output's energy_mhd_MJ_pred/power_ohm_MW_pred/power_radiated_MW_pred toward the measured
    energy_mhd_MJ/power_ohm_MW/power_radiated_MW targets, scaled by anchor_weight_energy_mhd/_p_oh/
    _p_rad and the per-device sample weights; a term drops out when its
    weight is zero or its signal is absent from the targets (transformer and
    torax target_vars), the anchors are exempt from the fresh_profile mask
    (their signals are measured at every timeslice), and the validation loss
    stays pure profile error."""


def test_optimizer_grouped_lr_and_clip():
    """TransportPredictorTRB.get_optimizer applies the global-norm clip and,
    with submodule_lr_factors set, runs every trainable leaf whose pytree
    path contains power_balance (including its nested p_oh/p_rad submodules)
    at the scaled schedule while profile_predictor leaves keep the full
    learning rate, for both the da=none and the transfer last-layer
    partitions; without factors it reduces to the plain clipped AdamW."""


def test_make_train_config_transfer_lr_scaling():
    """make_train_config for a transfer case points model_init at the
    transfer_pretrain twin's checkpoint (physics-coral is a stat
    normalization) and scales lr0/lrf down by TRANSFER_LR_FACTOR after the
    tuned merge."""


def test_env_create_state_per_model_type():
    """TransportPredictorEnv.create_state seeds the right state per module:
    transformer gets a (history_len, 2 n_rho) buffer tiled from the measured
    t0 profiles, sciml gets PowerBalance.State with the measured energy_mhd_MJ,
    torax rebuild gets ne/te from the measured profiles, and torax carry gets
    a full ToraxSimState built from the measured profiles with the edge points
    pinned to the NN boundary conditions."""


def test_env_get_trainable_selections():
    """get_trainable never includes normalizer statistics; for transfer it
    returns only last-layer leaves (transformer head, the three torax MLPs,
    sciml taue/profile last layers); for sciml with freeze_submodules both
    submodules drop out of the selection entirely; for the transformer at
    da=none the selection is exactly feature_embed, profile_embed, pos_embed,
    attention and head."""


def test_transformer_position_embedding_orders_history():
    """The transformer output must change when the profile history buffer is
    reversed (position embedding breaks permutation invariance), and
    pos_embed has shape (history_len, d_model) with a small nonzero init so
    the t0-seeded constant buffer still yields slot-distinguishable tokens."""


def test_transformer_history_holds_profiles_only():
    """The rolling State buffer contains only predicted ne/te profile rows,
    (history_len, 2 n_rho): past input features are never stored, only the
    current timestep's features form the attention query."""


def test_loss_fn_shapes_and_weighting():
    """The transport loss peak-normalizes each channel per timeslice, applies
    the huber delta on the normalized residual, weights samples by device via
    ds_source_idx, masks the profile terms by fresh_profile, and returns a
    scalar; the validation variant is delta-free error-bar-softened absolute
    error and both handle a (time, rho) prediction against xr-backed
    targets. The fresh-mask cases are implemented in test_loss_fn.py."""


def test_val_loss_error_bar_softening_only():
    """The validation loss down-weights the part of the residual inside the
    n_e_1e20_error/t_e_keV_error bars by within_error_weight while the
    part beyond the bar keeps full weight, the training loss ignores the
    error bars entirely, and an absent or zero error var reduces the
    validation loss to the plain absolute error (the 0 sentinel means a
    zero-width bar)."""


def test_huber_delta_train_loss_only():
    """The swept huber_delta changes the training loss but never the
    validation loss (delta-free absolute error), so the sweep metric
    val/loss.mean cannot be gamed by shrinking delta."""


def test_transformer_training_smoke():
    """A transformer case trains end to end on the sample datasets for a
    couple of epochs and writes a result file containing n_e_1e20_pred /
    t_e_keV_pred plus the error_abs_ts / error_rel_ts / error_abs_shot /
    error_rel_shot variables the base _summarize_case_errors reads."""


def test_torax_p_aux_feed_through():
    """The measured power_additional_MW input is wired directly to the TORAX
    generic_heat.P_total runtime update (MW to W) for the rebuild variant,
    the carry variant, and the env's initial TORAX state construction; the
    sources network predicts only the deposition shape (gaussian_location,
    gaussian_width, electron_heat_fraction), the gas-puff fueling, and the
    absorption_fraction, so changing power_additional_MW changes the one-step output
    with the module weights held fixed, and no NN output can override the
    measured injected heating magnitude."""


def test_torax_absorption_fraction_nn():
    """The transport sources network's last output sets absorption_fraction
    via the saturating Beer-Lambert form 1 - exp(-n_e_line_average_1e20 * softplus(nn
    output)): always in (0, 1), linear in line density when optically thin,
    smoothly saturating toward 1 with no gradient-dead cap. The value reaches
    the generic_heat.absorption_fraction runtime update in
    _build_provider_and_geo, the absorbed power inside TORAX equals P_total *
    absorption_fraction, and the profile predictor's TORAX modules keep the
    fixed config value 0.9."""


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
    energy_mhd_MJ, drops rows with NaN device index, and returns identity stats for
    devices with too few samples."""
