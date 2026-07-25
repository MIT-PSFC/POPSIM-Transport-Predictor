"""Tests for the universal transfer_pretrain twin (Study.Case.transfer_pretrain_case).

Every transfer case now pretrains through a dedicated transfer_pretrain twin,
selected on the target test set. Stateless normalizations (raw, physics) share
one twin at HYPERPARAM_TARGET_SHOTS, stat normalizations keep per-n twins.
Test bodies are blocked out with docstrings for manual implementation.
"""


def test_stateless_norm_twin_shared_across_target_shot_counts():
    """raw / physics transfer cases at different num_target_shots share one twin.

    transfer_pretrain_case() for n = 1 and n = 316 must return the same case
    (da = transfer_pretrain, num_target_shots = HYPERPARAM_TARGET_SHOTS), and
    make_cases must contain exactly one such twin per
    model_type x training_data x normalization x freeze combination.
    """


def test_stat_norm_twin_keeps_num_target_shots():
    """zscore / coral / physics-* twins still carry the transfer case's n.

    Guards the pre-existing behavior: the stat twin must differ per
    num_target_shots because the normalizer statistics are fitted on
    historic + n target shots.
    """


def test_impossibility_rules():
    """n = 0 possibility depends on domain adaptation and normalization.

    - da = transfer with n = 0: impossible (nothing to finetune on)
    - da = transfer_pretrain, stat normalization, n = 0: impossible
      (twin exists to fit target-aware stats)
    - da = transfer_pretrain, raw / physics, n = 0: possible (the shared twin)
    """


def test_finetune_checkpoint_points_at_twin_not_baseline():
    """make_train_config for a raw / physics transfer case wires the twin checkpoint.

    model_init_config["transfer_checkpoint"] must be the twin's model dir
    (targ_0.da_transfer_pretrain suffix), not the plain baseline case dir.
    """


def test_twin_trains_from_scratch_with_swept_schedule():
    """The twin's own train config has no transfer_checkpoint and no LR scaling.

    make_train_config for the twin must leave optimizer_config at the tuned
    values (transfer LR budgeting applies only to da = transfer) and must not
    set model_init_config["transfer_checkpoint"].
    """


def test_zero_shot_pretrain_datasets():
    """get_transfer_pretrain_datasets with num_target_shots = 0 is well formed.

    Train set = full historic data (train + val splits), normalizer-fit set
    equals the historic set (no target shots appended), val set = the
    target test set. No crash from concatenating the empty target selection.
    """


def test_submodule_twins_follow_parent():
    """A sciml-taue-nn / sciml-taue-scalinglaw twin has p_oh / p_rad twin prereqs at the same n.

    For a physics sciml-taue-nn transfer case the twin's prereqs must include
    p_oh / p_rad cases with da = transfer_pretrain and targ_0, pinned to the
    hyperparam freeze value.
    """
