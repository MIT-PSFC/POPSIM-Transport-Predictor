"""Tests for the step-budgeted transfer learning-rate schedule.

Covers Study._scale_transfer_lr and Study._transfer_steps_per_epoch in
transport_study/orchestration/study.py. Test bodies are blocked out with
docstrings for manual implementation.
"""


def test_step_starved_finetune_keeps_full_tuned_lr():
    """A finetune with 1 optimizer step per epoch gets scale 1.0.

    Build a transfer train_config whose measured train dataloader has fewer
    samples than batch_size (the power balance regime). _scale_transfer_lr
    must leave lr0 at the tuned value and set lrf equal to lr0.
    """


def test_step_rich_finetune_clips_to_floor():
    """A finetune with steps_per_epoch at or above 1 / TRANSFER_LR_FLOOR cools to the floor.

    With 10 or more steps per epoch, scale = max_epochs / total_steps drops to
    or below TRANSFER_LR_FLOOR and must clip there, reproducing the old fixed
    0.1 factor for step-rich finetunes like the profile study's.
    """


def test_intermediate_scale_is_inverse_steps_per_epoch():
    """Between the clips, scale equals 1 / steps_per_epoch independent of max_epochs.

    E.g. 4 steps per epoch gives scale 0.25 because total_steps =
    steps_per_epoch * max_epochs cancels the max_epochs numerator.
    """


def test_transfer_schedule_is_flat():
    """The returned optimizer_config has lrf == lr0.

    Optionally also check make_exponential_adamw's schedule evaluates to the
    same learning rate at step 0 and at a step count far past
    transition_steps, confirming optax exponential_decay clamps to constant.
    """


def test_steps_per_epoch_measured_from_dataloader():
    """_transfer_steps_per_epoch returns len(train_dl) from the TRB's get_dataloaders.

    Point the dataloader config at a small synthetic dataset and compare
    against the popsim DataLoader length directly (drop_last keeps a single
    partial batch, so tiny datasets still count 1). Also cover the
    data_train_run_builder override taking precedence over
    train_config.train_run_builder.
    """


def test_steps_per_epoch_cached_per_dataloader_config():
    """Repeated calls with the same dataloader config do not rebuild dataloaders.

    Two transfer cases sharing a dataloader config (e.g. freeze twins) must
    hit the same cache entry (assert get_dataloaders called once, e.g. via
    monkeypatch counter), while a changed batch_size must miss the cache.
    """


def test_non_transfer_cases_keep_swept_schedule():
    """make_train_config leaves optimizer_config untouched for non-transfer cases.

    For domain_adaptation None, weighted, and addition the tuned lr0, lrf,
    and decay settings must pass through unmodified, guarding against the
    transfer-only scaling leaking into other cases.
    """
