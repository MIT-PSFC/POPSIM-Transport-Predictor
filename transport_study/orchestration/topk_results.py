"""Final study results over the retained top-K checkpoints.

Best-checkpoint selection is a val-loss argmin over epochs, and GPU float
noise between reruns can flip which of several near-tied epochs wins,
moving the reported test metrics discontinuously (observed swings up to
0.25 rel_mean for the physics power balance models). Production training
runs therefore keep the config.num_result_checkpoints best-by-val-loss
checkpoints, and the case result written to disk keeps every test error variable of each of them on a ckpt dim.
Every case score then comes in three values (topk_statistics):
the best checkpoint's, the mean over the top-K checkpoints (the expected value under argmin noise),
and the std over them (the noise floor for case comparisons).
"""

import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainConfig, Trainer
from popsim.ml.checkpointing import restore_model
from popsim.ml.eval import run_evals
from popsim.ml.launch import get_train_run_builder_class

# Retained checkpoints of a case result, its coordinate holds their epochs
CKPT_DIM = "ckpt"
BEST_EPOCH_ATTR = "result_checkpoint_best_epoch"


def aggregate_topk_results(per_step: dict[int, xr.Dataset], best_step: int) -> xr.Dataset:
    """Combine per-checkpoint study_results datasets into one case result.

    Every data variable whose name contains "error" is stacked over the checkpoints on CKPT_DIM,
    whose coordinate is the epochs in order, stored as float32 to bound the file size.
    Prediction and target variables stay those of the best checkpoint:
    a mean over checkpoints of predicted trajectories would be smoother than any actual model
    and mislead the case report plots, which rank by the best checkpoint's errors to match.

    The best epoch lands in the BEST_EPOCH_ATTR attr.
    """
    out = per_step[best_step].copy()
    steps = sorted(per_step)
    for name in [str(var) for var in out.data_vars]:
        if "error" not in name:
            continue
        stacked = xr.concat([per_step[step][name] for step in steps], dim=CKPT_DIM)
        out[name] = stacked.assign_coords({CKPT_DIM: steps}).astype(np.float32)
    out.attrs[BEST_EPOCH_ATTR] = int(best_step)
    return out


def topk_statistics(per_ckpt: xr.Dataset, best_epoch: int) -> xr.Dataset:
    """The three top-K values of every variable of per_ckpt on CKPT_DIM, one case score per checkpoint.

    <name> is the mean over the checkpoints, <name>_best the best epoch's value
    and <name>_ckpt_std the std over the checkpoints (ddof 0).
    The mean and std skip a checkpoint whose score is NaN.
    """
    out = {}
    for name, per_ckpt_values in per_ckpt.data_vars.items():
        out[name] = per_ckpt_values.mean(CKPT_DIM)
        out[f"{name}_best"] = per_ckpt_values.sel({CKPT_DIM: best_epoch}, drop=True)
        out[f"{name}_ckpt_std"] = per_ckpt_values.std(CKPT_DIM)
    return xr.Dataset(out)


def topk_checkpoint_steps(manager, num_checkpoints: int) -> list[int]:
    """The num_checkpoints retained steps with the lowest validation loss, in epoch order.

    The checkpoint dir also keeps the latest step for resuming,
    which only counts when it ranks in the top num_checkpoints.
    Steps saved without a validation loss never rank.
    """
    step_losses = {}
    for step in manager.all_steps():
        step_metrics = manager.metrics(step)
        if step_metrics is None:
            continue
        step_losses[step] = step_metrics["loss"]
    steps_by_loss = sorted(step_losses, key=step_losses.__getitem__)
    return sorted(steps_by_loss[:num_checkpoints])


def compute_topk_study_results(
    trainer: Trainer,
    test_dl: DataLoader,
    train_config: TrainConfig,
    result_dict: dict,
) -> xr.Dataset:
    """Evaluate the test suite on the top-K checkpoints and aggregate.

    K is the train config's checkpoint_max_to_keep.
    The trainer already evaluated the best checkpoint at the end of
    Trainer.train (result_dict["test/study_results"]), so that dataset is
    reused and only the remaining top-K checkpoints are evaluated.
    With K = 1 the result still carries a length-1 ckpt dim, so every consumer reads one layout.
    """
    best_ds = result_dict["test/study_results"]
    manager = trainer.checkpoint_manager
    steps = topk_checkpoint_steps(manager, train_config.checkpoint_max_to_keep)
    best_step = manager.best_step()
    per_step = {best_step: best_ds}
    if len(steps) <= 1:
        return aggregate_topk_results(per_step, best_step)
    train_run_builder = get_train_run_builder_class(train_config.train_run_builder)
    eval_suite = train_run_builder.get_test_eval_suite(train_config.test_eval_suite_config)
    logger.info(f"Evaluating test suite on the top {len(steps)} checkpoints (epochs {steps}, best {best_step})")
    for step in steps:
        if step == best_step:
            continue
        model = restore_model(manager, trainer.train_state.model, step=step)
        per_step[step] = run_evals(model, test_dl, eval_suite)["study_results"]
    return aggregate_topk_results(per_step, best_step)
