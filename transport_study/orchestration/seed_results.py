"""Case results over independently seeded training runs.

Every production case trains config.num_seeds times.
Seed s sets the model initialization and the training shuffle (Study.launch_train),
keeps only its best-by-val-loss checkpoint and writes the test results of it (seed_result).
Once every seed has finished, Study.merge_seed_results combines them into the case result (aggregate_seed_results),
which keeps every test error variable of each seed on a seed dim.
Every case score then comes in three values (seed_statistics):
the best seed's (lowest validation loss), the mean over the seeds and the std over them,
the run-to-run spread a difference between cases has to beat.
"""

import math

import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml import Trainer

# The seeds of a case result, its coordinate holds the seed numbers
SEED_DIM = "seed"
# Attrs of one seed's result, the epoch of its best checkpoint and that checkpoint's validation loss
SEED_BEST_EPOCH_ATTR = "seed_best_epoch"
SEED_VAL_LOSS_ATTR = "seed_val_loss"
# Attr of a case result, the seed with the lowest validation loss
BEST_SEED_ATTR = "result_best_seed"
# Variables of a case result on SEED_DIM, each seed's best epoch and validation loss
SEED_BEST_EPOCH_VAR = "seed_best_epoch"
SEED_VAL_LOSS_VAR = "seed_val_loss"


def seed_result(trainer: Trainer, result_dict: dict) -> xr.Dataset:
    """The test results of one seed's best checkpoint, with its epoch and validation loss as attrs.

    Trainer.train already evaluated the best checkpoint on the test set (result_dict["test/study_results"]).
    """
    manager = trainer.checkpoint_manager
    best_epoch = manager.best_step()
    best_metrics = manager.metrics(best_epoch)
    seed_ds = result_dict["test/study_results"].copy()
    seed_ds.attrs[SEED_BEST_EPOCH_ATTR] = int(best_epoch)
    seed_ds.attrs[SEED_VAL_LOSS_ATTR] = float(best_metrics["loss"])
    return seed_ds


def lowest_val_loss_seed(per_seed: dict[int, xr.Dataset]) -> int:
    """The seed whose best checkpoint has the lowest validation loss.

    A NaN loss never wins, and with no finite loss at all the lowest seed number stands in.
    """
    val_losses = {seed: float(seed_ds.attrs[SEED_VAL_LOSS_ATTR]) for seed, seed_ds in per_seed.items()}
    finite_seeds = [seed for seed, val_loss in val_losses.items() if math.isfinite(val_loss)]
    if not finite_seeds:
        fallback_seed = min(per_seed)
        logger.warning(f"No seed has a finite validation loss ({val_losses}), seed {fallback_seed} stands in as the best")
        return fallback_seed
    return min(finite_seeds, key=val_losses.__getitem__)


def aggregate_seed_results(per_seed: dict[int, xr.Dataset]) -> xr.Dataset:
    """Combine the seed results of one case into its case result.

    Every data variable whose name contains "error" is stacked over the seeds on SEED_DIM, in seed order,
    stored as float32 to bound the file size.
    Prediction and target variables stay the best seed's:
    a mean over seeds of predicted trajectories would be smoother than any actual model
    and mislead the case report plots, which rank by the best seed's errors to match.
    Each seed's best epoch and validation loss land on SEED_DIM too, the best seed in the BEST_SEED_ATTR attr.
    """
    best_seed = lowest_val_loss_seed(per_seed)
    seeds = sorted(per_seed)
    out = per_seed[best_seed].copy()
    for name in [str(var) for var in out.data_vars]:
        if "error" not in name:
            continue
        stacked = xr.concat([per_seed[seed][name] for seed in seeds], dim=SEED_DIM)
        out[name] = stacked.astype(np.float32)
    best_epochs = [per_seed[seed].attrs[SEED_BEST_EPOCH_ATTR] for seed in seeds]
    val_losses = [per_seed[seed].attrs[SEED_VAL_LOSS_ATTR] for seed in seeds]
    out[SEED_BEST_EPOCH_VAR] = (SEED_DIM, np.array(best_epochs, dtype=int))
    out[SEED_VAL_LOSS_VAR] = (SEED_DIM, np.array(val_losses, dtype=float))
    out = out.assign_coords({SEED_DIM: seeds})
    out.attrs = {key: value for key, value in out.attrs.items() if key not in (SEED_BEST_EPOCH_ATTR, SEED_VAL_LOSS_ATTR)}
    out.attrs[BEST_SEED_ATTR] = int(best_seed)
    return out


def seed_statistics(per_seed: xr.Dataset, best_seed: int) -> xr.Dataset:
    """The three values of every variable of per_seed on SEED_DIM, one case score per seed.

    <name> is the mean over the seeds, <name>_best the best seed's value
    and <name>_seed_std the sample std over the seeds (ddof 1, the seeds are independent runs).
    The mean and std skip a seed whose score is NaN.
    """
    out = {}
    for name, per_seed_values in per_seed.data_vars.items():
        out[name] = per_seed_values.mean(SEED_DIM)
        out[f"{name}_best"] = per_seed_values.sel({SEED_DIM: best_seed}, drop=True)
        out[f"{name}_seed_std"] = per_seed_values.std(SEED_DIM, ddof=1)
    return xr.Dataset(out)
