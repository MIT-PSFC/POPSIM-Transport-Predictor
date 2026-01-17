import numpy as np
import xarray as xr
from popsim.ml.split_utils import split_dataset_by_fracs
from popsim.modules.transport_predictor.data import get_ds

from popsim_transport_predictor.popsim_study.config import config
from popsim_transport_predictor.popsim_study.orchestration import (
    HP_SHOTS_INCLUDED,
    TRAIN_VAL_TEST_SPLIT,
)


def add_performance(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Add performance metric to dataset
    We are saying performance is (Wtot_MJ^2 + Ip_MA^2)**0.5 for now
    """
    ds["performance"] = (ds["Wtot_MJ"] ** 2 + ds["Ip_MA"] ** 2) ** 0.5
    return ds


def get_train_val_test_datasets(
    training_data_case: str,
):
    """
    Split dataset into training, validation, and test sets based on the specified case.
    """

    if training_data_case in ["cmod", "tcv", "d3d_lp"]:
        ds, episode_coord = get_ds(
            config[f"{training_data_case}_dataset_path"], debug=config["debug"]
        )
        ds = add_performance(ds)
        train_ds, val_ds, test_ds = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=config["dataloader_prng_seed"],
            sortby="performance",
        )

    else:
        ds_cmod, episode_coord = get_ds(
            config["cmod_dataset_path"], debug=config["debug"]
        )
        ds_cmod = add_performance(ds_cmod)
        train_ds_cmod, val_ds_cmod, test_ds_cmod = split_dataset_by_fracs(
            ds_cmod,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=config["dataloader_prng_seed"],
            sortby="performance",
        )
        train_ds_cmod = train_ds_cmod.assign_coords(ds_source="cmod")
        val_ds_cmod = val_ds_cmod.assign_coords(ds_source="cmod")
        test_ds_cmod = test_ds_cmod.assign_coords(ds_source="cmod")

        ds_tcv, episode_coord = get_ds(
            config["tcv_dataset_path"], debug=config["debug"]
        )
        ds_tcv = add_performance(ds_tcv)
        train_ds_tcv, val_ds_tcv, test_ds_tcv = split_dataset_by_fracs(
            ds_tcv,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=config["dataloader_prng_seed"],
            sortby="performance",
        )
        train_ds_tcv = train_ds_tcv.assign_coords(ds_source="tcv")
        val_ds_tcv = val_ds_tcv.assign_coords(ds_source="tcv")
        test_ds_tcv = test_ds_tcv.assign_coords(ds_source="tcv")

        if training_data_case == "cmod_tcv":
            train_ds = xr.concat([train_ds_cmod, train_ds_tcv], dim=episode_coord)
            val_ds = xr.concat([val_ds_cmod, val_ds_tcv], dim=episode_coord)
            test_ds = xr.concat([test_ds_cmod, test_ds_tcv], dim=episode_coord)

        elif training_data_case == "cmod_tcv_d3d_lp":
            ds_d3d_lp, episode_coord = get_ds(
                config["d3d_lp_dataset_path"], debug=config["debug"]
            )
            ds_d3d_lp = add_performance(ds_d3d_lp)
            train_ds_d3d_lp, val_ds_d3d_lp, test_ds_d3d_lp = split_dataset_by_fracs(
                ds_d3d_lp,
                fracs=TRAIN_VAL_TEST_SPLIT,
                dim=episode_coord,
                seed=config["dataloader_prng_seed"],
                sortby="performance",
            )
            train_ds_d3d_lp = train_ds_d3d_lp.assign_coords(ds_source="d3d_lp")
            val_ds_d3d_lp = val_ds_d3d_lp.assign_coords(ds_source="d3d_lp")
            test_ds_d3d_lp = test_ds_d3d_lp.assign_coords(ds_source="d3d_lp")

            train_ds = xr.concat(
                [train_ds_cmod, train_ds_tcv, train_ds_d3d_lp], dim=episode_coord
            )
            val_ds = xr.concat(
                [val_ds_cmod, val_ds_tcv, val_ds_d3d_lp], dim=episode_coord
            )
            test_ds = xr.concat(
                [test_ds_cmod, test_ds_tcv, test_ds_d3d_lp], dim=episode_coord
            )

        else:
            raise ValueError(f"Unknown training data case: {training_data_case}")

    return train_ds, val_ds, test_ds


def get_train_test_datasets_transfer(
    training_data_case: str,
    num_hp_shots: int,
):
    """
    Split dataset into training, validation, and test sets for transfer learning case.
    The number of high-performance shots included in training is specified by `num_hp_shots`.
    """

    # Load the high-performance dataset and split into train/test
    # No validation needed because we are not tuning hyperparameters on transfer learning data
    ds_hp, episode_coord = get_ds(config.d3d_hp_dataset_path, debug=config["debug"])
    ds_hp = add_performance(ds_hp)
    ds_hp = ds_hp.assign_coords(ds_source="d3d_hp")
    sorted_shots = np.argsort(ds_hp[episode_coord].values)

    max_train_size = len(HP_SHOTS_INCLUDED)
    if num_hp_shots > max_train_size:
        raise ValueError(
            f"num_hp_shots {num_hp_shots} exceeds maximum available {max_train_size}"
        )

    train_shot_pool = sorted_shots[:max_train_size]
    test_shot_pool = sorted_shots[max_train_size:]

    test_ds = ds_hp.isel({episode_coord: test_shot_pool})
    train_ds_hp = ds_hp.isel({episode_coord: train_shot_pool[:num_hp_shots]})

    if training_data_case == "exnihilo":
        train_ds = train_ds_hp
    else:
        # Load historic data and put it all in the training set
        train_ds_hist, val_ds_hist, test_ds_hist = get_train_val_test_datasets(
            training_data_case
        )
        train_ds = xr.concat(
            [train_ds_hist, val_ds_hist, test_ds_hist, train_ds_hp], dim=episode_coord
        )

    return train_ds, test_ds
