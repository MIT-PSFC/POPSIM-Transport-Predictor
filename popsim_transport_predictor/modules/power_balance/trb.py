import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.dataloading import make_dataloaders

from popsim_transport_predictor.transfer_learning.orchestration.organize_data import (
    get_train_test_datasets_transfer,
    get_train_val_datasets,
)


class PowerBalanceTRB(TrainRunBuilder):
    """TrainRunBuilder for the power balance modules used in transfer learning"""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.

        For both standard learning and transfer learning, we essentially have two datasets.
        For standard learning, it's the standard train/val for hyperparameter tuning. No test is needed, so we can just return None.
        For transfer learning, we have the training dataset composed of all historic data and a small amount of new data,
        and the test dataset composed of a set amount of new data that is held out of training.
        We are not doing hyperparameter tuning for transfer learning.

        Also, slightly different from the POPSIM version, we're just returning the validation dataset.
        """

        if dataloader_config["transfer_learning"]:
            ds_train, ds_val = get_train_test_datasets_transfer(
                training_data_case=dataloader_config["training_data_case"],
                num_hp_shots=dataloader_config["num_hp_shots"],
                normalization_method=dataloader_config["normalization_method"],
            )
        else:
            ds_train, ds_val = get_train_val_datasets(
                training_data_case=dataloader_config["training_data_case"],
                normalization_method=dataloader_config["normalization_method"],
            )

        if "state_vars" in dataloader_config.keys():
            segment_lengths = [
                dataloader_config.get("segment_length_train", None),
                dataloader_config.get("segment_length_val", None),
            ]
            segment_overlaps = [
                dataloader_config.get("segment_overlap_train", None),
                dataloader_config.get("segment_overlap_val", None),
            ]
        else:
            segment_lengths = None
            segment_overlaps = None

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord="time",
            episode_coord="shot",
            input_vars=dataloader_config["input_vars"],
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars", None),
            state_init_vars=dataloader_config.get("state_vars", None),
            batch_size=dataloader_config.get("batch_size", None),
            segment_lengths=segment_lengths,
            segment_overlaps=segment_overlaps,
            shuffle=[True, False],
        )

        return ds_val, train_dl, val_dl, None
