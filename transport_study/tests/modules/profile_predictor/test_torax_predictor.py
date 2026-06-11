import pytest
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, load_config
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)


@pytest.mark.parametrize("transport_model", ["constant", "cgm", "gyrobohm"])
def test_torax_predictor(transport_model):
    config = StudyConfig(
        study_name=f"test_torax_predictor_{transport_model}",
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod-low1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod-high.nc",
        },
        target_device="cmod-high",
    )
    load_config(config)

    train_config = TrainConfig(**PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model])
    training_data = {
        "sources_unsorted": ["cmod-low"],
        "exnihilo": False,
    }
    train_config = train_config.model_copy(
        update={
            "project": config.study_name,
            "max_epochs": 4,
            "epochs_per_val": 2,
            "dataloader_config": {
                **train_config.dataloader_config,
                "training_data": training_data,
                "target_vars": ["Te_keV_psi", "ne20_psi", "ds_source_idx"],
                "batch_size": None,
            },
        }
    )

    _trainer, _train_dl, _val_dl, _test_dl, _ = launch_train(train_config)
