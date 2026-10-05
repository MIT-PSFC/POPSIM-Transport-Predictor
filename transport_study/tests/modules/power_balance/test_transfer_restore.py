"""A power balance transfer case keeps the p_oh / p_rad submodules of its own prereq cases (PowerBalanceTRB.model_init).

The finetune restores the whole env from its transfer_pretrain checkpoint,
which carries the pretrain's submodules, then puts back the submodules its p_oh / p_rad prereqs trained.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from popsim.ml.checkpointing import (
    TrainState,
    create_default_checkpoint_manager,
    save_train_state,
)

from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.modules.trb_utils import submodule_config_dict
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

SUBMODULE_TRBS = {"p_oh_predictor": OhmicPowerTRB, "p_rad_predictor": RadiatedPowerTRB}


def _shifted(model, offset: float):
    """model with every float leaf shifted by offset, standing in for trained weights."""
    return jax.tree.map(lambda leaf: leaf + offset if eqx.is_inexact_array(leaf) else leaf, model)


def _save(model, checkpoint_dir) -> None:
    manager = create_default_checkpoint_manager(checkpoint_dir)
    save_train_state(TrainState(step=1, epoch=1, model=model, opt_state={"dummy": jnp.zeros(1)}), manager, loss=0.0)


def _float_leaves(model) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))]


def test_transfer_restores_the_pretrain_then_its_own_submodules(synthetic_device_stores, tmp_path, monkeypatch):
    # Two target test shots, popsim rejects a single-shot whole-episode test set
    study = PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="test-transfer-restore",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            target_test_set_size=2,
            training_datasets=("cmod",),
            model_types=("sciml-taue-nn",),
            data_normalization_methods=("physics",),
            domain_adaptation_methods=("none", "transfer"),
            num_target_shots_options=(1,),
        )
    )
    # The synthetic shots are shorter than the study's training segments
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(
        study,
        "base_dataloader_config",
        lambda case: {**base_dataloader_config(case), "segment_length_train": 20, "segment_overlap_train": 10},
    )
    case = next(case for case in study.cases if case.domain_adaptation == "transfer" and case.model_type == "sciml-taue-nn")
    train_config = study.make_train_config(case)
    _, train_dl, _, _ = PowerBalanceTRB.get_dataloaders(train_config.dataloader_config)

    # The prereq submodules as trained, their own transfer restore is not under test here
    submodule_configs, trained_submodules = {}, {}
    for offset, (name, submodule_trb) in enumerate(SUBMODULE_TRBS.items(), start=1):
        submodule_config = submodule_config_dict(train_config.model_init_config["submodules"][name])
        submodule_config["model_init_config"] = {**submodule_config["model_init_config"], "transfer_checkpoint": None}
        submodule_config["checkpoint_dir"] = str(tmp_path / name)
        trained_submodules[name] = _shifted(submodule_trb.model_init(train_dl, submodule_config["model_init_config"]), float(offset))
        _save(trained_submodules[name], submodule_config["checkpoint_dir"])
        submodule_configs[name] = submodule_config
    model_init_config = {**train_config.model_init_config, "submodules": submodule_configs}

    # A pretrain whose submodules differ from the prereq ones
    pretrain_env = PowerBalanceTRB.model_init(train_dl, {**model_init_config, "transfer_checkpoint": None})
    pretrained_env = _shifted(pretrain_env, 100.0)
    _save(pretrained_env, tmp_path / "pretrain")

    finetune_env = PowerBalanceTRB.model_init(train_dl, {**model_init_config, "transfer_checkpoint": str(tmp_path / "pretrain")})

    for name, trained_submodule in trained_submodules.items():
        for restored, trained in zip(_float_leaves(getattr(finetune_env.module, name)), _float_leaves(trained_submodule), strict=True):
            np.testing.assert_array_equal(restored, trained)
    for restored, pretrained in zip(
        _float_leaves(finetune_env.module.taue_predictor), _float_leaves(pretrained_env.module.taue_predictor), strict=True
    ):
        np.testing.assert_array_equal(restored, pretrained)
