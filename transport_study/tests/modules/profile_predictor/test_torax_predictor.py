import numpy as np
import pytest
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, load_config
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.modules.profile_predictor.trb import resolve_relaxation_overrides
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS


@pytest.mark.slow
@pytest.mark.parametrize("transport_model", ["constant", "cgm", "gyrobohm", "qlknn"])
def test_torax_predictor(transport_model):
    config = StudyConfig(
        study_name=f"test_torax_predictor_{transport_model}",
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod-low1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod-high.nc",
        },
        target_device="cmod-high",
        # large batch with the the full ~100 shot sample dataset OOMs the GPU
        # 10 shots keeps this a cheap smoke test
        max_ds_size=10,
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
                "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                "batch_size": 512,
            },
            "model_init_config": {
                **train_config.model_init_config,
                # model_init indexes this strictly, the study normally supplies it
                "data_normalization": "physics-coral",
            },
        }
    )

    _trainer, _train_dl, _val_dl, _test_dl, _ = launch_train(train_config)


# cgm and qlknn are the models whose training blew up on MAST samples with
# the circular geometry, so they are the smoke coverage for the miller builder
@pytest.mark.slow
@pytest.mark.parametrize("transport_model", ["cgm", "qlknn"])
def test_torax_predictor_mast_miller(transport_model):
    config = StudyConfig(
        study_name=f"test_torax_predictor_mast_miller_{transport_model}",
        dataset_paths={
            "mast-low": PACKAGE_ROOT / "datasets" / "sample" / "mast-low1.nc",
            "mast-high": PACKAGE_ROOT / "datasets" / "sample" / "mast-high.nc",
        },
        target_device="mast-high",
        max_ds_size=10,
    )
    load_config(config)

    train_config = TrainConfig(**PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model])
    training_data = {
        "sources_unsorted": ["mast-low"],
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
                "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                "batch_size": 512,
            },
            "model_init_config": {
                **train_config.model_init_config,
                "geometry_builder": "miller",
                # model_init indexes this strictly, the study normally supplies it
                "data_normalization": "physics-coral",
            },
        }
    )

    _trainer, _train_dl, _val_dl, _test_dl, _ = launch_train(train_config)


@pytest.mark.slow
def test_torax_heat_source_response(make_torax_module, sample_timeslices):
    # Pins the generic_heat wiring end to end: prescribing more auxiliary
    # power through the NN-controlled source must heat the relaxed profile
    module = make_torax_module("cgm")
    timeslice = sample_timeslices("cmod-high.nc")[0]

    steps_cold, coeffs_cold = module.evolve(timeslice, prescribed={"P_aux_total": 0.0})
    steps_hot, coeffs_hot = module.evolve(timeslice, prescribed={"P_aux_total": 10.0})

    assert coeffs_cold["P_aux_total"] == pytest.approx(0.0)
    assert coeffs_hot["P_aux_total"] == pytest.approx(10.0)
    for steps in (steps_cold, steps_hot):
        for step in steps:
            assert np.all(np.isfinite(step["ne20"]))
            assert np.all(np.isfinite(step["te_keV"]))

    te_cold = steps_cold[-1]["te_keV"].mean()
    te_hot = steps_hot[-1]["te_keV"].mean()
    assert te_hot > te_cold * 1.05


@pytest.mark.slow
def test_torax_output_hits_edge_bc_and_smooth_init(make_torax_module, sample_timeslices):
    module = make_torax_module("cgm")
    timeslice = sample_timeslices("cmod-high.nc")[0]

    # The 51-point output must pass through the exact Dirichlet edge BC at
    # rho = 1 instead of flat-holding the outermost cell value
    outputs = module(timeslice)
    steps, coeffs = module.evolve(timeslice)
    assert outputs.te.values[-1] == pytest.approx(coeffs["T_e_right_bc"], rel=1e-3)
    assert outputs.ne.values[-1] == pytest.approx(coeffs["n_e_right_bc"], rel=1e-3)

    # The initial condition sampled on the cell grid must be a smooth parabola
    for key in ("te_keV", "ne20"):
        init = steps[0][key]
        d2 = np.diff(init, n=2)
        scale = np.abs(init).max()
        assert np.all(np.abs(d2 - d2.mean()) < 1e-3 * scale), key


def test_resolve_relaxation_overrides():
    # Nothing set: no overrides, torax_config numerics stay authoritative
    assert resolve_relaxation_overrides({}) == {}
    assert resolve_relaxation_overrides({"t_final": None, "fixed_dt": None, "n_solver_steps": None}) == {}

    # Explicit t_final / fixed_dt pass through unchanged
    assert resolve_relaxation_overrides({"t_final": 0.2}) == {"t_final": 0.2}
    assert resolve_relaxation_overrides({"t_final": 0.2, "fixed_dt": 0.02}) == {"t_final": 0.2, "fixed_dt": 0.02}

    # n_solver_steps derives fixed_dt = t_final / n_solver_steps, so the
    # swept horizon does not multiply per-sample solver cost
    assert resolve_relaxation_overrides({"t_final": 0.4, "n_solver_steps": 10}) == {
        "t_final": 0.4,
        "fixed_dt": pytest.approx(0.04),
    }
    assert resolve_relaxation_overrides({"t_final": 0.1, "n_solver_steps": 5, "fixed_dt": None}) == {
        "t_final": 0.1,
        "fixed_dt": pytest.approx(0.02),
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_relaxation_overrides({"t_final": 0.2, "fixed_dt": 0.02, "n_solver_steps": 5})
    with pytest.raises(ValueError, match="requires t_final"):
        resolve_relaxation_overrides({"n_solver_steps": 5})


def test_torax_max_steps_from_n_solver_steps(make_torax_module):
    # The module derives max_steps = ceil(t_final / fixed_dt) + 1, so an
    # n_solver_steps override must bound the scan length to n_solver_steps + 1
    module = make_torax_module("cgm", numerics_overrides=resolve_relaxation_overrides({"t_final": 0.4, "n_solver_steps": 10}))
    assert module.max_steps == 11
