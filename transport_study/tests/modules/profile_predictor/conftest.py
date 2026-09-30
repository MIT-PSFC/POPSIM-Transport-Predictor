"""Shared builders for the TORAX-backed profile predictor tests."""

import jax
import pytest
import xarray as xr

from transport_study.config import RHO_GRID
from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import N_NN_INPUTS
from transport_study.modules.profile_predictor.torax_module import ProfilePredictorTorax
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.profile_transfer.plot_torax_evolution import valid_timesteps
from transport_study.tests.sample_data import SAMPLE_DIR


@pytest.fixture
def sample_timeslices():
    """Load timeslices with a valid TORAX input state from a sample file.

    Tests using this fixture must carry requires_sample_data.

    The first and last valid slice of each shot are taken, so a small request
    still spans the shot's parameter range. Raw sample files predate the device
    index organize_data adds, which the modules read to pick a normalizer row.
    """

    def _load(sample_name: str, n_slices: int = 1) -> list[xr.Dataset]:
        ds = xr.open_dataset(SAMPLE_DIR / sample_name)
        slices: list[xr.Dataset] = []
        for shot in ds["shot"].values:
            shot_ds = ds.sel(shot=shot)
            valid = valid_timesteps(shot_ds)
            if len(valid) == 0:
                continue
            for time_idx in sorted({int(valid[0]), int(valid[-1])}):
                timeslice = shot_ds.isel(time_idx=time_idx)
                timeslice["ds_source_idx"] = 0.0
                slices.append(timeslice)
                if len(slices) >= n_slices:
                    return slices
        if not slices:
            raise ValueError(f"No valid timeslice in {sample_name}")
        return slices

    return _load


@pytest.fixture
def make_torax_module():
    """Build a ProfilePredictorTorax at the production config for a transport model."""

    def _make(
        transport_model: str,
        geometry_builder: str = "circular",
        numerics_overrides: dict | None = None,
    ) -> ProfilePredictorTorax:
        model_cfg = PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model]["model_init_config"]
        torax_config = model_cfg["torax_config"]
        if numerics_overrides:
            torax_config = {**torax_config, "numerics": {**torax_config["numerics"], **numerics_overrides}}
        return ProfilePredictorTorax(
            nn_width=model_cfg["nn_width"],
            nn_depth=model_cfg["nn_depth"],
            rhogrid=tuple(RHO_GRID.tolist()),
            torax_config=torax_config,
            key=jax.random.PRNGKey(42),
            normalizer=CoralFeatureNormalizer.identity(1, N_NN_INPUTS),
            transport_model=transport_model,
            geometry_builder=geometry_builder,
        )

    return _make
