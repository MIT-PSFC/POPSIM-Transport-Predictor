"""segmented_train_dataloader must reproduce the training dataloader popsim builds on the merged parts.

The reference is the path the time-dependent TRBs took before the parts were kept apart:
merge the raw parts with NaN padding, prepare the merged dataset, then one popsim make_dataloaders call.
Every case asserts the same samples in the same order with every coordinate and variable identical,
and one shuffled epoch of identical batches.
The slow test runs the same check on the real hs1 debug data of every time-dependent domain adaptation path.
"""

import resource
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from popsim.ml.dataloading import make_dataloaders

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import load_config, reset_config
from transport_study.modules.trb_utils import (
    ROOM_KEEPER_SHOT,
    prepared_for_dataloader,
    segmented_train_dataloader,
    time_dep_train_loader_kwargs,
)
from transport_study.orchestration.organize_data import (
    TrainingData,
    concat_with_nan_padding,
    get_train_test_datasets,
    get_train_val_datasets,
    get_transfer_pretrain_datasets,
)
from transport_study.orchestration.target_shots import configured_target_split

RHO = np.linspace(0.0, 1.0, 5)
DATALOADER_CONFIG = {
    "target_vars": ["n_e"],
    "state_vars": ["n_e"],
    "batch_size": 7,
    "segment_length_train": 100,
    "segment_overlap_train": 50,
}
INPUT_VARS = ["ip_MA", "ds_source_idx"]


def make_part(device_idx: int, lengths: list[int], rng, shot0: int, lead=None, gaps=None, tail_partial=None) -> xr.Dataset:
    """A raw training part as organize_data returns it: shots NaN-padded to the part's longest, a per-shot ds_source_idx.

    lead NaNs the first slices of a shot's profile, gaps an interior stretch,
    tail_partial the profile only over a shot's last slices (the scalars stay finite).
    """
    n_shots, n_times = len(lengths), max(lengths)
    times = np.full((n_shots, n_times), np.nan)
    ip_MA = np.full((n_shots, n_times), np.nan)
    n_e = np.full((n_shots, n_times, len(RHO)), np.nan)
    for i, length in enumerate(lengths):
        times[i, :length] = 1e-3 * np.arange(length)
        ip_MA[i, :length] = rng.normal(size=length)
        n_e[i, :length] = rng.normal(size=(length, len(RHO)))
    for i, n_lead in (lead or {}).items():
        n_e[i, :n_lead] = np.nan
    for i, (start, stop) in (gaps or {}).items():
        n_e[i, start:stop] = np.nan
    for i, n_tail in (tail_partial or {}).items():
        n_e[i, lengths[i] - n_tail : lengths[i]] = np.nan
    return xr.Dataset(
        {
            "ip_MA": ((EPISODE_DIM, TIME_DIM), ip_MA),
            "n_e": ((EPISODE_DIM, TIME_DIM, "rho"), n_e),
            "ds_source_idx": (EPISODE_DIM, np.full(n_shots, device_idx)),
        },
        coords={
            EPISODE_DIM: shot0 + np.arange(n_shots),
            TIME_DIM: np.arange(n_times),
            "rho": RHO,
            TIME_COORD: ((EPISODE_DIM, TIME_DIM), times),
        },
    ).assign_coords(ds_source=f"device{device_idx}")


def assert_identical_to_merged(raw_parts: list[xr.Dataset], dataloader_config: dict, input_vars: list[str]):
    """The segmented dataloader of the prepared parts against popsim on the merged then prepared parts."""
    loader_kwargs = time_dep_train_loader_kwargs(dataloader_config, input_vars)
    merged = prepared_for_dataloader(concat_with_nan_padding(raw_parts, concat_dim=EPISODE_DIM))
    (reference_dl,) = make_dataloaders(datasets=(merged,), **loader_kwargs)
    segmented_dl = segmented_train_dataloader([prepared_for_dataloader(part) for part in raw_parts], loader_kwargs)

    xr.testing.assert_identical(reference_dl.ds, segmented_dl.ds)
    assert ROOM_KEEPER_SHOT not in segmented_dl.ds[EPISODE_DIM].values
    batches_reference, batches_segmented = list(reference_dl), list(segmented_dl)
    assert len(batches_reference) == len(batches_segmented)
    for batch_reference, batch_segmented in zip(batches_reference, batches_segmented, strict=True):
        xr.testing.assert_identical(batch_reference.ds, batch_segmented.ds)
    return reference_dl


@pytest.mark.parametrize(
    "case",
    [
        "mixed_lengths",
        "longest_within_one_segment",
        "every_shot_shifted",
        "dead_shot_and_long_partial_tail",
        "shots_near_one_segment",
    ],
)
def test_segments_match_merged_dataloader(case):
    rng = np.random.default_rng(0)
    parts = {
        # MAST-like, C-Mod-like and DIII-D-like shot lengths, a single-shot part, leading NaN, a gap, a partial tail
        "mixed_lengths": [
            make_part(0, [289, 300, 150, 537], rng, shot0=100, lead={1: 30}),
            make_part(1, [1587, 1600, 1861], rng, shot0=200, gaps={0: (400, 460)}, tail_partial={2: 120}),
            make_part(2, [5520, 6582, 6000], rng, shot0=300, lead={0: 7}),
            make_part(0, [250], rng, shot0=400),
        ],
        # Each part's longest shot within one segment of the merged length
        "longest_within_one_segment": [
            make_part(0, [300, 299], rng, shot0=100),
            make_part(1, [310, 120], rng, shot0=200, lead={0: 3}),
        ],
        # No shot starts at index 0, and a gap close to the longest shot's end
        "every_shot_shifted": [
            make_part(0, [289, 300], rng, shot0=100, lead={0: 5, 1: 9}),
            make_part(1, [700, 640], rng, shot0=200, lead={0: 11, 1: 6}, gaps={0: (600, 690)}),
        ],
        # A shot with no valid slice, and a long stretch where only the scalars are finite
        "dead_shot_and_long_partial_tail": [
            make_part(0, [300, 280, 260], rng, shot0=100, lead={1: 280}),
            make_part(1, [900], rng, shot0=200, tail_partial={0: 250}),
        ],
        "shots_near_one_segment": [
            make_part(0, [120], rng, shot0=100),
            make_part(1, [101], rng, shot0=200),
            make_part(2, [99], rng, shot0=300),
        ],
    }[case]
    reference_dl = assert_identical_to_merged(parts, DATALOADER_CONFIG, INPUT_VARS)
    assert reference_dl.ds.sizes["sample"] > 0


def test_reserved_room_keeper_shot_is_refused():
    rng = np.random.default_rng(1)
    part = make_part(0, [300, 250], rng, shot0=ROOM_KEEPER_SHOT)
    with pytest.raises(ValueError, match="reserved"):
        segmented_train_dataloader([prepared_for_dataloader(part)], time_dep_train_loader_kwargs(DATALOADER_CONFIG, INPUT_VARS))


@pytest.mark.parametrize("domain_adaptation", ["addition", "weighted"])
def test_zero_target_shots_train_on_the_sources_alone(synthetic_device_stores, tmp_path, monkeypatch, domain_adaptation):
    """A zero-shot target contributes no training part, a zero-shot part fails the segmenting (hs1_pb_primary, 2026-10-06)."""
    from transport_study.config import config
    from transport_study.modules.power_balance.trb import PowerBalanceTRB
    from transport_study.power_balance_transfer.power_balance_study import (
        PowerBalanceStudy,
    )

    # Two target test shots, popsim rejects a single-shot whole-episode test set
    study = PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="test-zero-target-shots",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            target_test_set_size=2,
            training_datasets=("cmod",),
            model_types=("mlp",),
            data_normalization_methods=("physics",),
            domain_adaptation_methods=(domain_adaptation,),
            num_target_shots_options=(0,),
        )
    )
    # The synthetic shots are shorter than the study's training segments
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(
        study,
        "base_dataloader_config",
        lambda case: {**base_dataloader_config(case), "segment_length_train": 20, "segment_overlap_train": 10},
    )
    case = next(case for case in study.cases if case.domain_adaptation == domain_adaptation)
    train_config = study.make_train_config(case)
    _, train_dl, _, _ = PowerBalanceTRB.get_dataloaders(train_config.dataloader_config)

    train_device_idxs = np.unique(train_dl.ds["ds_source_idx"].values)
    assert train_device_idxs.tolist() == [config.ds_source_to_idx["cmod"]]


HS1_TOMLS = Path(__file__).parents[3] / "studies" / "hs1"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("study_rel", "case_name"),
    [
        ("power_balance/hs1_pb_primary.toml", "none"),
        ("power_balance/hs1_pb_primary.toml", "weighted"),
        ("power_balance/hs1_pb_primary.toml", "transfer_pretrain"),
        ("power_balance/hs1_pb_primary.toml", "exnihilo"),
        ("transport/hs1_transport_primary.toml", "none"),
        ("transport/hs1_transport_primary.toml", "weighted_spanning_133"),
    ],
)
def test_real_hs1_segments_match_merged_dataloader(study_rel, case_name):
    """The hs1 debug data of every time-dependent domain adaptation path, against the merged reference.

    exnihilo (like transfer) trains on the target part alone,
    its reference merges every source with the target and strips the sources again, as the merged path did.
    Prints the wall times and the process peak RSS after each path (cumulative, the segmented path runs first), -s shows them.
    """
    from transport_study.power_balance_transfer.power_balance_study import (
        PowerBalanceStudy,
    )
    from transport_study.transport_transfer.transport_transfer_study import (
        TRANSPORT_INPUT_VARS,
        TransportStudy,
    )

    study_cls = PowerBalanceStudy if study_rel.startswith("power_balance") else TransportStudy
    reset_config()
    cfg = load_config(study_cls.Config.from_toml(HS1_TOMLS / study_rel))
    sources = TrainingData(sources=sorted(set(cfg.dataset_paths) - {cfg.target_device}))
    study_type = study_cls.STUDY_TYPE
    if case_name == "none":
        raw_parts, _ = get_train_val_datasets(sources, study_type=study_type)
    elif case_name == "transfer_pretrain":
        raw_parts, _, _ = get_transfer_pretrain_datasets(sources, configured_target_split(10, "ascending"), study_type=study_type)
    elif case_name == "exnihilo":
        exnihilo = TrainingData(sources=sources.sources, exnihilo=True)
        raw_parts, _ = get_train_test_datasets(exnihilo, None, configured_target_split(30, "ascending"), study_type=study_type)
        all_parts, _ = get_train_test_datasets(sources, "addition", configured_target_split(30, "ascending"), study_type=study_type)
    else:
        n_target = 133 if case_name.endswith("133") else 10
        order = "spanning" if "spanning" in case_name else "ascending"
        raw_parts, _ = get_train_test_datasets(sources, "weighted", configured_target_split(n_target, order), study_type=study_type)

    if study_cls is PowerBalanceStudy:
        dataloader_config = {
            "target_vars": ["energy_mhd_MJ", "power_ohm_MW", "power_radiated_MW", "ds_source_idx"],
            "state_vars": ["energy_mhd_MJ"],
            "extra_vars": None,
        }
        input_vars = ["ip_MA", "b_geo", "geometric_axis_r", "minor_radius", "elongation", "n_e_line_average_1e20", "power_additional_MW"]
    else:
        from transport_study.transport_transfer.transport_transfer_study import (
            TRANSPORT_STATE_VARS,
            TRANSPORT_TARGET_VARS,
        )

        dataloader_config = {
            "target_vars": TRANSPORT_TARGET_VARS,
            "state_vars": TRANSPORT_STATE_VARS,
            "extra_vars": ["power_ohm_MW", "power_radiated_MW"],
        }
        input_vars = list(TRANSPORT_INPUT_VARS)
    dataloader_config |= {"batch_size": 512, "segment_length_train": 100, "segment_overlap_train": 50}
    input_vars = [*input_vars, "ds_source_idx"]
    loader_kwargs = time_dep_train_loader_kwargs(dataloader_config, input_vars)

    start = time.time()
    segmented_dl = segmented_train_dataloader([prepared_for_dataloader(part) for part in raw_parts], loader_kwargs)
    segmented_s = time.time() - start
    segmented_rss_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    start = time.time()
    if case_name == "exnihilo":
        merged = concat_with_nan_padding(all_parts, concat_dim=EPISODE_DIM)
        merged = merged.where(merged["ds_source_idx"] == cfg.ds_source_to_idx[cfg.target_device], drop=True)
    else:
        merged = concat_with_nan_padding(raw_parts, concat_dim=EPISODE_DIM)
    (reference_dl,) = make_dataloaders(datasets=(prepared_for_dataloader(merged),), **loader_kwargs)
    reference_s = time.time() - start
    reference_rss_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(
        f"{study_rel} {case_name}: {segmented_dl.ds.sizes['sample']} samples, "
        f"segmented {segmented_s:.0f} s with peak RSS {segmented_rss_GB:.1f} GB, "
        f"merged {reference_s:.0f} s with peak RSS {reference_rss_GB:.1f} GB"
    )
    samples_reference, samples_segmented = reference_dl.ds, segmented_dl.ds
    first_reference, first_segmented = next(iter(reference_dl)).ds, next(iter(segmented_dl)).ds
    if case_name == "exnihilo":
        # Stripped out of the merged sources the ds_source label stays per sample, the target part alone carries it once.
        # Both name the target on every sample, every other variable and coordinate must still be identical
        for samples in (samples_reference, samples_segmented, first_reference, first_segmented):
            assert (samples["ds_source"] == cfg.target_device).all()
        samples_reference, samples_segmented = samples_reference.drop_vars("ds_source"), samples_segmented.drop_vars("ds_source")
        first_reference, first_segmented = first_reference.drop_vars("ds_source"), first_segmented.drop_vars("ds_source")
    xr.testing.assert_identical(samples_reference, samples_segmented)
    xr.testing.assert_identical(first_reference, first_segmented)
