"""Validate every study's wandb sweep configs against the parts of the sweep
schema wandb enforces server-side, so a malformed file fails fast here instead
of failing wandb.sweep at study runtime.
"""

from pathlib import Path

import pytest
from popsim.ml.train_config import load_dict

from transport_study.config import load_config
from transport_study.modules.trb_utils import OPTIMIZER_CONFIG_KEYS
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.transport_transfer.transport_transfer_study import TransportStudy

STUDIES = (ProfileStudy, PowerBalanceStudy, TransportStudy)

# The orchestration ranks sweep runs by the logged history of this exact key
SWEEP_METRIC_NAME = "val/loss.mean"

# A distribution spec must carry its bounds alongside the distribution name
DISTRIBUTION_KEYS = {"distribution", "min", "max"}

# The optimizer_config keys a trial's optimizer reads.
# popsim turns transition_frac into transition_steps before the optimizer is built (launch.resolve_transition_frac)
SWEEPABLE_OPTIMIZER_KEYS = {*OPTIMIZER_CONFIG_KEYS, "transition_frac"}

SWEEP_CONFIG_PATHS = [(study, path) for study in STUDIES for path in sorted(Path(study.SWEEP_CONFIG_DIR).glob("*.yaml"))]


def load_study_config(study, **overrides):
    """A loaded config of the study over placeholder stores, the pseudo-type routing reads its variant fields."""
    defaults = dict(
        study_name="test_sweep_configs",
        dataset_paths={"cmod-low": "path/to/cmod_low.nc", "cmod-high": "path/to/cmod_high.nc"},
        target_device="cmod-high",
        training_datasets=("cmod-low",),
        target_test_set_size=1,
    )
    return load_config(study.Config(**{**defaults, **overrides}))


def check_parameter(name: str, spec: dict, path: Path):
    """Recursively assert one parameter spec is exactly one of the legal shapes."""
    if "parameters" in spec:
        for sub_name, sub_spec in spec["parameters"].items():
            check_parameter(f"{name}.{sub_name}", sub_spec, path)
        return
    if "values" in spec:
        assert isinstance(spec["values"], list), f"{path.name}: {name} 'values' must be a list, got {spec['values']!r}"
        assert spec["values"], f"{path.name}: {name} 'values' must be non-empty"
    elif "distribution" in spec:
        assert set(spec) >= DISTRIBUTION_KEYS, f"{path.name}: {name} distribution spec missing min/max: {spec!r}"
    elif "value" not in spec:
        pytest.fail(f"{path.name}: {name} has no value/values/distribution: {spec!r}")


@pytest.mark.parametrize("study", STUDIES, ids=lambda s: s.STUDY_TYPE)
def test_every_model_type_has_a_sweep_config(study):
    """Every model type a case can take must resolve to a sweep config to tune with."""
    load_study_config(study)
    for model_type in study.Case.VALID_MODEL_TYPES:
        assert study.sweep_config(model_type)["parameters"], model_type


@pytest.mark.parametrize("study", STUDIES, ids=lambda s: s.STUDY_TYPE)
def test_no_sweep_config_for_unknown_model_type(study):
    """A stray yaml whose stem is not a model type would never be loaded."""
    present = {path.stem for path in Path(study.SWEEP_CONFIG_DIR).glob("*.yaml")}
    unknown = present - set(study.Case.VALID_MODEL_TYPES)
    assert not unknown, f"{study.__name__}: sweep configs for unknown model types {sorted(unknown)}"


def test_transport_pseudo_types_sweep_with_the_owning_studys_yaml():
    """A sciml prereq tunes exactly as the owning study's case of the configured variant would."""
    load_study_config(TransportStudy, power_balance_model_type="sciml-taue-scalinglaw", profile_model_type="mlp")
    assert TransportStudy.sweep_config("power_balance") == PowerBalanceStudy.sweep_config("sciml-taue-scalinglaw")
    assert TransportStudy.sweep_config("p_oh") == PowerBalanceStudy.sweep_config("p_oh")
    assert TransportStudy.sweep_config("profile") == ProfileStudy.sweep_config("mlp")
    transport_transformer = load_dict(str(Path(TransportStudy.SWEEP_CONFIG_DIR) / "transformer.yaml"))
    assert TransportStudy.sweep_config("transformer") == transport_transformer


@pytest.mark.parametrize(("study", "path"), SWEEP_CONFIG_PATHS, ids=[f"{s.STUDY_TYPE}-{p.stem}" for s, p in SWEEP_CONFIG_PATHS])
def test_sweep_config_is_valid(study, path):
    # load_dict is what launch_sweep uses, so a file it cannot parse fails here
    cfg = load_dict(str(path))

    assert cfg["method"] in ("bayes", "grid", "random")
    assert cfg["metric"]["name"] == SWEEP_METRIC_NAME
    assert cfg["metric"]["goal"] == "minimize"

    assert cfg.get("parameters"), f"{path.name}: no parameters"
    # The loader config is never swept, the tuned merge takes none of it and the transfer lr0 is measured on the base loaders
    assert "dataloader_config" not in cfg["parameters"], f"{path.name}: sweeps must not change the training loader"
    for name, spec in cfg["parameters"].items():
        check_parameter(name, spec, path)


@pytest.mark.parametrize(("study", "path"), SWEEP_CONFIG_PATHS, ids=[f"{s.STUDY_TYPE}-{p.stem}" for s, p in SWEEP_CONFIG_PATHS])
def test_swept_optimizer_keys_reach_the_optimizer(study, path):
    """wandb merges any swept key into the trial config, so a key the optimizer never reads would be silently ignored."""
    cfg = load_dict(str(path))
    optimizer_parameters = cfg["parameters"].get("optimizer_config", {}).get("parameters", {})
    unread_keys = set(optimizer_parameters) - SWEEPABLE_OPTIMIZER_KEYS
    assert not unread_keys, f"{path.name}: the optimizer never reads {sorted(unread_keys)}"
