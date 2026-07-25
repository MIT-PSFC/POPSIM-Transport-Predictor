"""Validate every study's wandb sweep configs against the parts of the sweep
schema wandb enforces server-side, so a malformed file fails fast here instead
of failing wandb.sweep at study runtime.
"""

from pathlib import Path

import pytest
from popsim.ml.train_config import load_dict

from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.transport_transfer.transport_transfer_study import TransportStudy

STUDIES = (ProfileStudy, PowerBalanceStudy, TransportStudy)

# The orchestration picks the best sweep run by this exact summary key
SWEEP_METRIC_NAME = "val/loss.mean"

# A distribution spec must carry its bounds alongside the distribution name
DISTRIBUTION_KEYS = {"distribution", "min", "max"}

SWEEP_CONFIG_PATHS = [(study, path) for study in STUDIES for path in sorted(Path(study.SWEEP_CONFIG_DIR).glob("*.yaml"))]


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
def test_sweep_config_exists_for_every_model_type(study):
    """Every model type a case can take must have a sweep config to tune with."""
    present = {path.stem for path in Path(study.SWEEP_CONFIG_DIR).glob("*.yaml")}
    missing = set(study.Case.VALID_MODEL_TYPES) - present
    assert not missing, f"{study.__name__}: missing sweep configs for {sorted(missing)}"


@pytest.mark.parametrize("study", STUDIES, ids=lambda s: s.STUDY_TYPE)
def test_no_sweep_config_for_unknown_model_type(study):
    """A stray yaml whose stem is not a model type would never be loaded."""
    present = {path.stem for path in Path(study.SWEEP_CONFIG_DIR).glob("*.yaml")}
    unknown = present - set(study.Case.VALID_MODEL_TYPES)
    assert not unknown, f"{study.__name__}: sweep configs for unknown model types {sorted(unknown)}"


@pytest.mark.parametrize(("study", "path"), SWEEP_CONFIG_PATHS, ids=[f"{s.STUDY_TYPE}-{p.stem}" for s, p in SWEEP_CONFIG_PATHS])
def test_sweep_config_is_valid(study, path):
    # load_dict is what launch_sweep uses, so a file it cannot parse fails here
    cfg = load_dict(str(path))

    assert cfg["method"] in ("bayes", "grid", "random")
    assert cfg["metric"]["name"] == SWEEP_METRIC_NAME
    assert cfg["metric"]["goal"] == "minimize"

    assert cfg.get("parameters"), f"{path.name}: no parameters"
    for name, spec in cfg["parameters"].items():
        check_parameter(name, spec, path)
