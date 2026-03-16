from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    resources_available,
)


def test_count_idle_gpus():
    idle_gpus = count_idle_gpus()
    assert isinstance(idle_gpus, int)
    assert idle_gpus >= 0


def test_resources_available():
    available = resources_available()
    assert isinstance(available, bool)


if __name__ == "__main__":
    test_count_idle_gpus()
    test_resources_available()
