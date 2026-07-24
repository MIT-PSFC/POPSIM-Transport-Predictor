"""Tests for orchestration.study.configure_jax_platforms."""


def test_explicit_env_var_wins():
    """JAX_PLATFORMS already set in the environment is left untouched
    regardless of enable_parallelism."""


def test_parallelism_pins_cpu():
    """With enable_parallelism True and no JAX_PLATFORMS set, the env var is
    set to 'cpu' so the orchestrator can run on cpu nodes."""


def test_serial_leaves_env_unset():
    """With enable_parallelism False and no JAX_PLATFORMS set, the env var
    stays unset so jax's default backend selection picks gpu when present and
    falls back to cpu otherwise."""
