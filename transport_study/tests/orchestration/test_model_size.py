"""model_size_updates on the real sweep grids: the architecture each model size trains."""

from transport_study.orchestration.study import model_size_updates
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.profile_transfer.profile_study import ProfileStudy

TUNED_TRANSFORMER = {"d_model": 32, "num_heads": 4, "history_len": 20, "nn_width": 24, "nn_depth": 3}


def test_min_and_max_move_only_width_and_depth_to_the_grid_ends():
    parameters = PowerBalanceStudy.sweep_model_init_parameters("transformer")
    for model_size, grid_end in (("min", min), ("max", max)):
        updates = model_size_updates(model_size, parameters, TUNED_TRANSFORMER)
        # history_len and num_heads stay tuned
        assert set(updates) == {"d_model", "nn_width", "nn_depth"}
        for name, value in updates.items():
            assert value == grid_end(parameters[name]["values"])


def test_half_halves_the_tuned_widths_and_keeps_depth():
    parameters = PowerBalanceStudy.sweep_model_init_parameters("transformer")
    assert model_size_updates("half", parameters, TUNED_TRANSFORMER) == {"d_model": 16, "nn_width": 12}
    # d_model 4 would halve to 2, below one dimension per head for 4 heads, so it stays 4
    tuned_smallest_d_model = {**TUNED_TRANSFORMER, "d_model": 4}
    assert model_size_updates("half", parameters, tuned_smallest_d_model)["d_model"] == 4


def test_half_reservoir_halves_only_its_size():
    parameters = ProfileStudy.sweep_model_init_parameters("reservoir")
    tuned_reservoir = {"reservoir_size": 1920, "n_steps": 40, "spectral_radius": 0.9, "input_scaling": 0.5, "leak_rate": 0.6}
    assert model_size_updates("half", parameters, tuned_reservoir) == {"reservoir_size": 960}


def test_sweep_keeps_the_tuned_config():
    parameters = PowerBalanceStudy.sweep_model_init_parameters("transformer")
    assert model_size_updates("sweep", parameters, TUNED_TRANSFORMER) == {}


def test_a_model_type_without_its_own_swept_net_overrides_nothing():
    # The scaling law sweeps its optimizer alone, its sized nets are its p_oh / p_rad prereqs
    parameters = PowerBalanceStudy.sweep_model_init_parameters("sciml-taue-scalinglaw")
    assert parameters == {}
    assert model_size_updates("max", parameters, {}) == {}
