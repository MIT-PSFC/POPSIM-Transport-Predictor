from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "power_balance_transfer"

    # Input-variable pairs to plot per normalization method.
    # Power-balance datasets always carry power_additional_MW (signals.convert_to_working_units sums the heating powers),
    # so the aux-power derived variables are safe to plot.
    # Coral and physics-coral pairs use only the module's joint feature set
    # (adding energy_mhd_MJ or beta to the fit would change every var's transform).
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]] = {
        "raw": [
            ["ip_MA", "energy_mhd_MJ"],
            ["geometric_axis_r", "minor_radius"],
            ["n_e_line_average_1e20", "b_geo"],
            ["power_additional_MW", "elongation"],
        ],
        "physics": [
            ["ip_MA", "beta"],
            ["q_star", "epsilon"],
            ["f_G", "aB0"],
            ["surface_power_density", "elongation"],
        ],
        "zscore": [
            ["ip_MA_z", "energy_mhd_MJ_z"],
            ["geometric_axis_r_z", "minor_radius_z"],
            ["n_e_line_average_1e20_z", "b_geo_z"],
            ["power_additional_MW_z", "elongation_z"],
        ],
        "coral": [
            ["ip_MA_coral", "elongation_coral"],
            ["geometric_axis_r_coral", "minor_radius_coral"],
            ["n_e_line_average_1e20_coral", "b_geo_coral"],
            ["power_additional_MW_coral", "ip_MA_coral"],
        ],
        "physics-coral": [
            ["ip_MA_pcoral", "elongation_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["surface_power_density_pcoral", "elongation_pcoral"],
        ],
        "physics-zscore": [
            ["ip_MA_pz", "elongation_pz"],
            ["q_star_pz", "epsilon_pz"],
            ["f_G_pz", "aB0_pz"],
            ["surface_power_density_pz", "elongation_pz"],
        ],
    }
