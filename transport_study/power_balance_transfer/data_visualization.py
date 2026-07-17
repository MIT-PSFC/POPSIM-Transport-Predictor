from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "power_balance_transfer"

    # Input-variable pairs to plot per normalization method. Power-balance
    # datasets always carry P_aux_MW (get_ds sums the per-system aux power
    # signals), so the aux-power derived variables are safe to plot.
    # Coral and physics-coral pairs use only the module's joint feature set
    # (adding Wtot_MJ or beta to the fit would change every var's transform).
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]] = {
        "raw": [
            ["Ip_MA", "Wtot_MJ"],
            ["R0", "a_minor"],
            ["ne20_line_avg", "B0"],
            ["P_aux_MW", "kappa"],
        ],
        "physics": [
            ["Ip_MA", "beta"],
            ["q_star", "epsilon"],
            ["f_G", "aB0"],
            ["surface_power_density", "kappa"],
        ],
        "z_score": [
            ["Ip_MA_z", "Wtot_MJ_z"],
            ["R0_z", "a_minor_z"],
            ["ne20_line_avg_z", "B0_z"],
            ["P_aux_MW_z", "kappa_z"],
        ],
        "coral": [
            ["Ip_MA_coral", "kappa_coral"],
            ["R0_coral", "a_minor_coral"],
            ["ne20_line_avg_coral", "B0_coral"],
            ["P_aux_MW_coral", "Ip_MA_coral"],
        ],
        "physics-coral": [
            ["Ip_MA_pcoral", "kappa_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["surface_power_density_pcoral", "kappa_pcoral"],
        ],
    }
