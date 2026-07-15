from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "profile_transfer"

    # Input-variable pairs to plot per normalization method. Only references
    # variables guaranteed present for profile-transfer datasets
    # (no P_aux_MW / surface_power_density, which are zero here).
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]] = {
        "raw": [
            ["Ip_MA", "Wtot_MJ"],
            ["R0", "a_minor"],
            ["ne20_line_avg", "B0"],
            ["betan", "kappa"],
        ],
        "physics": [
            ["Ip_MA", "beta"],
            ["q_star", "epsilon"],
            ["f_G", "aB0"],
            ["beta", "kappa"],
        ],
        "z_score": [
            ["Ip_MA_z", "Wtot_MJ_z"],
            ["R0_z", "a_minor_z"],
            ["ne20_line_avg_z", "B0_z"],
            ["Wtot_MJ_z", "kappa_z"],
        ],
        "coral": [
            ["Ip_MA_coral", "Wtot_MJ_coral"],
            ["R0_coral", "a_minor_coral"],
            ["ne20_line_avg_coral", "B0_coral"],
            ["Wtot_MJ_coral", "kappa_coral"],
        ],
    }
