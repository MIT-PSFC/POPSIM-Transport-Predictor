from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "profile_transfer"

    # Input-variable pairs to plot per normalization method. Only references
    # variables guaranteed present for profile-transfer datasets
    # (no P_aux_MW / surface_power_density, which are zero here).
    # Coral and physics-coral pairs use only the module's joint feature set
    # (adding Wtot_MJ or beta to the fit would change every var's transform).
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
        "zscore": [
            ["Ip_MA_z", "Wtot_MJ_z"],
            ["R0_z", "a_minor_z"],
            ["ne20_line_avg_z", "B0_z"],
            ["Wtot_MJ_z", "kappa_z"],
        ],
        "coral": [
            ["Ip_MA_coral", "kappa_coral"],
            ["R0_coral", "a_minor_coral"],
            ["ne20_line_avg_coral", "B0_coral"],
            ["Ip_MA_coral", "B0_coral"],
        ],
        "physics-coral": [
            ["Ip_MA_pcoral", "kappa_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["q_star_pcoral", "aB0_pcoral"],
        ],
        "physics-zscore": [
            ["Ip_MA_pz", "kappa_pz"],
            ["q_star_pz", "epsilon_pz"],
            ["f_G_pz", "aB0_pz"],
            ["q_star_pz", "aB0_pz"],
        ],
    }
