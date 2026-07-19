from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "transport_transfer"

    # Input-variable pairs to plot per normalization method, restricted to the
    # methods this study actually uses (physics features, optionally CORAL
    # aligned or z-scored). Transport datasets always carry P_aux_MW (get_ds sums the
    # per-system aux power signals) and the triangularities, so those are safe
    # to plot. Coral pairs use only the module's joint feature set (adding
    # Wtot_MJ or beta to the fit would change every var's transform).
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
        "physics-coral": [
            ["Ip_MA_pcoral", "kappa_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["surface_power_density_pcoral", "kappa_pcoral"],
        ],
        "physics-zscore": [
            ["Ip_MA_pz", "kappa_pz"],
            ["q_star_pz", "epsilon_pz"],
            ["f_G_pz", "aB0_pz"],
            ["surface_power_density_pz", "kappa_pz"],
        ],
    }
