from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "transport_transfer"
    # The transport models normalize their own 11 transport features, not the
    # power balance inputs, so the physics* plots show that feature space
    FEATURE_SPACE: ClassVar[str] = "transport"

    # Input-variable pairs to plot per normalization method, restricted to the
    # methods this study actually uses (physics features, optionally CORAL
    # aligned or z-scored).
    # The physics* pairs are transport_nn_inputs slots, so they are exactly the
    # features the stat stage aligns (adding anything outside the joint feature
    # set to a CORAL fit would change every var's transform). The beta-derived
    # slots come from the measured Wtot here, at runtime the modules use the
    # state-implied one.
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]] = {
        "raw": [
            ["Ip_MA", "Wtot_MJ"],
            ["R0", "a_minor"],
            ["ne20_line_avg", "B0"],
            ["P_aux_MW", "kappa"],
        ],
        "physics": [
            ["beta", "betan"],
            ["q_star", "epsilon"],
            ["f_G", "aB0"],
            ["paux_norm", "kappa"],
        ],
        "physics-coral": [
            ["beta_pcoral", "betan_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["paux_norm_pcoral", "kappa_pcoral"],
        ],
        "physics-zscore": [
            ["beta_pz", "betan_pz"],
            ["q_star_pz", "epsilon_pz"],
            ["f_G_pz", "aB0_pz"],
            ["paux_norm_pz", "kappa_pz"],
        ],
    }
