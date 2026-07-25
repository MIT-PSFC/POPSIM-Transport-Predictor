from typing import ClassVar

from transport_study.orchestration.data_visualization import DataVisualizationBase


class DataVisualization(DataVisualizationBase):
    STUDY_TYPE: ClassVar[str] = "profile_transfer"
    # The profile models normalize their own 10 dimensionless nn_inputs, not the
    # power balance inputs, so the physics* plots show that feature space
    FEATURE_SPACE: ClassVar[str] = "profile"

    # Input-variable pairs to plot per normalization method. Only the methods
    # this study can actually run (VALID_DATA_NORMALIZATIONS) plus the raw
    # physical inputs for reference.
    # The physics* pairs are nn_inputs slots, so they are exactly the features
    # the stat stage aligns (adding anything outside the joint feature set to a
    # CORAL fit would change every var's transform).
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]] = {
        "raw": [
            ["Ip_MA", "Wtot_MJ"],
            ["R0", "a_minor"],
            ["ne20_line_avg", "B0"],
            ["betan", "kappa"],
        ],
        "physics": [
            ["beta", "betan"],
            ["q_star", "epsilon"],
            ["f_G", "aB0"],
            ["log_nu_star", "kappa"],
        ],
        "physics-coral": [
            ["beta_pcoral", "betan_pcoral"],
            ["q_star_pcoral", "epsilon_pcoral"],
            ["f_G_pcoral", "aB0_pcoral"],
            ["log_nu_star_pcoral", "kappa_pcoral"],
        ],
        "physics-zscore": [
            ["beta_pz", "betan_pz"],
            ["q_star_pz", "epsilon_pz"],
            ["f_G_pz", "aB0_pz"],
            ["log_nu_star_pz", "kappa_pz"],
        ],
    }
