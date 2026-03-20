from dataclasses import dataclass

import netCDF4  # noqa: F401

from transport_study.orchestration.study import Study


class PowerBalanceStudy(Study):
    HYPERPARAM_TRAINING_DATA = "cmod_tcv"
    HYPERPARAM_DATA_NORMALIZATION = "raw"
    HYPERPARAM_DOMAIN_ADAPTATION = None
    HYPERPARAM_FREEZE_SHAPES = True
    HYPERPARAM_NUM_HP_SHOTS = -1

    ##################
    # INITIALIZATION #
    ##################
    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of power_balance model to use.
        - scaling_law: H89, H98, and P_LH scaling laws to predict tau_e
        - sciml: neural network predicts tau_e, and we do the power balance calculation
        - unstructured_nn: a single neural network directly predicts stored energy evolution

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

        data_normalization: The method for normalizing the input data.
        - raw: No normalization, Ip, Wtot, etc. are in their original units
        - physics: Convert to typical dimensionless parameters like beta, q95, f_G, etc.
        - z_score: Within each device, normalize each variable to zero mean and unit variance.
        - coral: Use the CORAL method to align covariances of source and target domains (https://arxiv.org/abs/1612.01939)

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - mixing: Add a small amount of highly-weighted target data during training
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data

        freeze_submodules: Whether to freeze certain submodules of the model during training.
        The P_oh and P_rad signals are hard to quantify, we might want to let them drift from the original targets to better match Wtot_MJ

        num_hp_shots: The number of high-performance shots included in the training data, or -1 to include all high-performance shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        model_type: str  # scaling_law, sciml, unstructured_nn
        training_data: str  # cmod, tcv, cmod_tcv, exnihilo
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_submodules: bool
        num_hp_shots: int  # Number of high-performance shots included in training, or -1 for all (should be -1 if domain_adaptation is None)
        # So that's 3 (model type) x 4 (training data) x 4 (normalization) x 3 (domain adaptation) x 2 (freeze or not) x 6 (hp shots included) = 1728 results
        # Even less since the hyperparameter tuning is only done for a subset of cases
        prereqs: (
            list[Study.Case] | None
        )  # If not None, this case depends on the results of another case, and should only be run after that case has been run

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == PowerBalanceStudy.HYPERPARAM_TRAINING_DATA
                and self.data_normalization
                == PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION
                and self.domain_adaptation
                == PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION
                and self.freeze_submodules
                == PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES
                and self.num_hp_shots == PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS
            ):
                return True
            else:
                return False

        def is_impossible(self) -> bool:
            """Some cases don't make sense to run. Mark those cases as impossible and raise an error if we try to run them."""
            # Can't do transfer learning or training from nothing with 0 high-performance shots.
            if (
                self.domain_adaptation == "transfer" or self.training_data == "exnihilo"
            ) and self.num_hp_shots == 0:
                return True

            return False

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            else:
                return PowerBalanceStudy.Case(
                    model_type=self.model_type,
                    training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                    data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                    domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                    freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                    num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                )

        def __init__(
            self,
            model_type: str,
            training_data: str,
            data_normalization: str,
            domain_adaptation: str,
            freeze_submodules: bool,
            num_hp_shots: int,
        ):
            self.model_type = model_type
            self.training_data = training_data
            self.data_normalization = data_normalization
            self.domain_adaptation = domain_adaptation
            self.freeze_submodules = freeze_submodules
            self.num_hp_shots = num_hp_shots

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            if model_type not in [
                "scaling_law",
                "sciml",
                "unstructured_nn",
                "p_oh",
                "p_rad",
            ]:
                raise ValueError(f"Unknown model type: {model_type}")
            if domain_adaptation is None and num_hp_shots != -1:
                raise ValueError(
                    "If domain_adaptation is None, num_hp_shots must be -1 since this means we're training and testing on the same dataset and no high-performance data is being used"
                )
            if (
                model_type in ["p_oh", "p_rad"]
                and freeze_submodules != PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES
            ):
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES}) for submodule {model_type}"
                )

            prereqs = []

            # Add prereqs based on whether this is a hyperparameter tuning case or not.
            if not self.is_hyperparam_case():
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type=model_type,
                        training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                        data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                        domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                    )
                ]

            # Set prereqs based on model type
            if model_type in ["sciml", "scaling_law"]:
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type="p_oh",
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=num_hp_shots,
                    ),
                    PowerBalanceStudy.Case(
                        model_type="p_rad",
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=num_hp_shots,
                    ),
                ]

            # Set prereqs based on domain adaptation
            if domain_adaptation == "transfer":
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type=model_type,
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=None,
                        freeze_submodules=freeze_submodules,
                        num_hp_shots=-1,
                    )
                ]

            if len(prereqs) > 0:
                self.prereqs = prereqs
            else:
                self.prereqs = None

        def __str__(self):
            if self.domain_adaptation:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.da_{self.domain_adaptation}.freezesub_{self.freeze_submodules}.hp_{self.num_hp_shots}"
            else:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.freezesub_{self.freeze_submodules}"

        def __hash__(self):
            if self.domain_adaptation:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_submodules,
                        self.num_hp_shots,
                    )
                )
            else:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_submodules,
                    )
                )
