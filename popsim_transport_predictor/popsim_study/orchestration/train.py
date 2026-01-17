"""
POPSIM already has some training code in place, but for this study we need more flexibility
regarding how the datasets are constructed and how the models are trained.
"""


def train_model_standard(
    model_dir: str,
    training_data_case: str,
    model_case: str,
):
    """
    Train a model using standard learning (train and test on the same data distribution).
    """

    # 1. Organize datasets and make dataloaders
    # 2. Set up model configuration
    # 3. Train the model


def train_model_transfer(
    model_dir: str,
    training_data_case: str,
    model_case: str,
    num_hp_shots: int,
):
    """
    Train a model using transfer learning (train on historic data, test on high-performance data).
    The number of high-performance shots included in training is specified by `num_hp_shots`.
    """

    # 1. Organize datasets and make dataloaders
    # 2. Load pre-trained model from standard learning
    # 3. Fine-tune the model with high-performance shots
