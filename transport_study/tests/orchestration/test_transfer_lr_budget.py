"""The step-budgeted transfer learning-rate scale (Study._scale_transfer_lr through transfer_lr_scale)."""

import pytest

from transport_study.orchestration.study import TRANSFER_LR_FLOOR, transfer_lr_scale


@pytest.mark.parametrize("max_epochs", [10, 240])
@pytest.mark.parametrize(
    ("steps_per_epoch", "expected_scale"),
    [
        # Step-starved: a dataset smaller than one batch keeps the full tuned lr
        (1, 1.0),
        # Between the clips the scale is 1 / steps_per_epoch, max_epochs cancels
        (4, 0.25),
        # Step-rich finetunes cool to the floor
        (10, TRANSFER_LR_FLOOR),
        (50, TRANSFER_LR_FLOOR),
    ],
)
def test_transfer_lr_scale(steps_per_epoch, max_epochs, expected_scale):
    assert transfer_lr_scale(steps_per_epoch, max_epochs) == pytest.approx(expected_scale)
