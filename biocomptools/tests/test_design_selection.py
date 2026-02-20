import numpy as np
import pytest

from biocomptools.toollib.design_selection import get_selection_losses_from_step_history


def test_get_selection_losses_uses_step_history_snapshot_mapping():
    class _StepHistoryLike(dict):
        pass

    step_history = _StepHistoryLike(
        all_losses=np.array([[[0.9, 0.1, 0.2]]], dtype=np.float32)
    )
    selected = get_selection_losses_from_step_history(
        step_history=step_history,
        n_replicates=1,
        n_targets=1,
        n_networks=3,
    )

    np.testing.assert_allclose(selected, np.array([[[0.9, 0.1, 0.2]]], dtype=np.float32))


def test_get_selection_losses_uses_step_history_last_batch_for_4d():
    step_history = {
        "all_losses": np.array(
            [
                [
                    [[0.9, 0.8, 0.7]],
                    [[0.3, 0.2, 0.1]],
                ],
                [
                    [[0.6, 0.5, 0.4]],
                    [[0.4, 0.5, 0.6]],
                ],
            ],
            dtype=np.float32,
        )
    }
    selected = get_selection_losses_from_step_history(
        step_history=step_history,
        n_replicates=2,
        n_targets=1,
        n_networks=3,
    )

    np.testing.assert_allclose(
        selected,
        np.array(
            [
                [[0.3, 0.2, 0.1]],
                [[0.4, 0.5, 0.6]],
            ],
            dtype=np.float32,
        ),
    )


def test_get_selection_losses_raises_when_shape_mismatch():
    step_history = {"all_losses": np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)}
    with pytest.raises(ValueError, match="shape mismatch"):
        get_selection_losses_from_step_history(
            step_history=step_history,
            n_replicates=2,
            n_targets=1,
            n_networks=3,
        )


def test_get_selection_losses_raises_when_step_history_missing():
    with pytest.raises(ValueError, match="step_history is None"):
        get_selection_losses_from_step_history(
            step_history=None,
            n_replicates=1,
            n_targets=1,
            n_networks=2,
        )
