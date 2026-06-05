import pytest

from biocomptools.toollib.models import DataFile
from biocomptools.toollib.networkselector import _calibration_version_key, pick_latest_datafile


def _df(cal: str, priority: int = 0) -> DataFile:
    return DataFile(
        file=f"/d/{cal}.parquet", calibration_name=cal, recipe_name="r", priority=priority
    )


def test_version_key_orders_numerically_not_lexicographically():
    assert _calibration_version_key("x_v1-H") < _calibration_version_key("x_v2_satfix-H")
    assert _calibration_version_key("x_v2-H") < _calibration_version_key("x_v10-H")


def test_version_key_ignores_trailing_content_hash():
    # the hash after the last '-' must not contribute a spurious version token
    assert _calibration_version_key("exp_v1-YF5EKX7OBZ47Q")[0] == 1


def test_version_key_falls_back_to_date_ordering_without_v_token():
    assert _calibration_version_key("2025-08-21_MIT1-H") < _calibration_version_key(
        "2026-02-15_fd-H"
    )


@pytest.mark.parametrize(
    "order",
    [
        ["e_v1-H", "e_v2_satfix-H", "e_v3_satfix-H"],
        ["e_v3_satfix-H", "e_v1-H", "e_v2_satfix-H"],
    ],
)
def test_pick_latest_chooses_highest_version(order):
    assert pick_latest_datafile([_df(c) for c in order]).calibration_name == "e_v3_satfix-H"


def test_pick_latest_empty_is_none():
    assert pick_latest_datafile([]) is None


def test_pick_latest_tiebreak_prefers_best_priority():
    chosen = pick_latest_datafile([_df("e_v3-H", priority=5), _df("e_v3-H", priority=0)])
    assert chosen.priority == 0
