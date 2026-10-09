import pandas as pd

from failfast_arc.controls import assert_no_split_leakage, synthetic_positive_control


def test_leakage_is_rejected():
    frame = pd.DataFrame({"repo": ["r", "r"], "instance_id": ["a", "b"], "split": ["train", "test"]})
    try:
        assert_no_split_leakage(frame)
    except ValueError as exc:
        assert "repository" in str(exc)
    else:
        raise AssertionError("repository leakage was accepted")


def test_synthetic_control_is_detectable():
    assert synthetic_positive_control(seed=4, n=300) >= 0.90

