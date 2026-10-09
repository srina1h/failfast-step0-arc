import pandas as pd

from failfast_arc.metrics import within_task_auc


def test_within_auc_counts_ties_as_half():
    frame = pd.DataFrame({
        "instance_id": ["a", "a", "b", "b"],
        "resolved": [1, 0, 1, 0],
        "score": [0.9, 0.1, 0.5, 0.5],
    })
    result = within_task_auc(frame)
    assert result.pair_weighted == 0.75
    assert result.task_weighted == 0.75
    assert result.tasks_used == 2


def test_unanimous_tasks_are_reported_not_scored():
    frame = pd.DataFrame({
        "instance_id": ["a", "a", "b", "b"],
        "resolved": [1, 0, 1, 1],
        "score": [0.9, 0.1, 0.8, 0.7],
    })
    result = within_task_auc(frame)
    assert result.tasks_used == 1
    assert result.tasks_excluded == 1

