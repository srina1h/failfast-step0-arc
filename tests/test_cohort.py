import pandas as pd

from failfast_arc.cohort import select_cohort, split_repositories
from failfast_arc.config import ExperimentConfig


def _rows():
    rows = []
    for repo_idx in range(10):
        for task_idx in range(2):
            instance = f"r{repo_idx}-t{task_idx}"
            for attempt in range(12):
                rows.append({
                    "repo": f"repo-{repo_idx}",
                    "instance_id": instance,
                    "trajectory_id": f"{instance}-a{attempt}",
                    "resolved": attempt % 2,
                })
    return pd.DataFrame(rows)


def test_selection_is_stable_and_has_eleven_attempts():
    cfg = ExperimentConfig(tasks=12, attempts_per_task=11)
    a = select_cohort(_rows(), cfg)
    b = select_cohort(_rows().sample(frac=1, random_state=9), cfg)
    assert sorted(a.trajectory_id) == sorted(b.trajectory_id)
    assert (a.groupby("instance_id").size() == 11).all()
    assert a.instance_id.nunique() == 12


def test_repository_never_crosses_splits():
    selected = select_cohort(_rows(), ExperimentConfig(tasks=20, attempts_per_task=11))
    split = split_repositories(selected, seed=42)
    assert split.groupby("repo").split.nunique().max() == 1
    assert split.groupby("instance_id").split.nunique().max() == 1
    assert set(split.split) == {"train", "validation", "test"}


def test_repository_split_keeps_both_outcomes_in_every_split():
    rows = []
    for repo_idx in range(12):
        resolved = int(repo_idx >= 6)
        for attempt in range(3):
            rows.append({
                "repo": f"repo-{repo_idx}",
                "instance_id": f"task-{repo_idx}",
                "trajectory_id": f"task-{repo_idx}-a{attempt}",
                "resolved": resolved,
            })
    split = split_repositories(pd.DataFrame(rows), seed=0)
    assert (split.groupby("split").resolved.nunique() == 2).all()
