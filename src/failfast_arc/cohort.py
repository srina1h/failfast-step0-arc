from __future__ import annotations

import hashlib

import pandas as pd

from .config import ExperimentConfig


def _hash(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(p) for p in parts)])
    return hashlib.sha256(value.encode()).hexdigest()


def select_cohort(rows: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    required = {"repo", "instance_id", "trajectory_id", "resolved"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing cohort columns: {sorted(missing)}")
    counts = rows.groupby("instance_id").trajectory_id.nunique()
    eligible = counts[counts >= config.attempts_per_task].index
    tasks = pd.DataFrame({"instance_id": eligible})
    task_repo = rows.drop_duplicates("instance_id").set_index("instance_id").repo
    tasks["repo"] = tasks.instance_id.map(task_repo)
    tasks["order"] = tasks.apply(lambda r: _hash(config.seed, r.repo, r.instance_id), axis=1)
    chosen_tasks = set(tasks.sort_values("order").head(config.tasks).instance_id)
    if len(chosen_tasks) < config.tasks:
        raise ValueError(f"only {len(chosen_tasks)} tasks have {config.attempts_per_task} attempts")
    out = rows[rows.instance_id.isin(chosen_tasks)].copy()
    out["attempt_order"] = out.apply(
        lambda r: _hash(config.seed, r.instance_id, r.trajectory_id), axis=1
    )
    out = out.sort_values(["instance_id", "attempt_order", "trajectory_id"])
    out = out.groupby("instance_id", group_keys=False).head(config.attempts_per_task)
    return out.drop(columns="attempt_order").reset_index(drop=True)


def split_repositories(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    task_counts = rows.drop_duplicates("instance_id").groupby("repo").size()
    ordered = sorted(task_counts.index, key=lambda repo: _hash(seed, repo))
    total = int(task_counts.sum())
    targets = {"train": 0.70 * total, "validation": 0.10 * total, "test": 0.20 * total}
    assigned = {name: 0 for name in targets}
    mapping: dict[str, str] = {}
    for repo in ordered:
        deficits = {name: targets[name] - assigned[name] for name in targets}
        split = max(deficits, key=lambda name: (deficits[name], name))
        mapping[repo] = split
        assigned[split] += int(task_counts[repo])
    out = rows.copy()
    out["split"] = out.repo.map(mapping)
    if out.groupby("repo").split.nunique().max() != 1:
        raise AssertionError("repository crossed splits")
    if out.groupby("instance_id").split.nunique().max() != 1:
        raise AssertionError("task crossed splits")
    return out
