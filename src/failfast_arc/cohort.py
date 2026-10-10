from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
    ordered = rows.sort_values(["repo", "instance_id", "trajectory_id"]).reset_index(drop=True)
    fractions = {"train": 0.70, "validation": 0.10, "test": 0.20}
    overall_rate = float(ordered.resolved.mean())
    total_tasks = ordered.instance_id.nunique()
    total_rows = len(ordered)
    best_rank: tuple[float, str] | None = None
    best_mapping: dict[str, str] | None = None
    for trial in range(1_024):
        trial_seed = int(_hash(seed, "repo-split", trial)[:8], 16)
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=trial_seed)
        trainval_idx, test_idx = next(outer.split(ordered, groups=ordered.repo))
        trainval = ordered.iloc[trainval_idx].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=trial_seed ^ 0x5F3759DF)
        train_idx, validation_idx = next(inner.split(trainval, groups=trainval.repo))
        repo_sets = {
            "train": set(trainval.iloc[train_idx].repo),
            "validation": set(trainval.iloc[validation_idx].repo),
            "test": set(ordered.iloc[test_idx].repo),
        }
        mapping = {repo: split for split, repos in repo_sets.items() for repo in repos}
        candidate = ordered.assign(split=ordered.repo.map(mapping))
        if candidate.groupby("split").resolved.nunique().min() < 2:
            continue
        score = 0.0
        for split, fraction in fractions.items():
            part = candidate[candidate.split == split]
            task_fraction = part.instance_id.nunique() / total_tasks
            row_fraction = len(part) / total_rows
            score += ((task_fraction - fraction) / fraction) ** 2
            score += ((row_fraction - fraction) / fraction) ** 2
            score += (float(part.resolved.mean()) - overall_rate) ** 2
        signature = "|".join(f"{repo}:{mapping[repo]}" for repo in sorted(mapping))
        ranked = (score, signature)
        if best_rank is None or ranked < best_rank:
            best_rank = ranked
            best_mapping = mapping
    if best_mapping is None:
        raise ValueError("cannot create repository-disjoint splits with both outcomes")
    mapping = best_mapping
    out = rows.copy()
    out["split"] = out.repo.map(mapping)
    if out.groupby("repo").split.nunique().max() != 1:
        raise AssertionError("repository crossed splits")
    if out.groupby("instance_id").split.nunique().max() != 1:
        raise AssertionError("task crossed splits")
    return out
