from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


@dataclass(frozen=True)
class MetricResult:
    pair_weighted: float
    task_weighted: float
    pairs: int
    tasks_used: int
    tasks_excluded: int

    def to_dict(self) -> dict:
        return asdict(self)


def within_task_auc(frame: pd.DataFrame) -> MetricResult:
    credits: list[float] = []
    per_task: list[float] = []
    excluded = 0
    for _, group in frame.groupby("instance_id"):
        positive = group.loc[group.resolved == 1, "score"].to_numpy(float)
        negative = group.loc[group.resolved == 0, "score"].to_numpy(float)
        if not len(positive) or not len(negative):
            excluded += 1
            continue
        task = []
        for p in positive:
            for n in negative:
                task.append(1.0 if p > n else 0.5 if p == n else 0.0)
        credits.extend(task)
        per_task.append(float(np.mean(task)))
    return MetricResult(
        pair_weighted=float(np.mean(credits)) if credits else float("nan"),
        task_weighted=float(np.mean(per_task)) if per_task else float("nan"),
        pairs=len(credits),
        tasks_used=len(per_task),
        tasks_excluded=excluded,
    )


def summarize_scores(frame: pd.DataFrame) -> dict:
    labels = frame.resolved.to_numpy(int)
    scores = frame.score.to_numpy(float)
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    within = within_task_auc(frame)
    return {
        "n": int(len(frame)),
        "tasks": int(frame.instance_id.nunique()),
        "success_rate": float(labels.mean()),
        "pooled_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, clipped)),
        "brier": float(brier_score_loss(labels, clipped)),
        "within_task": within.to_dict(),
    }


def fpr_operating_points(test: pd.DataFrame, validation: pd.DataFrame, rates=(0.05, 0.10, 0.25)) -> pd.DataFrame:
    # Higher failure_score means more likely to fail. resolved=0 is positive here.
    y_val = 1 - validation.resolved.to_numpy(int)
    s_val = validation.failure_score.to_numpy(float)
    fpr, _, thresholds = roc_curve(y_val, s_val)
    rows = []
    for target in rates:
        eligible = np.where(fpr <= target)[0]
        index = eligible[-1] if len(eligible) else 0
        threshold = float(thresholds[index])
        pred = test.failure_score.to_numpy(float) >= threshold
        y = 1 - test.resolved.to_numpy(int)
        negatives = y == 0
        positives = y == 1
        rows.append({
            "target_fpr": target,
            "threshold": threshold,
            "test_fpr": float(pred[negatives].mean()) if negatives.any() else float("nan"),
            "test_recall": float(pred[positives].mean()) if positives.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def bootstrap_intervals(frame: pd.DataFrame, seed: int, replicates: int = 1_000) -> dict:
    rng = np.random.default_rng(seed)
    repos = frame.repo.drop_duplicates().to_numpy()
    tasks = frame.instance_id.drop_duplicates().to_numpy()
    pooled, within_pair, within_task = [], [], []
    for _ in range(replicates):
        sampled_repos = rng.choice(repos, size=len(repos), replace=True)
        repo_parts = []
        for draw, repo in enumerate(sampled_repos):
            part = frame[frame.repo == repo].copy()
            part["bootstrap_repo"] = draw
            repo_parts.append(part)
        repo_sample = pd.concat(repo_parts, ignore_index=True)
        if repo_sample.resolved.nunique() == 2:
            pooled.append(roc_auc_score(repo_sample.resolved, repo_sample.score))

        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_parts = []
        for draw, task in enumerate(sampled_tasks):
            part = frame[frame.instance_id == task].copy()
            part["instance_id"] = part.instance_id.astype(str) + f"#bootstrap-{draw}"
            task_parts.append(part)
        task_result = within_task_auc(pd.concat(task_parts, ignore_index=True))
        if np.isfinite(task_result.pair_weighted):
            within_pair.append(task_result.pair_weighted)
            within_task.append(task_result.task_weighted)

    def interval(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [float("nan"), float("nan")]

    return {
        "replicates": replicates,
        "pooled_auc_95ci": interval(pooled),
        "within_pair_weighted_auc_95ci": interval(within_pair),
        "within_task_weighted_auc_95ci": interval(within_task),
    }
