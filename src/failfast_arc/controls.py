from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def assert_no_split_leakage(frame: pd.DataFrame) -> None:
    if frame.groupby("repo").split.nunique().max() > 1:
        raise ValueError("repository leakage across splits")
    if frame.groupby("instance_id").split.nunique().max() > 1:
        raise ValueError("task leakage across splits")


def synthetic_positive_control(seed: int = 0, n: int = 500) -> float:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    signal = labels + rng.normal(0, 0.08, size=n)
    model = LogisticRegression(random_state=seed).fit(signal.reshape(-1, 1), labels)
    scores = model.predict_proba(signal.reshape(-1, 1))[:, 1]
    return float(roc_auc_score(labels, scores))


def shuffled_within_task(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["resolved"] = out.groupby("instance_id").resolved.transform(lambda x: rng.permutation(x.to_numpy()))
    return out
