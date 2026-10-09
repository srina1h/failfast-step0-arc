from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .controls import assert_no_split_leakage, shuffled_within_task, synthetic_positive_control
from .metrics import fpr_operating_points, summarize_scores, within_task_auc
from .model import load_trained_monitor


@torch.no_grad()
def _score(model, tokenizer, frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    model.eval()
    values = []
    for start in range(0, len(frame), 2):
        part = frame.iloc[start : start + 2]
        batch = tokenizer(
            part.text.tolist(), truncation=True, max_length=cfg.max_monitor_tokens,
            padding="max_length", return_tensors="pt"
        )
        batch = {key: value.cuda() for key, value in batch.items()}
        logits = model(**batch).logits.reshape(-1).float()
        values.extend(torch.sigmoid(logits).cpu().tolist())
    out = frame[["trajectory_id", "instance_id", "repo", "split", "resolved"]].copy()
    out["score"] = values
    out["failure_score"] = 1.0 - out.score
    return out


def _hashes(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "sha256sums.txt":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "sha256sums.txt").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.prepared / "examples.parquet")
    assert_no_split_leakage(frame)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_trained_monitor(cfg, args.checkpoint).cuda()
    validation = _score(model, tokenizer, frame[frame.split == "validation"].reset_index(drop=True), cfg)
    test = _score(model, tokenizer, frame[frame.split == "test"].reset_index(drop=True), cfg)
    test.to_parquet(args.output / "test_scores.parquet", index=False)
    validation.to_parquet(args.output / "validation_scores.parquet", index=False)
    metrics = summarize_scores(test)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    fpr_operating_points(test, validation).to_csv(args.output / "operating_points.csv", index=False)
    shuffled = shuffled_within_task(test, cfg.seed)
    controls = {
        "split_leakage": False,
        "synthetic_positive_auc": synthetic_positive_control(cfg.seed),
        "within_task_shuffle_auc": within_task_auc(shuffled).pair_weighted,
        "interpretation_gate": True,
    }
    controls["interpretation_gate"] = bool(controls["synthetic_positive_auc"] >= 0.90)
    (args.output / "controls.json").write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n")
    provenance = {
        "config": cfg.to_dict(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _hashes(args.output)
    if not controls["interpretation_gate"]:
        raise RuntimeError("performance controls failed")
    (args.output / "EVAL_OK").write_text("ok\n")


if __name__ == "__main__":
    main()
