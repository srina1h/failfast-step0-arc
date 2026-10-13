from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .cohort import select_cohort, split_repositories
from .config import ExperimentConfig
from .controls import assert_no_split_leakage
from .hf_loading import load_cached_hub_file, load_tokenizer
from .prefix import append_synthetic_outcome_marker, serialize_example


def _trajectory_id(row: dict) -> str:
    return str(row.get("trajectory_id") or row.get("id"))


def _messages(row: dict) -> list[dict]:
    value = row.get("trajectory") or row.get("messages") or []
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _total_tokens(messages: list[dict], tokenizer) -> int:
    total = 0
    for message in messages:
        body = str(message.get("content") or "") + str(message.get("reasoning_content") or "")
        for call in message.get("tool_calls") or []:
            body += str((call.get("function") or {}).get("arguments") or "")
        total += len(tokenizer.encode(body, add_special_tokens=False))
    return total


def _binary_label(value) -> int:
    if value in (True, 1, "1", "true", "True"):
        return 1
    if value in (False, 0, "0", "false", "False"):
        return 0
    raise ValueError(f"invalid resolved label: {value!r}")


def _summary(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    by_split = {}
    for split, group in frame.groupby("split"):
        mixed = group.groupby("instance_id").resolved.nunique().eq(2).sum()
        by_split[split] = {
            "repositories": int(group.repo.nunique()),
            "tasks": int(group.instance_id.nunique()),
            "trajectories": int(len(group)),
            "success_rate": float(group.resolved.mean()),
            "mixed_outcome_tasks": int(mixed),
        }
    return {"config": cfg.to_dict(), "splits": by_split}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    parquet_path = load_cached_hub_file(
        repo_id=cfg.dataset_id,
        repo_type="dataset",
        filename=cfg.dataset_file,
        revision=cfg.dataset_revision,
        cache_dir=args.cache,
    )
    agent_tokenizer = load_tokenizer(
        cfg.landmark_tokenizer_id, revision=cfg.landmark_tokenizer_revision, cache_dir=args.cache
    )
    monitor_tokenizer = load_tokenizer(
        cfg.model_id, revision=cfg.model_revision, cache_dir=args.cache
    )
    parquet = pq.ParquetFile(parquet_path)
    metadata = []
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            messages = _messages(row)
            total = _total_tokens(messages, agent_tokenizer)
            if total <= cfg.landmark_tokens:
                continue
            tid = _trajectory_id(row)
            metadata.append({
                "repo": str(row["repo"]),
                "instance_id": str(row["instance_id"]),
                "trajectory_id": tid,
                "resolved": _binary_label(row["resolved"]),
                "total_tokens": total,
            })
    selected = split_repositories(select_cohort(pd.DataFrame(metadata), cfg), cfg.seed)
    assert_no_split_leakage(selected)
    selected_ids = set(selected.trajectory_id)
    meta_by_id = {row["trajectory_id"]: row for row in selected.to_dict("records")}
    records = []
    for batch in parquet.iter_batches(batch_size=256):
        for raw in batch.to_pylist():
            tid = _trajectory_id(raw)
            if tid not in selected_ids:
                continue
            meta = meta_by_id[tid]
            row = {**meta, "messages": _messages(raw)}
            serialized = serialize_example(
                row,
                cfg.input_arm,
                cfg.landmark_tokens,
                cfg.max_monitor_tokens,
                agent_tokenizer,
                monitor_tokenizer,
            )
            text = serialized.text
            monitor_count = serialized.monitor_token_count
            if cfg.synthetic_outcome_token:
                text, monitor_count = append_synthetic_outcome_marker(
                    text,
                    meta["resolved"],
                    monitor_tokenizer,
                    cfg.max_monitor_tokens,
                )
            records.append({
                **meta,
                "text": text,
                "last_included_message": serialized.last_included_message,
                "landmark_token_count": serialized.landmark_token_count,
                "monitor_token_count": monitor_count,
            })
    prepared = pd.DataFrame(records)
    prepared.to_parquet(args.output / "examples.parquet", index=False)
    (args.output / "cohort_summary.json").write_text(json.dumps(_summary(prepared, cfg), indent=2, sort_keys=True) + "\n")
    config_text = json.dumps(cfg.to_dict(), sort_keys=True).encode()
    (args.output / "config.sha256").write_text(hashlib.sha256(config_text).hexdigest() + "\n")


if __name__ == "__main__":
    main()
