from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig
from .hf_loading import load_tokenizer
from .model import build_monitor, compute_loss


def check_headroom(total_bytes: int, reserved_bytes: int, minimum_bytes: int) -> None:
    if total_bytes - reserved_bytes < minimum_bytes:
        raise RuntimeError(f"insufficient GPU headroom: {total_bytes - reserved_bytes} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = load_tokenizer(cfg.model_id, revision=cfg.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_monitor(cfg).cuda().train()
    text = "preflight " * (cfg.max_monitor_tokens * 2)
    batch = tokenizer([text, text], truncation=True, max_length=cfg.max_monitor_tokens, padding="max_length", return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    labels = torch.tensor([1.0, 0.0], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    logits = model(**batch).logits.reshape(-1)
    loss, _ = compute_loss(logits, labels, torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda"), cfg.rank_weight)
    loss.backward()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    result = {
        "device": props.name,
        "total_bytes": props.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loss": float(loss.detach().cpu()),
        "max_monitor_tokens": cfg.max_monitor_tokens,
    }
    result["headroom_bytes"] = result["total_bytes"] - result["peak_reserved_bytes"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "memory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    check_headroom(result["total_bytes"], result["peak_reserved_bytes"], cfg.minimum_headroom_bytes)
    (args.output / "PREFLIGHT_OK").write_text("ok\n")


if __name__ == "__main__":
    main()
