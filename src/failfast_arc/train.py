from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .config import ExperimentConfig
from .model import build_monitor, compute_loss


def _mixed_groups(frame: pd.DataFrame) -> list[tuple[str, list[int], list[int]]]:
    groups = []
    for instance_id, group in frame.groupby("instance_id"):
        positive = group.index[group.resolved == 1].tolist()
        negative = group.index[group.resolved == 0].tolist()
        if positive and negative:
            groups.append((instance_id, positive, negative))
    return groups


def _batch(tokenizer, texts: list[str], cfg: ExperimentConfig, device: str):
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=cfg.max_monitor_tokens,
        padding="max_length",
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _validation_loss(model, tokenizer, frame: pd.DataFrame, cfg: ExperimentConfig, device: str) -> float:
    model.eval()
    labels, scores = [], []
    for start in range(0, len(frame), 2):
        part = frame.iloc[start : start + 2]
        logits = model(**_batch(tokenizer, part.text.tolist(), cfg, device)).logits.reshape(-1).float()
        labels.extend(part.resolved.astype(int).tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    model.train()
    return float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    frame = pd.read_parquet(args.prepared / "examples.parquet")
    train = frame[frame.split == "train"].reset_index(drop=True)
    validation = frame[frame.split == "validation"].reset_index(drop=True)
    groups = _mixed_groups(train)
    if not groups:
        raise RuntimeError("training split has no mixed-outcome tasks")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_monitor(cfg).cuda().train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, cfg.optimizer_steps // 20),
        num_training_steps=cfg.optimizer_steps,
    )
    rng = random.Random(cfg.seed)
    best = math.inf
    history = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, cfg.optimizer_steps + 1):
        _, positive, negative = rng.choice(groups)
        left = rng.choice(positive)
        right = rng.choice(negative)
        part = train.loc[[left, right]]
        batch = _batch(tokenizer, part.text.tolist(), cfg, "cuda")
        labels = torch.tensor(part.resolved.to_numpy(float), dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.reshape(-1)
            loss, pieces = compute_loss(
                logits,
                labels,
                torch.tensor([0], device="cuda"),
                torch.tensor([1], device="cuda"),
                cfg.rank_weight,
            )
            scaled = loss / cfg.gradient_accumulation
        scaled.backward()
        if step % cfg.gradient_accumulation == 0 or step == cfg.optimizer_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        entry = {"step": step, "loss": float(loss.detach().cpu()), "bce": float(pieces["bce"]), "rank": float(pieces["rank"])}
        validation_interval = min(500, cfg.optimizer_steps)
        if step % validation_interval == 0 or step == cfg.optimizer_steps:
            entry["validation_log_loss"] = _validation_loss(model, tokenizer, validation, cfg, "cuda")
            if entry["validation_log_loss"] < best:
                best = entry["validation_log_loss"]
                model.save_pretrained(args.output / "best")
                tokenizer.save_pretrained(args.output / "best")
        history.append(entry)
        if step % 10 == 0:
            with (args.output / "train_history.jsonl").open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
    (args.output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n")
    (args.output / "TRAIN_OK").write_text("ok\n")


if __name__ == "__main__":
    main()
