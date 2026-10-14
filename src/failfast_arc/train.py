from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from transformers import get_cosine_schedule_with_warmup

from .config import ExperimentConfig
from .hf_loading import load_tokenizer
from .model import build_monitor, compute_loss, load_trained_monitor
from .metrics import within_task_auc


def restore_training_state(state, cfg, prepared_hash, optimizer, scheduler, rng):
    if state['config'] != cfg.to_dict() or state['prepared_hash'] != prepared_hash:
        raise ValueError('resume checkpoint does not match configuration or prepared data')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    rng.setstate(state['pair_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'], state['best'], state['history']


def save_training_state(output, model, tokenizer, cfg, prepared_hash, optimizer, scheduler,
                        rng, step, best, history):
    checkpoint = output / f'checkpoint-{step}'
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save({
        'config': cfg.to_dict(), 'prepared_hash': prepared_hash,
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'pair_rng': rng.getstate(), 'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'step': step, 'best': best, 'history': history,
    }, checkpoint / 'training_state.pt')
    # Publish the pointer only after every checkpoint file is saved.
    pointer = output / 'resume_path.tmp'
    pointer.write_text(str(checkpoint.resolve()) + '\n')
    pointer.replace(output / 'resume_path.txt')


def binary_log_loss(labels, scores) -> float:
    return float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1]))


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
def _validation_metrics(model, tokenizer, frame: pd.DataFrame, cfg: ExperimentConfig, device: str):
    model.eval()
    labels, scores = [], []
    for start in range(0, len(frame), 2):
        part = frame.iloc[start : start + 2]
        logits = model(**_batch(tokenizer, part.text.tolist(), cfg, device)).logits.reshape(-1).float()
        labels.extend(part.resolved.astype(int).tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    model.train()
    return binary_log_loss(labels, scores), within_task_auc(frame.assign(score=scores)).pair_weighted


def checkpoint_key(cfg, validation_log_loss, validation_within_auc):
    if cfg.checkpoint_selection == 'within_auc':
        if not math.isfinite(validation_within_auc):
            raise ValueError('within-task checkpoint selection needs mixed validation tasks')
        return (-validation_within_auc, validation_log_loss)
    return validation_log_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-run-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if args.max_run_steps is not None and args.max_run_steps <= 0:
        raise ValueError('max-run-steps must be positive')
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
    tokenizer = load_tokenizer(cfg.model_id, revision=cfg.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import hashlib
    prepared_hash = hashlib.sha256((args.prepared / 'examples.parquet').read_bytes()).hexdigest()
    resume_path = None
    if args.resume:
        resume_path = Path((args.output / 'resume_path.txt').read_text().strip())
    elif (args.output / 'resume_path.txt').exists():
        raise ValueError('existing checkpoint requires --resume')
    model = (load_trained_monitor(cfg, resume_path, trainable=True)
             if resume_path else build_monitor(cfg)).cuda().train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, cfg.optimizer_steps // 20),
        num_training_steps=cfg.optimizer_steps,
    )
    rng = random.Random(cfg.seed)
    best = (math.inf,math.inf) if cfg.checkpoint_selection == 'within_auc' else math.inf
    history = []
    completed = 0
    if resume_path:
        state = torch.load(resume_path / 'training_state.pt', map_location='cpu', weights_only=False)
        completed, best, history = restore_training_state(
            state, cfg, prepared_hash, optimizer, scheduler, rng
        )
    stop = min(cfg.optimizer_steps, completed + (args.max_run_steps or cfg.optimizer_steps))
    if cfg.gradient_accumulation != 1 and args.max_run_steps is not None:
        raise ValueError('segmented training requires gradient_accumulation=1')
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for step in range(completed + 1, stop + 1):
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
        entry = {"step": step, "loss": float(loss.detach().cpu()), "bce": float(pieces["bce"]), "rank": float(pieces["rank"]),
                 "elapsed_job_seconds":time.monotonic()-started, "learning_rate":scheduler.get_last_lr()[0]}
        validation_interval = min(cfg.validation_interval, cfg.optimizer_steps)
        if step == cfg.initial_validation_step or step % validation_interval == 0 or step == cfg.optimizer_steps:
            print(json.dumps({'stage':'validation_start','step':step}),flush=True)
            entry["validation_log_loss"], entry['validation_within_auc'] = _validation_metrics(model, tokenizer, validation, cfg, "cuda")
            candidate = checkpoint_key(cfg,entry['validation_log_loss'],entry['validation_within_auc'])
            if candidate < best:
                best = candidate
                model.save_pretrained(args.output / "best")
                tokenizer.save_pretrained(args.output / "best")
        history.append(entry)
        if step % 10 == 0:
            print(json.dumps(entry, sort_keys=True), flush=True)
        if step % cfg.checkpoint_interval == 0 or step == stop:
            (args.output / 'train_history.jsonl').write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in history)
            )
            save_training_state(args.output, model, tokenizer, cfg, prepared_hash,
                                optimizer, scheduler, rng, step, best, history)
    (args.output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n")
    if stop == cfg.optimizer_steps:
        (args.output / "TRAIN_OK").write_text("ok\n")


if __name__ == "__main__":
    main()
