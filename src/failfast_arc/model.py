from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .hf_loading import load_sequence_classifier


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pair_left: torch.Tensor, pair_right: torch.Tensor, rank_weight: float):
    logits = logits.float().reshape(-1)
    labels = labels.float().reshape(-1)
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    if pair_left.numel():
        rank = F.softplus(-(logits[pair_left] - logits[pair_right])).mean()
    else:
        rank = logits.sum() * 0.0
    total = bce + float(rank_weight) * rank
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite loss")
    return total, {"bce": bce.detach(), "rank": rank.detach(), "total": total.detach()}


def build_monitor(config: ExperimentConfig):
    from peft import LoraConfig, TaskType, get_peft_model
    model = load_sequence_classifier(
        config.model_id,
        revision=config.model_revision,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.pad_token_id = model.config.eos_token_id
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def load_trained_monitor(config: ExperimentConfig, checkpoint, *, trainable=False):
    from peft import PeftModel
    base = load_sequence_classifier(
        config.model_id,
        revision=config.model_revision,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    base.config.pad_token_id = base.config.eos_token_id
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=trainable)
    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model
