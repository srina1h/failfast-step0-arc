from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "full_issue_window"
    seed: int = 20261009
    input_arm: str = "issue_window"
    tasks: int = 500
    attempts_per_task: int = 11
    landmark_tokens: int = 20_000
    max_monitor_tokens: int = 4_096
    recent_steps: int = 8
    model_id: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    landmark_tokenizer_id: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    landmark_tokenizer_revision: str = "9d90cf8fca1bf7b7acca42d3fc9ae694a2194069"
    dataset_id: str = "nebius/SWE-rebench-openhands-trajectories"
    dataset_revision: str = "35455389ab51bf5e2306bfd436ef72d0f98bf882"
    dataset_file: str = "trajectories.parquet"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    rank_weight: float = 0.25
    optimizer_steps: int = 1_500
    gradient_accumulation: int = 1
    synthetic_outcome_token: bool = False
    minimum_headroom_bytes: int = 1_073_741_824
    validation_interval: int = 500
    initial_validation_step: int = 500
    checkpoint_interval: int = 50
    checkpoint_selection: str = 'log_loss'

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.input_arm not in {"issue", "issue_window"}:
            raise ValueError(f"invalid input_arm: {self.input_arm}")
        for name in ("tasks", "attempts_per_task", "optimizer_steps", "gradient_accumulation",
                     "validation_interval", "initial_validation_step", "checkpoint_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.landmark_tokens != 20_000:
            raise ValueError("landmark_tokens must remain 20000")
        if self.max_monitor_tokens != 4_096:
            raise ValueError("max_monitor_tokens must remain 4096")
        if self.checkpoint_selection not in ('log_loss','within_auc'):
            raise ValueError('invalid checkpoint_selection')

    def to_dict(self) -> dict:
        return asdict(self)
