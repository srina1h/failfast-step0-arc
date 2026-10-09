from pathlib import Path

import pytest

from failfast_arc.config import ExperimentConfig


def test_full_config_keeps_scientific_constants():
    cfg = ExperimentConfig.load(Path("configs/full_issue_window.json"))
    assert cfg.landmark_tokens == 20_000
    assert cfg.max_monitor_tokens == 4_096
    assert cfg.model_id == "Qwen/Qwen3-0.6B"
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.optimizer_steps == 1_500
    assert cfg.input_arm == "issue_window"


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"unknown": 1}')
    with pytest.raises(ValueError, match="unknown"):
        ExperimentConfig.load(path)
