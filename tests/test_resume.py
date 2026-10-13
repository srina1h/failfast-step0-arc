import random

import numpy as np
import pytest
import torch

from failfast_arc.config import ExperimentConfig
from failfast_arc.train import restore_training_state, save_training_state


class TinyModel(torch.nn.Linear):
    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / 'model.pt')


class Tokenizer:
    def save_pretrained(self, path):
        (path / 'tokenizer.json').write_text('{}')


def test_resume_matches_uninterrupted_optimizer_and_randomness(tmp_path):
    cfg = ExperimentConfig()
    model = TinyModel(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    rng = random.Random(9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    save_training_state(tmp_path, model, Tokenizer(), cfg, 'data-hash', optimizer,
                        scheduler, rng, 50, 0.7, [{'step': 50}])
    expected = (rng.random(), random.random(), np.random.rand(), torch.rand(1))
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected_weights = model.weight.detach().clone()
    state = torch.load(tmp_path / 'checkpoint-50/training_state.pt', weights_only=False)
    restored = TinyModel(2, 1)
    restored.load_state_dict(torch.load(tmp_path / 'checkpoint-50/model.pt', weights_only=True))
    opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    step, best, history = restore_training_state(state, cfg, 'data-hash', opt, sched, rng)
    assert (step, best, history) == (50, 0.7, [{'step': 50}])
    actual = (rng.random(), random.random(), np.random.rand(), torch.rand(1))
    assert actual[:3] == expected[:3]
    assert torch.equal(actual[3], expected[3])
    restored(torch.ones(1, 2)).sum().backward()
    opt.step()
    assert torch.equal(restored.weight, expected_weights)
    with pytest.raises(ValueError, match='does not match'):
        restore_training_state(state, cfg, 'wrong-data', opt, sched, rng)


def test_full_control_keeps_model_and_threshold_settings():
    from pathlib import Path
    cfg = ExperimentConfig.load(Path('configs/full_synthetic.json'))
    assert cfg.tasks == 500 and cfg.optimizer_steps == 1500
    assert cfg.synthetic_outcome_token and cfg.max_monitor_tokens == 4096
