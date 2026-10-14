import json

import pandas as pd
import pytest

from failfast_arc.scale import FIT_SEQUENCE, MODEL_SEEDS, freeze_cohort, write_configs


def metadata(tasks=900, mixed=True):
    return pd.DataFrame([
        {'repo':f'repo-{task//2}', 'instance_id':f'task-{task}',
         'trajectory_id':f'{task}-{attempt}', 'resolved':int(attempt >= 5) if mixed else 1,
         'total_tokens':25000}
        for task in range(tasks) for attempt in range(11)
    ])


def test_outer_folds_cover_every_attempt_once_and_keep_repositories_separate(tmp_path):
    cohort, folds, manifest = freeze_cohort(metadata())
    assert manifest['eligible_tasks'] == 900
    assert manifest['mixed_tasks'] == 900
    tests = folds[folds.split == 'test']
    assert set(tests.trajectory_id) == set(cohort.trajectory_id)
    assert tests.trajectory_id.is_unique
    assert tests.groupby('repo').fold.nunique().max() == 1
    for _, fold in folds.groupby('fold'):
        assert fold.groupby('repo').split.nunique().max() == 1
        assert fold.groupby('instance_id').split.nunique().max() == 1
    write_configs(tmp_path, manifest)
    for fold in range(3):
        configs = [json.loads((tmp_path/f'fold-{fold}/configs/issue_window-{seed}.json').read_text())
                   for seed in MODEL_SEEDS]
        assert [c['seed'] for c in configs] == list(MODEL_SEEDS)
        assert len({c['optimizer_steps'] for c in configs}) == 1
        assert 1500 < configs[0]['optimizer_steps'] <= 9000
        assert all(not c['synthetic_outcome_token'] for c in configs)
        assert all(c['max_monitor_tokens'] == 4096 for c in configs)
        assert all(c['initial_validation_step'] == 500 and c['validation_interval'] == 1500 for c in configs)
        assert all(c['checkpoint_selection'] == 'within_auc' for c in configs)
        control = json.loads((tmp_path/f'fold-{fold}/configs/synthetic-{MODEL_SEEDS[0]}.json').read_text())
        assert control['synthetic_outcome_token']


def test_audit_rejects_insufficient_mixed_tasks():
    with pytest.raises(ValueError, match='mixed tasks'):
        freeze_cohort(metadata(mixed=False))


def test_task_selection_does_not_change_with_metadata_row_order():
    first, folds_a, _ = freeze_cohort(metadata())
    second, folds_b, _ = freeze_cohort(metadata().sample(frac=1,random_state=44))
    assert set(first.trajectory_id) == set(second.trajectory_id)
    a=folds_a.sort_values(['fold','trajectory_id']).reset_index(drop=True)
    b=folds_b.sort_values(['fold','trajectory_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)


def test_finite_job_sequence_has_one_control_and_three_seeds_for_both_arms():
    assert FIT_SEQUENCE[0] == ('synthetic',20261009)
    assert len(FIT_SEQUENCE) == 7 and len(set(FIT_SEQUENCE)) == 7
    assert set(FIT_SEQUENCE[1:]) == {(arm,seed) for arm in ('issue','issue_window') for seed in MODEL_SEEDS}


def test_equivalence_summary_accepts_ties_and_rejects_strong_signal():
    from failfast_arc.scale_report import precision_summary
    frame=metadata(tasks=30).assign(score=0.5)
    tied=precision_summary({seed:frame for seed in MODEL_SEEDS},replicates=100)
    assert tied['bootstrap_equivalence_criterion_met']
    assert tied['pair_weighted_auc_mean'] == 0.5
    strong=frame.assign(score=frame.resolved.astype(float))
    detected=precision_summary({seed:strong for seed in MODEL_SEEDS},replicates=100)
    assert not detected['bootstrap_equivalence_criterion_met']
    assert detected['pair_weighted_auc_mean'] == 1.0


def test_checkpoint_selection_prefers_within_task_ranking_then_probability_error():
    from failfast_arc.config import ExperimentConfig
    from failfast_arc.train import checkpoint_key
    cfg=ExperimentConfig(checkpoint_selection='within_auc')
    assert checkpoint_key(cfg,1.2,0.7) < checkpoint_key(cfg,0.3,0.5)
    assert checkpoint_key(cfg,0.5,0.7) < checkpoint_key(cfg,1.2,0.7)
    with pytest.raises(ValueError,match='mixed validation'):
        checkpoint_key(cfg,0.5,float('nan'))
