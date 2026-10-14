"""Describe all seeds and test practical equivalence without selecting test results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scale import MODEL_SEEDS


def task_credits(frame):
    rows=[]
    for task, part in frame.groupby('instance_id'):
        assert part.repo.nunique() == 1
        p=part.loc[part.resolved == 1,'score'].to_numpy()
        n=part.loc[part.resolved == 0,'score'].to_numpy()
        credits=(p[:,None] > n).astype(float)+0.5*(p[:,None] == n)
        rows.append({'instance_id':task,'repo':part.repo.iloc[0],
                     'pairs':int(p.size*n.size),'credits':float(credits.sum())})
    return pd.DataFrame(rows).set_index('instance_id').sort_index()


def precision_summary(per_seed, replicates=5000):
    tables=[task_credits(per_seed[seed]) for seed in MODEL_SEEDS]
    for table in tables[1:]:
        pd.testing.assert_frame_equal(tables[0][['repo','pairs']],table[['repo','pairs']])
    pair_counts=tables[0].pairs.to_numpy()
    assert pair_counts.sum() > 0
    credit=np.stack([table.credits.to_numpy() for table in tables])
    seed_aucs=credit.sum(axis=1)/pair_counts.sum()
    repos=tables[0].repo.drop_duplicates().to_numpy()
    repo_codes=pd.Categorical(tables[0].repo,categories=repos).codes
    rng=np.random.default_rng(20261014)
    conditional, both=[] , []
    for _ in range(replicates):
        draws=rng.integers(0,len(repos),size=len(repos))
        weights=np.bincount(draws,minlength=len(repos))[repo_codes]
        denom=float(np.dot(weights,pair_counts))
        if denom == 0:
            continue
        aucs=credit @ weights / denom
        conditional.append(float(aucs.mean()))
        seed_draws=rng.integers(0,len(MODEL_SEEDS),size=len(MODEL_SEEDS))
        both.append(float(aucs[seed_draws].mean()))
    ci90=np.quantile(both,[0.05,0.95]).tolist()
    return {
        'pair_weighted_auc_mean':float(seed_aucs.mean()),
        'per_seed_pair_weighted_auc':dict(zip(map(str,MODEL_SEEDS),map(float,seed_aucs))),
        'mixed_test_tasks':int((pair_counts>0).sum()),'test_tasks':len(pair_counts),
        'pairs_per_seed':int(pair_counts.sum()),
        'repository_and_seed_bootstrap_90ci':ci90,
        'repository_and_seed_bootstrap_95ci':np.quantile(both,[0.025,0.975]).tolist(),
        'fixed_seed_repository_bootstrap_95ci':np.quantile(conditional,[0.025,0.975]).tolist(),
        'practical_equivalence_band':[0.45,0.55],
        'bootstrap_equivalence_criterion_met':bool(ci90[0]>0.45 and ci90[1]<0.55),
        'bootstrap_replicates':replicates,'bootstrap_seed':20261014,
        'uncertainty_scope':'resampling saved repository groups and three seeds; overlapping training folds are not retrained',
    }


def report(root):
    collected={arm:{seed:[] for seed in MODEL_SEEDS} for arm in ('issue','issue_window')}
    details=[]
    commit=(root/'PREPARED_OK').read_text().strip()
    for fold in range(3):
        control=root/f'fold-{fold}/synthetic/seed-{MODEL_SEEDS[0]}'
        assert (control/'CONTROL_OK').read_text().strip() == commit
        metrics=json.loads((control/'eval/metrics.json').read_text())
        assert metrics['pooled_auc']>=0.90 and metrics['within_task']['pair_weighted']>=0.90
        for arm in collected:
            for seed in MODEL_SEEDS:
                path=root/f'fold-{fold}/{arm}/seed-{seed}'
                assert (path/'FIT_OK').read_text().strip() == commit
                for row in (path/'eval/sha256sums.txt').read_text().splitlines():
                    digest,name=row.split(maxsplit=1)
                    assert hashlib.sha256((path/'eval'/name).read_bytes()).hexdigest() == digest
                cfg=json.loads((path/'train/config.json').read_text())
                assert cfg['seed']==seed and not cfg['synthetic_outcome_token']
                history=[json.loads(row) for row in (path/'train/train_history.jsonl').read_text().splitlines()]
                assert [h['step'] for h in history]==list(range(1,cfg['optimizer_steps']+1))
                best=min([h for h in history if 'validation_log_loss' in h],
                         key=lambda h:(-h['validation_within_auc'],h['validation_log_loss'],h['step']))
                frame=pd.read_parquet(path/'eval/test_scores.parquet')
                assert frame.split.eq('test').all()
                if arm=='issue':
                    assert frame.groupby('instance_id').score.nunique().eq(1).all()
                collected[arm][seed].append(frame)
                details.append({'fold':fold,'arm':arm,'seed':seed,
                                'best_validation_step':best['step'],
                                'metrics':json.loads((path/'eval/metrics.json').read_text()),
                                'operating_points':pd.read_csv(path/'eval/operating_points.csv').to_dict('records')})
    combined={arm:{} for arm in collected}
    identity=None
    columns=['trajectory_id','instance_id','repo','split','resolved']
    for arm in collected:
        for seed in MODEL_SEEDS:
            frame=pd.concat(collected[arm][seed],ignore_index=True).sort_values('trajectory_id').reset_index(drop=True)
            assert frame.trajectory_id.is_unique
            if identity is None:
                identity=frame[columns]
            else:
                pd.testing.assert_frame_equal(identity,frame[columns])
            combined[arm][seed]=frame
    return {'code_commit':commit,'all_complete_and_matched':True,
            'issue':precision_summary(combined['issue']),
            'issue_window':precision_summary(combined['issue_window']),
            'fold_seed_results':details}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    print(json.dumps(report(args.root),indent=2))


if __name__=='__main__':
    main()
