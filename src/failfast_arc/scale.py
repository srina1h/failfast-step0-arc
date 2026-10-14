"""Freeze an all-eligible cohort, repository folds, and larger-run configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .cohort import select_cohort
from .config import ExperimentConfig
from .controls import assert_no_split_leakage
from .hf_loading import load_cached_hub_file, load_tokenizer
from .prefix import append_synthetic_outcome_marker, serialize_example
from .prepare import _binary_label, _messages, _summary, _total_tokens, _trajectory_id


DATA_SEED = 20261009
MODEL_SEEDS = (20261009, 20261010, 20261011)
MIN_MIXED_TOTAL = 600
MIN_MIXED_TEST = 150
MAX_UPDATES = 9000
FIT_SEQUENCE = [('synthetic', MODEL_SEEDS[0])]
FIT_SEQUENCE += [(arm, seed) for seed in MODEL_SEEDS for arm in ('issue', 'issue_window')]


def freeze_cohort(metadata: pd.DataFrame):
    counts = metadata.groupby('instance_id').trajectory_id.nunique()
    tasks = int(counts.ge(11).sum())
    if tasks < 500:
        raise ValueError('fewer than 500 eligible 11-attempt tasks')
    cohort = select_cohort(metadata, ExperimentConfig(tasks=tasks, seed=DATA_SEED))
    assert cohort.trajectory_id.is_unique
    mixed = int(cohort.groupby('instance_id').resolved.nunique().eq(2).sum())
    if mixed < MIN_MIXED_TOTAL:
        raise ValueError(f'only {mixed} mixed tasks; need {MIN_MIXED_TOTAL}')
    outer = GroupKFold(n_splits=3, shuffle=True, random_state=DATA_SEED)
    frames, summaries = [], []
    for fold, (trainval_ix, test_ix) in enumerate(outer.split(cohort, groups=cohort.repo)):
        trainval = cohort.iloc[trainval_ix].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=DATA_SEED+fold)
        train_ix, val_ix = next(inner.split(trainval, groups=trainval.repo))
        pieces = []
        for split, frame in [('train', trainval.iloc[train_ix]),
                             ('validation', trainval.iloc[val_ix]),
                             ('test', cohort.iloc[test_ix])]:
            pieces.append(frame.assign(split=split, fold=fold))
        frame = pd.concat(pieces, ignore_index=True)
        assert_no_split_leakage(frame)
        summary = _summary(frame, ExperimentConfig(tasks=tasks))['splits']
        if any(frame[frame.split == s].resolved.nunique() != 2 for s in ('train','validation','test')):
            raise ValueError(f'fold {fold} has a one-class split')
        if summary['test']['mixed_outcome_tasks'] < MIN_MIXED_TEST:
            raise ValueError(f'fold {fold} has too few mixed test tasks')
        if summary['validation']['mixed_outcome_tasks'] < 20:
            raise ValueError(f'fold {fold} has too few mixed validation tasks')
        if summary['train']['mixed_outcome_tasks'] < 100:
            raise ValueError(f'fold {fold} has too few mixed training tasks')
        updates = max(1500, math.ceil((1500*summary['train']['mixed_outcome_tasks']/105)/500)*500)
        if updates > MAX_UPDATES:
            raise ValueError(f'fold {fold} needs {updates} updates, above the {MAX_UPDATES} bound')
        frames.append(frame)
        summaries.append({'fold':fold, 'splits':summary, 'optimizer_steps':updates})
    frozen = pd.concat(frames, ignore_index=True)
    tests = frozen[frozen.split == 'test']
    assert tests.trajectory_id.is_unique and len(tests) == len(cohort)
    assert tests.groupby('repo').fold.nunique().max() == 1
    return cohort, frozen, {'eligible_tasks':tasks, 'attempts':len(cohort),
                           'mixed_tasks':mixed, 'folds':summaries,
                           'data_seed':DATA_SEED, 'model_seeds':list(MODEL_SEEDS)}


def write_configs(output: Path, manifest):
    for fold in manifest['folds']:
        directory = output / f"fold-{fold['fold']}" / 'configs'
        directory.mkdir(parents=True, exist_ok=True)
        for arm in ('issue','issue_window','synthetic'):
            for seed in (MODEL_SEEDS[:1] if arm == 'synthetic' else MODEL_SEEDS):
                cfg = ExperimentConfig(
                    name=f"large_fold{fold['fold']}_{arm}_{seed}",
                    tasks=manifest['eligible_tasks'], seed=seed,
                    input_arm='issue' if arm == 'issue' else 'issue_window',
                    synthetic_outcome_token=arm == 'synthetic',
                    optimizer_steps=fold['optimizer_steps'],
                    validation_interval=1500, initial_validation_step=500,
                    checkpoint_interval=500,
                    checkpoint_selection='within_auc',
                )
                (directory / f'{arm}-{seed}.json').write_text(json.dumps(cfg.to_dict(), indent=2)+'\n')


def audit(output: Path, cache: Path):
    cfg = ExperimentConfig()
    output.mkdir(parents=True, exist_ok=True)
    raw = load_cached_hub_file(repo_id=cfg.dataset_id, repo_type='dataset',
                               filename=cfg.dataset_file, revision=cfg.dataset_revision, cache_dir=cache)
    tokenizer = load_tokenizer(cfg.landmark_tokenizer_id, revision=cfg.landmark_tokenizer_revision, cache_dir=cache)
    metadata = []
    seen = 0
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            total = _total_tokens(_messages(row), tokenizer)
            if total > cfg.landmark_tokens:
                metadata.append({'trajectory_id':_trajectory_id(row), 'repo':str(row['repo']),
                                 'instance_id':str(row['instance_id']),
                                 'resolved':_binary_label(row['resolved']), 'total_tokens':total})
        seen += batch.num_rows
        if seen % 4096 == 0:
            print(json.dumps({'stage':'metadata', 'rows_scanned':seen,'active_rows':len(metadata)}), flush=True)
    pd.DataFrame(metadata).to_parquet(output / 'active_metadata.parquet', index=False)
    cohort, frozen, manifest = freeze_cohort(pd.DataFrame(metadata))
    cohort.to_parquet(output / 'selected.parquet', index=False)
    frozen.to_parquet(output / 'folds.parquet', index=False)
    manifest['dataset_revision'] = cfg.dataset_revision
    manifest['folds_sha256'] = hashlib.sha256((output/'folds.parquet').read_bytes()).hexdigest()
    (output / 'cohort_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_configs(output, manifest)
    print(json.dumps({'stage':'audit_complete',**manifest}),flush=True)
    (output / 'AUDIT_OK').write_text('ok\n')


def prepare(output: Path, cache: Path):
    cfg = ExperimentConfig()
    cohort = pd.read_parquet(output / 'selected.parquet')
    frozen = pd.read_parquet(output / 'folds.parquet')
    manifest = json.loads((output/'cohort_manifest.json').read_text())
    assert hashlib.sha256((output/'folds.parquet').read_bytes()).hexdigest() == manifest['folds_sha256']
    selected = cohort.set_index('trajectory_id').to_dict('index')
    raw = load_cached_hub_file(repo_id=cfg.dataset_id, repo_type='dataset',
                               filename=cfg.dataset_file, revision=cfg.dataset_revision, cache_dir=cache)
    agent = load_tokenizer(cfg.landmark_tokenizer_id,revision=cfg.landmark_tokenizer_revision,cache_dir=cache)
    monitor = load_tokenizer(cfg.model_id,revision=cfg.model_revision,cache_dir=cache)
    records = {arm:[] for arm in ('issue','issue_window','synthetic')}
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            tid = _trajectory_id(row)
            if tid not in selected:
                continue
            meta = {'trajectory_id':tid, **selected[tid]}
            for arm in ('issue','issue_window'):
                serial = serialize_example({**meta,'messages':_messages(row)},arm,cfg.landmark_tokens,
                                           cfg.max_monitor_tokens,agent,monitor)
                record = {**meta,'text':serial.text,'monitor_token_count':serial.monitor_token_count,
                          'landmark_token_count':serial.landmark_token_count,
                          'last_included_message':serial.last_included_message}
                records[arm].append(record)
                if arm == 'issue_window':
                    text, count = append_synthetic_outcome_marker(serial.text,meta['resolved'],monitor,cfg.max_monitor_tokens)
                    records['synthetic'].append({**record,'text':text,'monitor_token_count':count})
        print(json.dumps({'stage':'serialize','selected_rows':len(records['issue'])}),flush=True)
    for arm, rows in records.items():
        base = pd.DataFrame(rows)
        assert len(base) == len(cohort) and base.trajectory_id.is_unique
        for fold in range(3):
            split = frozen[frozen.fold == fold][['trajectory_id','split']]
            frame = base.merge(split,on='trajectory_id',validate='one_to_one')
            assert_no_split_leakage(frame)
            directory = output / f'fold-{fold}' / arm / 'prepared'
            directory.mkdir(parents=True,exist_ok=True)
            frame.to_parquet(directory/'examples.parquet',index=False)
            config = ExperimentConfig.load(output/f'fold-{fold}/configs/{arm}-{MODEL_SEEDS[0]}.json')
            (directory/'cohort_summary.json').write_text(json.dumps(_summary(frame,config),indent=2)+'\n')
            digest=hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()
            (directory/'config.sha256').write_text(digest+'\n')
            print(json.dumps({'stage':'prepared','fold':fold,'arm':arm,'rows':len(frame)}),flush=True)
    (output/'PREPARED_OK').write_text('ok\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['audit','prepare'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    (audit if args.stage=='audit' else prepare)(args.output,args.cache)


if __name__=='__main__':
    main()
