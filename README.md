# FailFast-style value monitor on ARC

This repository tests one narrow question. Can a small text monitor rank a successful coding-agent attempt above a failed attempt for the same task at an early point in the run?

The experiment fine-tunes `Qwen/Qwen3-0.6B` with LoRA. It uses public OpenHands trajectories from the Nebius SWE-rebench release. It does not run a coding agent. It does not reproduce the full FailFast–RestartSmart system. In particular, it does not use dense test labels and it does not run the RestartSmart edit overlay.

## Fixed design

- Landmark: 20,000 transcript tokens.
- Monitor input: at most 4,096 Qwen tokens.
- Cohort: 500 tasks and 11 attempts per task for the full run.
- Split: repositories do not cross train, validation, and test sets.
- Inputs: issue text alone, or issue text plus the eight most recent completed steps and the latest earlier edit.
- Target: terminal success from the released harness.
- Loss: binary cross-entropy plus a within-task pairwise ranking loss.
- Main result: within-task AUC on held-out repositories.

The pinned model, tokenizer, data revisions, and all training settings are in `src/failfast_arc/config.py` and `configs/`.

## Why the within-task result matters

A pooled predictor can learn that some tasks are harder than other tasks. That result does not show that the run prefix contains information about the current attempt. Within-task AUC compares passing and failing attempts of the same task. It gives half credit when two attempts have the same score.

The issue-only arm is a tie reference. All attempts of one task receive the same issue text. Therefore, this arm cannot rank attempts of that task unless there is leakage or numerical instability.

## Controls

The test suite checks landmark boundaries, malformed tool arguments, deterministic selection, repository isolation, tied scores, and GPU headroom. The smoke run also trains a separate synthetic control. That control adds an obvious outcome token to each input. The pipeline must recover pooled and within-task AUC of at least 0.90 before the real result is interpreted.

## Local checks

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Local tests do not download the production model or dataset.

## ARC batch workflow

All cluster work uses Slurm batch jobs. The login node only submits jobs and stores small logs. Data, environments, model files, checkpoints, and outputs stay under `/mnt/beegfs/ssrini27/failfast-step0`.

Copy the files in `arc/` to `/home/ssrini27/failfast-arc`. Then run these commands on the ARC login node:

```bash
cd /home/ssrini27/failfast-arc
bash submit_setup.sh
bash submit_preflight.sh
bash submit_smoke.sh
```

Each command checks the previous gate before it submits the next job. The preflight performs a forward and backward pass on one passing–failing pair. Each sequence has 4,096 tokens. It rejects the 16 GB queue if less than 1 GiB remains reserved.

Full training is not submitted until setup, preflight, and smoke evidence all pass:

```bash
bash submit_full.sh 4060
```

The `a6000` profile is explicit. No script changes the queue automatically:

```bash
bash submit_full.sh a6000
```

## Outputs

Each evaluation writes:

- `metrics.json`
- `operating_points.csv`
- `test_scores.parquet`
- `validation_scores.parquet`
- `controls.json`
- `provenance.json`
- `sha256sums.txt`

The score table can be used by a separate abort-and-restart replay. This repository does not claim a stopping-policy benefit until that replay is complete.

## Data and license

The code downloads the public `nebius/SWE-rebench-openhands-trajectories` dataset at the pinned revision. Raw trajectories and model weights are not stored in this repository. Users must follow the licenses and terms of the source dataset and model.
