#!/usr/bin/env bash
#SBATCH --job-name=ff-full-a6000
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/full-a6000-%A-%a.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/full-a6000-%A-%a.err
#SBATCH --array=0-1
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
require_file "$GATES/PREFLIGHT_OK"
require_file "$GATES/SMOKE_OK"
CONFIGS=(configs/full_issue.json configs/full_issue_window.json)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$CONFIG" .json)
RUN="$ROOT/runs/full-a6000-$SLURM_ARRAY_JOB_ID/$NAME"
mkdir -p "$RUN"
cd "$REPO"
"$PYTHON" -m failfast_arc.prepare --config "$CONFIG" --output "$RUN/prepared" --cache "$CACHE"
"$PYTHON" -m failfast_arc.train --config "$CONFIG" --prepared "$RUN/prepared" --output "$RUN/train"
"$PYTHON" -m failfast_arc.evaluate --config "$CONFIG" --prepared "$RUN/prepared" --checkpoint "$RUN/train/best" --output "$RUN/eval"
