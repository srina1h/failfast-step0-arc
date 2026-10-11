#!/usr/bin/env bash
#SBATCH --job-name=ff-smoke
#SBATCH --partition=rtx4060ti16g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/smoke-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/smoke-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
require_gate_commit "$GATES/PREFLIGHT_OK"
RUN="$ROOT/runs/smoke-$SLURM_JOB_ID"
HOME_GATES=/home/ssrini27/failfast-arc/gates
HOME_EVIDENCE=/home/ssrini27/failfast-arc/recovered/smoke-$SLURM_JOB_ID
mkdir -p "$RUN"
rm -f "$GATES/SMOKE_OK"
rm -f "$HOME_GATES/SMOKE_OK"
printf '%s\n' "$RUN" > "$GATES/smoke_run_path.txt"
printf '%s\n' "$RUN" > "$HOME_GATES/smoke_run_path.txt"

export_evidence() {
  set +e
  mkdir -p "$HOME_EVIDENCE"
  find "$RUN" -type f \( \
    -name 'metrics.json' -o \
    -name 'controls.json' -o \
    -name 'cohort_summary.json' -o \
    -name 'train_history.jsonl' -o \
    -name 'operating_points.csv' -o \
    -name 'test_scores.parquet' -o \
    -name 'validation_scores.parquet' -o \
    -name 'provenance.json' -o \
    -name 'config.json' -o \
    -name 'config.sha256' -o \
    -name 'sha256sums.txt' -o \
    -name 'EVAL_OK' -o \
    -name 'TRAIN_OK' -o \
    -name 'SMOKE_OK' \
  \) -print0 | while IFS= read -r -d '' file; do
    relative=${file#"$RUN"/}
    mkdir -p "$HOME_EVIDENCE/$(dirname "$relative")"
    cp "$file" "$HOME_EVIDENCE/$relative"
  done
  printf '%s\n' "$RUN" > "$HOME_EVIDENCE/source_run_path.txt"
}
trap export_evidence EXIT

cd "$REPO"
for CONFIG in configs/smoke_issue.json configs/smoke.json configs/smoke_synthetic.json; do
  NAME=$(basename "$CONFIG" .json)
  "$PYTHON" -m failfast_arc.prepare --config "$CONFIG" --output "$RUN/$NAME/prepared" --cache "$CACHE"
  "$PYTHON" -m failfast_arc.train --config "$CONFIG" --prepared "$RUN/$NAME/prepared" --output "$RUN/$NAME/train"
  "$PYTHON" -m failfast_arc.evaluate --config "$CONFIG" --prepared "$RUN/$NAME/prepared" --checkpoint "$RUN/$NAME/train/best" --output "$RUN/$NAME/eval"
done
"$PYTHON" - "$RUN/smoke_synthetic/eval/metrics.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
if metrics["pooled_auc"] < 0.90 or metrics["within_task"]["pair_weighted"] < 0.90:
    raise SystemExit("end-to-end synthetic positive control failed")
PY
printf '%s\n' "$CURRENT_COMMIT" > "$RUN/SMOKE_OK"
cp "$RUN/SMOKE_OK" "$GATES/SMOKE_OK"
cp "$RUN/SMOKE_OK" "$HOME_GATES/SMOKE_OK"
