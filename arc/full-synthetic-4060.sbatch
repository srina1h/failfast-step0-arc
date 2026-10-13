#!/usr/bin/env bash
#SBATCH --job-name=ff-synth-full
#SBATCH --partition=rtx4060ti16g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/synth-full-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/synth-full-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
require_gate_commit "$GATES/PREFLIGHT_OK"
RUN="$ROOT/runs/full-synthetic-$CURRENT_COMMIT"
HOME_EVIDENCE="/home/ssrini27/failfast-arc/recovered/full-synthetic-$CURRENT_COMMIT"
mkdir -p "$RUN" "$HOME_EVIDENCE"
export_evidence() {
  set +e
  find "$RUN" -type f \( -name metrics.json -o -name controls.json -o \
    -name cohort_summary.json -o -name train_history.jsonl -o -name config.json -o \
    -name config.sha256 -o -name provenance.json -o -name sha256sums.txt -o \
    -name test_scores.parquet -o -name validation_scores.parquet -o \
    -name operating_points.csv -o -name TRAIN_OK -o -name EVAL_OK -o \
    -name FULL_SYNTHETIC_OK -o -name resume_path.txt \) -print0 |
  while IFS= read -r -d '' file; do
    relative=${file#"$RUN"/}
    mkdir -p "$HOME_EVIDENCE/$(dirname "$relative")"
    cp "$file" "$HOME_EVIDENCE/$relative"
  done
  printf '%s\n' "$RUN" > "$HOME_EVIDENCE/source_run_path.txt"
}
trap export_evidence EXIT
cd "$REPO"
if [[ ! -f "$RUN/prepared/config.sha256" ]]; then
  "$PYTHON" -m failfast_arc.prepare --config configs/full_synthetic.json \
    --output "$RUN/prepared" --cache "$CACHE"
fi
RESUME=()
if [[ -f "$RUN/train/resume_path.txt" ]]; then RESUME=(--resume); fi
if [[ ! -f "$RUN/train/TRAIN_OK" ]]; then
  "$PYTHON" -m failfast_arc.train --config configs/full_synthetic.json \
    --prepared "$RUN/prepared" --output "$RUN/train" --max-run-steps 250 "${RESUME[@]}"
fi
if [[ -f "$RUN/train/TRAIN_OK" && ! -f "$RUN/eval/EVAL_OK" ]]; then
  "$PYTHON" -m failfast_arc.evaluate --config configs/full_synthetic.json \
    --prepared "$RUN/prepared" --checkpoint "$RUN/train/best" --output "$RUN/eval"
fi
if [[ -f "$RUN/eval/EVAL_OK" ]]; then
  "$PYTHON" - "$RUN/eval/metrics.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
if metrics['pooled_auc'] < 0.90 or metrics['within_task']['pair_weighted'] < 0.90:
    raise SystemExit('full synthetic positive control failed')
PY
  printf '%s\n' "$CURRENT_COMMIT" > "$RUN/FULL_SYNTHETIC_OK"
  cp "$RUN/FULL_SYNTHETIC_OK" /home/ssrini27/failfast-arc/gates/FULL_SYNTHETIC_OK
fi
