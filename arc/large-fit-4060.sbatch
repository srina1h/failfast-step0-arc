#!/usr/bin/env bash
#SBATCH --job-name=ff-large-fit
#SBATCH --partition=rtx4060ti16g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/large-fit-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/large-fit-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
require_gate_commit "$GATES/PREFLIGHT_OK"
BASE="$ROOT/runs/large-$CURRENT_COMMIT"
require_gate_commit "$BASE/PREPARED_OK"
FOLD=$1
ARM=$2
SEED=$3
[[ "$FOLD" =~ ^[012]$ ]]
[[ "$SEED" =~ ^202610(09|10|11)$ ]]
case "$ARM" in issue|issue_window|synthetic) ;; *) exit 2 ;; esac
FOLD_ROOT="$BASE/fold-$FOLD"
if [[ "$ARM" != synthetic ]]; then require_gate_commit "$FOLD_ROOT/CONTROL_OK"; fi
CONFIG="$FOLD_ROOT/configs/$ARM-$SEED.json"
PREPARED="$FOLD_ROOT/$ARM/prepared"
RUN="$FOLD_ROOT/$ARM/seed-$SEED"
HOME_EVIDENCE="/home/ssrini27/failfast-arc/recovered/large-$CURRENT_COMMIT/fold-$FOLD/$ARM/seed-$SEED"
mkdir -p "$RUN" "$HOME_EVIDENCE"
FOLD_EVIDENCE="/home/ssrini27/failfast-arc/recovered/large-$CURRENT_COMMIT/fold-$FOLD"
printf '%s|%s|%s|%s\n' "$SLURM_JOB_ID" "$FOLD" "$ARM" "$SEED" >> "$FOLD_EVIDENCE/jobs.tsv"
stage() { printf '%s fold=%s arm=%s seed=%s stage=%s\n' "$(date -u +%FT%TZ)" "$FOLD" "$ARM" "$SEED" "$1"; }
export_evidence() {
  set +e
  for name in cohort_summary.json config.sha256; do cp "$PREPARED/$name" "$HOME_EVIDENCE/$name"; done
  find "$RUN" -type f \( -name metrics.json -o -name controls.json -o \
    -name train_history.jsonl -o -name config.json -o -name provenance.json -o \
    -name sha256sums.txt -o -name test_scores.parquet -o -name validation_scores.parquet -o \
    -name operating_points.csv -o -name TRAIN_OK -o -name EVAL_OK -o \
    -name FIT_OK -o -name resume_path.txt \) -print0 |
  while IFS= read -r -d '' file; do
    relative=${file#"$RUN"/}
    mkdir -p "$HOME_EVIDENCE/$(dirname "$relative")"
    cp "$file" "$HOME_EVIDENCE/$relative"
  done
}
trap export_evidence EXIT
cd "$REPO"
stage start
RESUME=()
if [[ -f "$RUN/train/resume_path.txt" ]]; then RESUME=(--resume); fi
if [[ ! -f "$RUN/train/TRAIN_OK" ]]; then
  "$PYTHON" -m failfast_arc.train --config "$CONFIG" --prepared "$PREPARED" \
    --output "$RUN/train" --max-run-steps 1000 "${RESUME[@]}"
fi
if [[ -f "$RUN/train/TRAIN_OK" && ! -f "$RUN/eval/EVAL_OK" ]]; then
  stage evaluation
  "$PYTHON" -m failfast_arc.evaluate --config "$CONFIG" --prepared "$PREPARED" \
    --checkpoint "$RUN/train/best" --output "$RUN/eval"
fi
if [[ -f "$RUN/eval/EVAL_OK" ]]; then
  if [[ "$ARM" == synthetic ]]; then
    "$PYTHON" - "$RUN/eval/metrics.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
if m['pooled_auc'] < 0.90 or m['within_task']['pair_weighted'] < 0.90:
    raise SystemExit('large fold synthetic control failed')
PY
    printf '%s\n' "$CURRENT_COMMIT" > "$FOLD_ROOT/CONTROL_OK"
    cp "$FOLD_ROOT/CONTROL_OK" "$HOME_EVIDENCE/CONTROL_OK"
  fi
  printf '%s\n' "$CURRENT_COMMIT" > "$RUN/FIT_OK"
  stage completed
fi

# Advance only through this approved finite sequence; failed gates never reach this line.
NEXT_ARM=$ARM
NEXT_SEED=$SEED
if [[ -f "$RUN/FIT_OK" ]]; then
  NEXT=$("$PYTHON" - "$ARM" "$SEED" <<'PY'
import sys
from failfast_arc.scale import FIT_SEQUENCE
position=FIT_SEQUENCE.index((sys.argv[1],int(sys.argv[2])))
if position+1 < len(FIT_SEQUENCE):
    arm, seed=FIT_SEQUENCE[position+1]
    print(f'{arm}|{seed}')
PY
)
  if [[ -z "$NEXT" ]]; then
    printf '%s\n' "$CURRENT_COMMIT" > "$FOLD_EVIDENCE/FOLD_COMPLETE"
    stage fold-complete
    exit 0
  fi
  IFS='|' read -r NEXT_ARM NEXT_SEED <<< "$NEXT"
fi
FOLLOW_JOB=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" \
  /home/ssrini27/failfast-arc/large-fit-4060.sbatch "$FOLD" "$NEXT_ARM" "$NEXT_SEED")
printf '%s next_job=%s fold=%s arm=%s seed=%s\n' "$(date -u +%FT%TZ)" "$FOLLOW_JOB" "$FOLD" "$NEXT_ARM" "$NEXT_SEED"
