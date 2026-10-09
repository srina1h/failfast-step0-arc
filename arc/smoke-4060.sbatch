#!/usr/bin/env bash
#SBATCH --job-name=ff-smoke
#SBATCH --partition=rtx4060ti16g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/smoke-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/smoke-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
require_file "$GATES/PREFLIGHT_OK"
RUN="$ROOT/runs/smoke-$SLURM_JOB_ID"
mkdir -p "$RUN"
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
printf '%s\n' "$(git rev-parse HEAD)" > "$RUN/SMOKE_OK"
cp "$RUN/SMOKE_OK" "$GATES/SMOKE_OK"
printf '%s\n' "$RUN" > "$GATES/smoke_run_path.txt"
