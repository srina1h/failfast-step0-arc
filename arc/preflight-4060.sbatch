#!/usr/bin/env bash
#SBATCH --job-name=ff-preflight
#SBATCH --partition=rtx4060ti16g
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/preflight-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/preflight-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
RUN="$ROOT/runs/preflight-$SLURM_JOB_ID"
HOME_GATES=/home/ssrini27/failfast-arc/gates
mkdir -p "$RUN"
rm -f "$GATES/PREFLIGHT_OK" "$GATES/SMOKE_OK"
rm -f "$HOME_GATES/PREFLIGHT_OK" "$HOME_GATES/SMOKE_OK"
cd "$REPO"
"$PYTHON" -m failfast_arc.preflight --config configs/full_issue_window.json --output "$RUN"
cp "$RUN/memory.json" "$GATES/memory.json"
cp "$RUN/memory.json" "$HOME_GATES/memory.json"
printf '%s\n' "$CURRENT_COMMIT" | tee "$GATES/PREFLIGHT_OK" "$HOME_GATES/PREFLIGHT_OK" >/dev/null
