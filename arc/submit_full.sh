#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/beegfs/ssrini27/failfast-step0
test -f "$ROOT/gates/PREFLIGHT_OK"
test -f "$ROOT/gates/SMOKE_OK"
COMMIT=$(git -C "$ROOT/current/repo" rev-parse HEAD)
test "$(cat "$ROOT/gates/PREFLIGHT_OK")" = "$COMMIT"
test "$(cat "$ROOT/gates/SMOKE_OK")" = "$COMMIT"
cd /home/ssrini27/failfast-arc
PROFILE=${1:-4060}
if [[ "$PROFILE" == "4060" ]]; then
  sbatch --parsable full-4060.sbatch
elif [[ "$PROFILE" == "a6000" ]]; then
  sbatch --parsable full-a6000.sbatch
else
  echo "profile must be 4060 or a6000" >&2
  exit 2
fi
