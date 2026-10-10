#!/usr/bin/env bash
set -euo pipefail
GATES=/home/ssrini27/failfast-arc/gates
test -f "$GATES/SETUP_OK"
test -f "$GATES/PREFLIGHT_OK"
test -f "$GATES/SMOKE_OK"
COMMIT=$(git ls-remote https://github.com/srina1h/failfast-step0-arc.git refs/heads/main | awk '{print $1}')
test "$(cat "$GATES/SETUP_OK")" = "$COMMIT"
test "$(cat "$GATES/PREFLIGHT_OK")" = "$COMMIT"
test "$(cat "$GATES/SMOKE_OK")" = "$COMMIT"
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
