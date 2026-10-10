#!/usr/bin/env bash
set -euo pipefail
GATES=/home/ssrini27/failfast-arc/gates
test -f "$GATES/SETUP_OK"
REMOTE=$(git ls-remote https://github.com/srina1h/failfast-step0-arc.git refs/heads/main | awk '{print $1}')
test "$(cat "$GATES/SETUP_OK")" = "$REMOTE"
cd /home/ssrini27/failfast-arc
sbatch --parsable preflight-4060.sbatch
