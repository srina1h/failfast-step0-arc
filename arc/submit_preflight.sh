#!/usr/bin/env bash
set -euo pipefail
test -f /mnt/beegfs/ssrini27/failfast-step0/gates/SETUP_OK
ROOT=/mnt/beegfs/ssrini27/failfast-step0
test "$(cat "$ROOT/gates/SETUP_OK")" = "$(git -C "$ROOT/current/repo" rev-parse HEAD)"
cd /home/ssrini27/failfast-arc
sbatch --parsable preflight-4060.sbatch
