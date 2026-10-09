#!/usr/bin/env bash
set -euo pipefail
test -f /mnt/beegfs/ssrini27/failfast-step0/gates/PREFLIGHT_OK
cd /home/ssrini27/failfast-arc
sbatch --parsable smoke-4060.sbatch

