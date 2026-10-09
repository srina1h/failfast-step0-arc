#!/usr/bin/env bash
set -euo pipefail
test -f /mnt/beegfs/ssrini27/failfast-step0/gates/SETUP_OK
cd /home/ssrini27/failfast-arc
sbatch --parsable preflight-4060.sbatch

