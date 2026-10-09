#!/usr/bin/env bash
set -euo pipefail
cd /home/ssrini27/failfast-arc
mkdir -p logs
sbatch --parsable --export=ALL,FAILFAST_REPO_URL="${FAILFAST_REPO_URL:-https://github.com/srina1h/failfast-step0-arc.git}",FAILFAST_REF="${FAILFAST_REF:-main}" setup.sbatch

