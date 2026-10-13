#!/usr/bin/env bash
#SBATCH --job-name=ff-setup-control
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/setup-control-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/setup-control-%j.err
set -euo pipefail
COMMIT=$1
BUNDLE=$2
WHEEL=$3
BUNDLE_HASH=$4
WHEEL_HASH=$5
ROOT=/mnt/beegfs/ssrini27/failfast-step0
CURRENT="$ROOT/current"
RUN="$ROOT/setup-runs/$SLURM_JOB_ID"
HOME_GATES=/home/ssrini27/failfast-arc/gates
test "$(sha256sum "$BUNDLE" | awk '{print $1}')" = "$BUNDLE_HASH"
test "$(sha256sum "$WHEEL" | awk '{print $1}')" = "$WHEEL_HASH"
test -x "$CURRENT/env/bin/python"
mkdir -p "$RUN" "$ROOT/gates" "$HOME_GATES"
git clone "$BUNDLE" "$RUN/repo"
git -C "$RUN/repo" checkout "$COMMIT"
test "$(git -C "$RUN/repo" rev-parse HEAD)" = "$COMMIT"
"$ROOT/tools/uv-0.12.17/uv" pip install --python "$CURRENT/env/bin/python" \
  --offline --no-deps --reinstall "$WHEEL"
ln -sfn "$RUN/repo" "$CURRENT/repo"
"$CURRENT/env/bin/python" - <<PY
import json
from pathlib import Path
import failfast_arc, torch, transformers, peft
manifest = {
    'git_commit': '$COMMIT', 'job_id': '$SLURM_JOB_ID',
    'mode': 'offline-reuse-verified-environment',
    'package_path': str(Path(failfast_arc.__file__).resolve()),
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'peft': peft.__version__,
}
Path('$RUN/setup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
cp "$RUN/setup_manifest.json" "$ROOT/gates/setup_manifest.json"
cp "$RUN/setup_manifest.json" "$HOME_GATES/setup_manifest.json"
printf '%s\n' "$COMMIT" | tee "$ROOT/gates/SETUP_OK" "$HOME_GATES/SETUP_OK" >/dev/null
