#!/usr/bin/env bash
#SBATCH --job-name=ff-setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/setup-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/setup-%j.err
set -euo pipefail

ROOT=/mnt/beegfs/ssrini27/failfast-step0
REPO_URL=${FAILFAST_REPO_URL:-https://github.com/srina1h/failfast-step0-arc.git}
REF=${FAILFAST_REF:-main}
RUN="$ROOT/setup-runs/$SLURM_JOB_ID"
TOOLS="$ROOT/tools"
mkdir -p "$RUN" "$TOOLS" "$ROOT/cache" "$ROOT/gates" "$ROOT/current" /home/ssrini27/failfast-arc/logs

curl --proto '=https' --tlsv1.2 -LsSf https://releases.astral.sh/github/uv/releases/download/0.12.17/uv-installer.sh -o "$RUN/uv-installer.sh"
UV_INSTALL_DIR="$TOOLS/uv-0.12.17" sh "$RUN/uv-installer.sh"
UV="$TOOLS/uv-0.12.17/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
"$UV" python install 3.11
git clone --filter=blob:none "$REPO_URL" "$RUN/repo"
git -C "$RUN/repo" checkout "$REF"
COMMIT=$(git -C "$RUN/repo" rev-parse HEAD)
"$UV" venv --python 3.11 "$RUN/env"
"$UV" pip install --python "$RUN/env/bin/python" torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$RUN/env/bin/python" "${RUN}/repo[test]"

export HF_HOME="$ROOT/cache/huggingface"
"$RUN/env/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
snapshot_download("Qwen/Qwen3-Coder-480B-A35B-Instruct", revision="9d90cf8fca1bf7b7acca42d3fc9ae694a2194069", allow_patterns=["tokenizer*", "*.json", "*.jinja"])
hf_hub_download("nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet", repo_type="dataset", revision="35455389ab51bf5e2306bfd436ef72d0f98bf882")
PY

ln -sfn "$RUN/repo" "$ROOT/current/repo"
ln -sfn "$RUN/env" "$ROOT/current/env"
"$RUN/env/bin/python" - <<PY
import json, platform
from pathlib import Path
import torch, transformers, peft
manifest = {
  "job_id": "$SLURM_JOB_ID",
  "git_commit": "$COMMIT",
  "python": platform.python_version(),
  "torch": torch.__version__,
  "transformers": transformers.__version__,
  "peft": peft.__version__,
  "repo_url": "$REPO_URL",
  "ref": "$REF",
}
Path("$RUN/setup_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
cp "$RUN/setup_manifest.json" "$ROOT/gates/setup_manifest.json"
printf '%s\n' "$COMMIT" > "$ROOT/gates/SETUP_OK"
