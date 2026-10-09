#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/beegfs/ssrini27/failfast-step0
CURRENT="$ROOT/current"
REPO="$CURRENT/repo"
PYTHON="$CURRENT/env/bin/python"
CACHE="$ROOT/cache"
GATES="$ROOT/gates"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required gate is missing: $1" >&2
    exit 2
  fi
}

require_file "$GATES/SETUP_OK"
export HF_HOME="$CACHE/huggingface"
export TRANSFORMERS_CACHE="$CACHE/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

