#!/usr/bin/env bash
#SBATCH --job-name=ff-large-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/ssrini27/failfast-arc/logs/large-data-%j.out
#SBATCH --error=/home/ssrini27/failfast-arc/logs/large-data-%j.err
set -euo pipefail
source /home/ssrini27/failfast-arc/common.sh
RUN="$ROOT/runs/large-$CURRENT_COMMIT"
HOME_EVIDENCE="/home/ssrini27/failfast-arc/recovered/large-$CURRENT_COMMIT"
mkdir -p "$RUN" "$HOME_EVIDENCE"
export_manifest() {
  set +e
  for name in cohort_manifest.json AUDIT_OK PREPARED_OK storage.json; do
    if [[ -f "$RUN/$name" ]]; then cp "$RUN/$name" "$HOME_EVIDENCE/$name"; fi
  done
}
trap export_manifest EXIT
cd "$REPO"
"$PYTHON" - "$ROOT" "$RUN" <<'PY'
import json, shutil, sys
from pathlib import Path
free=shutil.disk_usage(sys.argv[1]).free
Path(sys.argv[2],'storage.json').write_text(json.dumps({'free_bytes':free,'required_bytes':100*1024**3})+'\n')
print(json.dumps({'stage':'storage','free_bytes':free}),flush=True)
if free < 100*1024**3:
    raise SystemExit('less than 100 GiB filesystem space available')
PY
printf '%s stage=audit-start\n' "$(date -u +%FT%TZ)"
"$PYTHON" -m failfast_arc.scale audit --output "$RUN" --cache "$CACHE"
printf '%s stage=prepare-start\n' "$(date -u +%FT%TZ)"
"$PYTHON" -m failfast_arc.scale prepare --output "$RUN" --cache "$CACHE"
printf '%s\n' "$CURRENT_COMMIT" > "$RUN/PREPARED_OK"
printf '%s stage=prepared\n' "$(date -u +%FT%TZ)"
