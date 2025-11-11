#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; cd "$REPO_ROOT"

BASE="${BASE_CONFIG:-foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml}"
OUT="${OUT_CONFIG:-foundation/prct-core/configs/wr_hyper_active.toml}"
THREADS="${RAYON_NUM_THREADS:-24}"
TIME_LIM="${TIMEOUT:-90m}"

if [[ $# -gt 0 ]]; then
  LAYERS=( "$@" )
else
  LAYERS=( foundation/prct-core/configs/global_hyper.toml )
fi

mkdir -p foundation/prct-core/configs results/logs results/summaries

tools/toml_layered_merge.sh "$BASE" "$OUT" "${LAYERS[@]}"

ts="$(date -Iseconds)"; LOG="results/logs/wr_hyper_${ts}.log"
echo "[run] base=$BASE"; echo "[run] out=$OUT"; echo "[run] log=$LOG"

RAYON_NUM_THREADS="$THREADS" timeout "$TIME_LIM" \
  ./target/release/examples/world_record_dsjc1000 "$OUT" 2>&1 | tee "$LOG"

if [[ -f tools/summarize_wr_log.py ]]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  "$PY" tools/summarize_wr_log.py "$LOG" --base-config "$BASE" \
    --csv-append results/summaries/wr_hyper_summary.csv \
    --json-out "results/summaries/wr_hyper_${ts}.json" || true
  echo "[run] summary -> results/summaries/wr_hyper_${ts}.json"
fi
