#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-300}"

echo "📡 Starting continuous governance monitor (interval: ${INTERVAL}s)…"

while true; do
  timestamp="$(date --iso-8601=seconds)"
  echo "[$timestamp] Running compliance validator…"
  if ! python3 scripts/compliance_validator.py --allow-missing-artifacts; then
    echo "[$timestamp] ⚠️ Compliance issues detected."
  else
    echo "[$timestamp] ✅ Compliance check passed."
  fi
  sleep "${INTERVAL}"
done
