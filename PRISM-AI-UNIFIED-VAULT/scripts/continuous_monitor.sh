#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL=300
ITERATIONS=0

usage() {
  cat <<USAGE
Usage: ${0##*/} [--interval SECONDS] [--iterations N]

Continuously runs compliance validation (allowing missing artifacts) on a cadence.
Omit --iterations to run indefinitely.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL="$2"; shift 2;;
    --iterations)
      ITERATIONS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

echo "📡 Starting continuous governance monitor (interval=${INTERVAL}s, iterations=${ITERATIONS:-∞})"

COUNT=0
while [[ "$ITERATIONS" -eq 0 || $COUNT -lt $ITERATIONS ]]; do
  COUNT=$((COUNT + 1))
  timestamp="$(date --iso-8601=seconds)"
  echo "[$timestamp] Running compliance validator…"
  if ! python3 "$ROOT/scripts/compliance_validator.py" --allow-missing-artifacts; then
    echo "[$timestamp] ⚠️ Compliance issues detected."
  else
    echo "[$timestamp] ✅ Compliance check passed."
  fi
  if [[ "$ITERATIONS" -ne 0 && $COUNT -ge $ITERATIONS ]]; then
    break
  fi
  sleep "$INTERVAL"
done
