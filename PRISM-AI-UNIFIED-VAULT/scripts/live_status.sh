#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-120}"

exec python3 "${ROOT}/scripts/task_monitor.py" --watch "${INTERVAL}" --run-compliance --strict
