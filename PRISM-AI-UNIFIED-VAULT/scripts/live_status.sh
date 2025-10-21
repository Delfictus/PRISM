#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PRISM_VAULT_ROOT:-}" ]]; then
  ROOT="$(cd "${PRISM_VAULT_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
INTERVAL="${1:-120}"

exec python3 "${ROOT}/scripts/task_monitor.py" --watch "${INTERVAL}" --run-compliance --strict
