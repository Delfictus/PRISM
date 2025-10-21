#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PRISM_VAULT_ROOT:-}" ]]; then
  ROOT="$(cd "${PRISM_VAULT_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"

echo "🏁 Executing Sprint 1 hardening pipeline with advanced metrics…"
python3 03-AUTOMATION/master_executor.py --strict --use-sample-metrics "$@"
