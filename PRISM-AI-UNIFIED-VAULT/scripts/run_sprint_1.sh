#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PRISM_VAULT_ROOT:-}" ]]; then
  if [[ ! -d "${PRISM_VAULT_ROOT}" ]]; then
    echo "PRISM_VAULT_ROOT points to missing directory: ${PRISM_VAULT_ROOT}" >&2
    exit 1
  fi
  ROOT="$(cd "${PRISM_VAULT_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"

echo "🏁 Executing Sprint 1 hardening pipeline with advanced metrics…"
python3 03-AUTOMATION/master_executor.py --strict --use-sample-metrics "$@"
