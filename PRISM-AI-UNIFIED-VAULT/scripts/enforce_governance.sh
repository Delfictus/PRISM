#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PRISM_VAULT_ROOT:-}" ]]; then
  ROOT="$(cd "${PRISM_VAULT_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"

echo "🔐 Enforcing PRISM-AI governance (strict mode)…"
python3 scripts/compliance_validator.py --strict "$@"
