#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT}"

echo "🔐 Enforcing PRISM-AI governance (strict mode)…"
python3 scripts/compliance_validator.py --strict "$@"
