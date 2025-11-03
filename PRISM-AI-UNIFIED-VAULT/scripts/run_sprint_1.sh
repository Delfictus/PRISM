#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT}"

echo "🏁 Executing Sprint 1 hardening pipeline with advanced metrics…"
python3 03-AUTOMATION/master_executor.py --strict --use-sample-metrics "$@"
