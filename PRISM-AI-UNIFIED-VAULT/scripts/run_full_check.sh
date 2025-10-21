#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if WORKTREE="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO="${WORKTREE}"
else
  REPO="$(cd "${ROOT}/.." && pwd)"
fi
MANIFEST="${REPO}/benchmarks/bench_manifest.json"

step() {
  echo
  echo "=== $1 ==="
}

step "Verifying benchmark artifacts"
python3 "${ROOT}/scripts/verify_benchmarks.py" --manifest "${MANIFEST}"

step "Snapshotting project task status"
python3 "${ROOT}/scripts/task_monitor.py" --once

step "Running strict compliance validator"
python3 "${ROOT}/scripts/compliance_validator.py" --strict

step "Executing governed master pipeline (sample metrics + federated sim)"
python3 "${ROOT}/03-AUTOMATION/master_executor.py" --strict --use-sample-metrics --skip-build --skip-tests --skip-benchmarks

echo
echo "✅ Full compliance suite completed"
