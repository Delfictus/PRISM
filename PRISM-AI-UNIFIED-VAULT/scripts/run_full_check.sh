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

if [[ -n "${PRISM_REPO_ROOT:-}" ]]; then
  if [[ ! -d "${PRISM_REPO_ROOT}" ]]; then
    echo "PRISM_REPO_ROOT points to missing directory: ${PRISM_REPO_ROOT}" >&2
    exit 1
  fi
  REPO="$(cd "${PRISM_REPO_ROOT}" && pwd)"
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

step "Executing governed master pipeline (sample metrics)"
python3 "${ROOT}/03-AUTOMATION/master_executor.py" --strict --use-sample-metrics --skip-build --skip-tests --skip-benchmarks

echo
echo "✅ Full compliance suite completed"
