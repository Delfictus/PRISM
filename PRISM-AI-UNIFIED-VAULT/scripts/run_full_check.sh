#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if WORKTREE="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO="${WORKTREE}"
else
  REPO="$(cd "${ROOT}/.." && pwd)"
fi
MANIFEST="${REPO}/benchmarks/bench_manifest.json"

verify_federated_signatures() {
  echo
  echo "=== Verifying federated signatures ==="
  local scenario
  local label
  local summary
  local ledger
  local expected

  for scenario in "${ROOT}/artifacts/mec/M5/scenarios"/*.json; do
    [[ -f "${scenario}" ]] || continue
    label="$(basename "${scenario}")"
    label="${label%.json}"

    if [[ "${label}" == "baseline" ]]; then
      summary="${ROOT}/artifacts/mec/M5/simulations/epoch_summary.json"
      expected="default"
    else
      summary="${ROOT}/artifacts/mec/M5/simulations/epoch_summary_${label}.json"
      expected="${label}"
    fi
    ledger="${ROOT}/artifacts/mec/M5/ledger"

    if [[ ! -f "${summary}" ]]; then
      echo "❌ Missing federated summary for scenario '${label}': ${summary}"
      exit 1
    fi

    cargo run --quiet --manifest-path "${REPO}/Cargo.toml" --bin federated_sim -- \
      --verify-summary "${summary}" --verify-ledger "${ledger}" --expect-label "${expected}"
  done
}

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

verify_federated_signatures
