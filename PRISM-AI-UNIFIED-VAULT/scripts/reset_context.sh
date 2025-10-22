#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if WORKTREE="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO="${WORKTREE}"
else
  REPO="$(cd "${ROOT}/.." && pwd)"
fi

usage() {
  cat <<USAGE
Usage: ${0##*/} [--strict]

Refreshes compliance artifacts and task snapshots after reboot. Steps:
  1. Cleans telemetry/artifact outputs
  2. Verifies benchmark manifest
  3. Prints current roadmap status
  4. Runs compliance validator (strict by default)
  5. Executes governed master executor (sample metrics)
  6. Shows git status

Options:
  --strict    Run validator strictly (default)
  --lenient   Allow missing artifacts (overrides --strict)
USAGE
}

STRICT_FLAG="--strict"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT_FLAG="--strict"; shift;;
    --lenient)
      STRICT_FLAG="--allow-missing-artifacts"; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac

done

cleanup() {
  echo ":: Cleaning telemetry and artifact outputs"
  rm -f "${ROOT}/telemetry"/*.jsonl 2>/dev/null || true
  rm -f "${ROOT}/reports"/run_*.json || true
}

benchmark_check() {
  echo ":: Verifying benchmark artifacts"
  python3 "${ROOT}/scripts/verify_benchmarks.py" --manifest "${REPO}/benchmarks/bench_manifest.json"
}

status_snapshot() {
  echo ":: Snapshotting task status"
  python3 "${ROOT}/scripts/task_monitor.py" --once
}

meta_snapshot() {
  echo ":: Meta feature registry snapshot"
  local registry="${ROOT}/meta/meta_flags.json"
  if [[ ! -f "${registry}" ]]; then
    echo "   (meta registry not initialized yet)"
    return
  fi
  REGISTRY_PATH="${registry}" python3 - <<'PY'
import json
import os
from pathlib import Path

registry_path = Path(os.environ["REGISTRY_PATH"])
data = json.loads(registry_path.read_text())
for record in sorted(data["records"], key=lambda r: r["id"]):
    state = record["state"]["mode"]
    print(f"   {record['id']:>24s} -> {state}")
PY
}

compliance_run() {
  echo ":: Running compliance validator"
  python3 "${ROOT}/scripts/compliance_validator.py" ${STRICT_FLAG}
}

master_executor() {
  echo ":: Running governed master executor (includes federated simulation)"
  python3 "${ROOT}/03-AUTOMATION/master_executor.py" --strict --use-sample-metrics --skip-build --skip-tests --skip-benchmarks
}

verify_federated_signatures() {
  echo ":: Verifying federated signatures"
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

prune_build_artifacts() {
  echo ":: Pruning transient build artifacts"
  "${ROOT}/scripts/prune_build_artifacts.sh"
}

show_git_status() {
  echo ":: Git status"
  git -C "${REPO}" status -sb
}

cleanup
benchmark_check
status_snapshot
meta_snapshot
compliance_run
master_executor
verify_federated_signatures
prune_build_artifacts
show_git_status

echo "✅ Context refresh complete"
