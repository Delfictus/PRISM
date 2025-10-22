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

REPO="$(cd "${ROOT}/.." && pwd)"
TELEMETRY_FILE="${REPO}/telemetry/semantic_plasticity.jsonl"
DATASET="${ROOT}/meta/representation/dataset.json"
PIPELINE="${ROOT}/meta/representation/pipelines/extract_telemetry.py"

if [[ ! -f "${PIPELINE}" ]]; then
  echo "Missing telemetry extraction pipeline: ${PIPELINE}" >&2
  exit 1
fi

if [[ ! -f "${TELEMETRY_FILE}" ]]; then
  echo "No semantic plasticity telemetry found at ${TELEMETRY_FILE}" >&2
  exit 1
fi

echo ":: Extracting semantic plasticity dataset from telemetry"
python3 "${PIPELINE}" \
  --adapter-id semantic_plasticity_m4 \
  --dimension 4 \
  --input "${TELEMETRY_FILE}" \
  --output "${DATASET}"

USE_SAMPLE="--use-sample-metrics"
if [[ "${1:-}" == "--no-sample" ]]; then
  USE_SAMPLE=""
  shift
fi

echo ":: Running master executor semantic plasticity gate"
python3 "${ROOT}/03-AUTOMATION/master_executor.py" \
  --skip-build \
  --skip-tests \
  --skip-benchmarks \
  --strict \
  ${USE_SAMPLE:+$USE_SAMPLE} \
  "$@"

echo ":: Representation gate completed"
