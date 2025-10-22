#!/usr/bin/env bash
set -euo pipefail

# Removes transient build outputs that should not linger in version control.
# Patterns are based on the meta rollout housekeeping guidance.

usage() {
  cat <<'USAGE'
Usage: prune_build_artifacts.sh [--dry-run] [--verbose]

Deletes transient build outputs (deps/, libonnxruntime*, libprism_ai*, comprehensive_gpu_benchmark)
from the repository root so subsequent compliance runs start clean.

Options:
  --dry-run   Show what would be removed without deleting anything.
  --verbose   Print matched paths even if nothing is deleted (useful with --dry-run).
  -h, --help  Display this help message.
USAGE
}

DRY_RUN=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if WORKTREE="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO="${WORKTREE}"
else
  REPO="$(cd "${ROOT}/.." && pwd)"
fi

shopt -s nullglob
shopt -s dotglob

PATTERNS=(
  "deps"
  "libonnxruntime*"
  "libprism_ai*"
  "comprehensive_gpu_benchmark"
)

removed_any=0

log_match() {
  local rel="$1"
  if (( VERBOSE )); then
    echo "  match: ${rel}"
  fi
}

for pattern in "${PATTERNS[@]}"; do
  matches=("${REPO}/${pattern}")
  for path in "${matches[@]}"; do
    [[ -e "${path}" ]] || continue
    rel_path="${path#"${REPO}/"}"
    log_match "${rel_path}"
    if (( DRY_RUN )); then
      echo "[DRY] Would remove ${rel_path}"
    else
      rm -rf -- "${path}"
      echo "Removed ${rel_path}"
    fi
    removed_any=1
  done
done

if (( ! removed_any )); then
  if (( DRY_RUN )); then
    echo "[DRY] No transient build artifacts found."
  else
    echo "No transient build artifacts found."
  fi
fi

