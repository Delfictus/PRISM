#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

err() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || err "Missing dependency: $1"; }

case "${SUB:-help}" in
  stubs)
    need rg
    # Find disallowed stubs/shortcuts (fixed regex with properly escaped parentheses)
    rg -n 'todo!|unimplemented!|panic!\(|dbg!\(|unwrap\(|expect\(' foundation/prct-core foundation/neuromorphic || true
    ;;
  magic)
    need rg
    # Heuristic for hardcoded literals in code paths (excludes obvious config constants)
    rg -n '(let|const)\s+.=\s(\d+\.?\d*|true|false)' foundation/prct-core/src | rg -v 'DEFAULT|Config|const DEFAULT' || true
    ;;
  cuda_gates)
    need rg
    rg -n '#\[cfg(feature = "cuda")\]' foundation/prct-core foundation/neuromorphic || true
    ;;
  gpu_reservoir)
    need rg
    rg -n 'GpuReservoirComputer|process_gpu|DeviceBuffer' foundation || true
    ;;
  cargo_check)
    need cargo
    cargo check --no-default-features
    ;;
  cargo_check_cuda)
    need cargo
    cargo check --features cuda
    ;;
  clippy_cuda)
    need cargo
    cargo clippy --features cuda -- -D warnings
    ;;
  gpu_info)
    need nvidia-smi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo
    if command -v nvcc >/dev/null 2>&1; then
      nvcc --version
    else
      echo "nvcc not found (CUDA toolkit not in PATH)"
    fi
    ;;
  view_ncu)
    shopt -s globstar nullglob
    for f in reports/ncu/**/ncu_full.csv; do
      echo "=== $f ==="
      sed -n '1,80p' "$f"
      echo
    done
    ;;
  view_nsys)
    shopt -s globstar nullglob
    for f in reports/nsys/**/nsys.log; do
      echo "=== $f ==="
      sed -n '1,120p' "$f"
      echo
    done
    ;;
  bench_matrix)
    if [ -x baseline-v1.0/scripts/run_full_dimacs_test.sh ]; then
      baseline-v1.0/scripts/run_full_dimacs_test.sh 2>&1 | tee bench.log
    else
      err "missing baseline-v1.0/scripts/run_full_dimacs_test.sh"
    fi
    ;;
  help|*)
    cat <<'USAGE'
SUB subcommands:
  stubs            - find todo!/unimplemented!/panic!/dbg!/unwrap/expect
  magic            - find hardcoded values in loops (heuristic)
  cuda_gates       - verify #[cfg(feature = "cuda")] presence
  gpu_reservoir    - verify neuromorphic GPU reservoir references
  cargo_check      - cargo check --no-default-features
  cargo_check_cuda - cargo check --features cuda
  clippy_cuda      - cargo clippy --features cuda (deny warnings)
  gpu_info         - GPU/driver/toolkit info
  view_ncu         - preview Nsight Compute CSVs
  view_nsys        - preview Nsight Systems logs
  bench_matrix     - optional benchmark runner
USAGE
    ;;
esac
