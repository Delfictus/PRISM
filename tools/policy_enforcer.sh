#!/usr/bin/env bash
set -euo pipefail

# PRISM Policy Enforcer
# Locates violations, prints file:line hits, and suggests remediation steps
# Usage:
#   tools/policy_enforcer.sh run_all
#   tools/policy_enforcer.sh unwrap_audit
#   tools/policy_enforcer.sh cuda_migration_audit
#   tools/policy_enforcer.sh cuda_gates_audit
#   tools/policy_enforcer.sh kernel_symbol_audit
#   tools/policy_enforcer.sh config_loader_audit
#   tools/policy_enforcer.sh help

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOC_CHECKLIST="docs/GPU_QUANTUM_MODULES_CHECKLIST.md"
DOC_CONTRACT="AGENTS.md"
DOC_CONFIG="foundation/prct-core/configs/README.md"
DOC_SUMMARY="foundation/prct-core/COMPREHENSIVE_CONFIG_COMPLETE.md"

have() { command -v "$1" >/dev/null 2>&1; }
need() { have "$1" || { echo "Missing dependency: $1" >&2; exit 1; }; }

header() { printf "\n\e[1;36m== %s ==\e[0m\n" "$*"; }
note()   { printf "  - %s\n" "$*"; }
ref()    { printf "    • Ref: %s\n" "$*"; }

print_refs_common() {
  [ -f "$ROOT/$DOC_CHECKLIST" ] && ref "$DOC_CHECKLIST" || ref "GPU Quantum Modules Checklist (expected at $DOC_CHECKLIST)"
  [ -f "$ROOT/$DOC_CONTRACT" ] && ref "$DOC_CONTRACT" || ref "PRISM GPU‑First Integration Contract (AGENTS.md)"
}
print_refs_config() {
  [ -f "$ROOT/$DOC_CONFIG" ] && ref "$DOC_CONFIG" || ref "Config README (expected at $DOC_CONFIG)"
  [ -f "$ROOT/$DOC_SUMMARY" ] && ref "$DOC_SUMMARY" || ref "Configuration summary (expected at $DOC_SUMMARY)"
}

unwrap_audit() {
  header "Runtime unwrap/expect audit"
  need rg
  rg -n '(unwrap\(|expect\()' \
     foundation/prct-core/src \
     foundation/neuromorphic/src \
     -g '!**/tests/**' -g '!**/test/**' -g '!**/examples/**' \
     || true
  note "Remediate by:"
  note " - Replacing immediate post-assignment Option unwraps with .expect(\"just set <thing>\")"
  note " - Using ok_or_else(...) to map errors to PRCTError"
  note " - Avoiding partial_cmp unwraps with unwrap_or(Ordering::Equal)"
  print_refs_common
}

cuda_migration_audit() {
  header "Legacy CUDA API audit (cust/old cudarc patterns)"
  need rg
  rg -n 'CudaContext|ContextHandle|Module\b|DeviceBuffer<' foundation || true
  rg -n 'launch_builder\(|default_stream\(|memcpy_' foundation || true
  note "Migrate to cudarc 0.9 patterns:"
  note " - Device: CudaDevice (single per module); Streams optional"
  note " - Memory: device.alloc_zeros, htod_copy/htod_copy_into, dtoh_sync_copy, memset_zeros"
  note " - Kernels: device.load_ptx(...), device.get_func(...), func.clone().launch(LaunchConfig{...}, (args))"
  note " - Load kernels once; reuse CudaFunction"
  print_refs_common
}

cuda_gates_audit() {
  header "CUDA gates audit"
  need rg
  rg -n '#\[cfg\(feature\s*=\s*"cuda"\)\]' foundation || true
  note "Ensure GPU-only code is gated; provide CPU fallback or clear PRCTError in non-CUDA code paths."
  note "Check imports/exports and usage sites (e.g., platform.rs, quantum/lib.rs)."
  print_refs_common
}

kernel_symbol_audit() {
  header "Kernel symbol/launch audit"
  need rg
  rg -n 'load_ptx\(|get_func\(' foundation || true
  rg -n 'compile_ptx\(' foundation || true
  note "Verify:"
  note " - Symbols listed in load_ptx match those used in get_func"
  note " - No per-iteration load_ptx/get_func; load once, reuse functions"
  note " - Launch tuples pass scalars by value; device buffers by reference"
  print_refs_common
}

config_loader_audit() {
  header "Configuration I/O audit"
  need rg
  rg -n 'WorldRecordConfig::from_file|serde_json::from_str|toml::from_str' foundation/prct-core || true
  note "Validate:"
  note " - prct-core has serde + toml + serde_json deps"
  note " - shared-types serde feature enabled in prct-core/Cargo.toml"
  note " - WorldRecordConfig::validate() is invoked early (constructor)"
  note " - Examples load configs from file, not struct literals"
  print_refs_config
}

run_all() {
  header "Running full policy suite"
  unwrap_audit
  cuda_migration_audit
  cuda_gates_audit
  kernel_symbol_audit
  config_loader_audit

  header "Quick policy tools (optional, via prism-policy)"
  note 'To run: {"SUB":"cargo_check_cuda"}  {"SUB":"stubs"}  {"SUB":"cuda_gates"}'
}

case "${1:-help}" in
  run_all)                 run_all ;;
  unwrap_audit)            unwrap_audit ;;
  cuda_migration_audit)    cuda_migration_audit ;;
  cuda_gates_audit)        cuda_gates_audit ;;
  kernel_symbol_audit)     kernel_symbol_audit ;;
  config_loader_audit)     config_loader_audit ;;
  help|*) cat <<'USAGE'
PRISM Policy Enforcer
Usage:
  tools/policy_enforcer.sh run_all
  tools/policy_enforcer.sh unwrap_audit
  tools/policy_enforcer.sh cuda_migration_audit
  tools/policy_enforcer.sh cuda_gates_audit
  tools/policy_enforcer.sh kernel_symbol_audit
  tools/policy_enforcer.sh config_loader_audit

This script prints file:line violations and remediation guidance with doc references.
USAGE
  ;;
esac

