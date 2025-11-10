# PRISM GPU Orchestrator Agent - Usage Guide

## Overview

The `prism-gpu-orchestrator` is a specialized agent designed for PRISM world-record graph-coloring pipeline work with strict GPU-first, no-stubs, no-fallbacks standards.

## When to Use This Agent

Use the prism-gpu-orchestrator agent for:

### ✅ GPU Implementation/Refactor Tasks
- Reservoir prediction (neuromorphic GPU)
- Thermodynamic equilibration (Phase 2 GPU kernels)
- Quantum-classical hybrid operations
- Transfer-entropy kernels
- Orchestration wiring between phases

### ✅ FluxNet RL Specific Tasks
- Wiring neuromorphic GPU reservoir into DSATUR tie-breaking
- Ensuring single CudaDevice across all phases
- Per-phase stream management and event synchronization
- ForceProfile GPU buffer management (device mirrors)
- Phase 2 thermodynamic kernel modifications for force commands

### ✅ Validation Tasks
- Running `prism-policy` checks with various SUB flags:
  - `SUB=cargo_check_cuda` - Verify CUDA compilation
  - `SUB=stubs` - Check for stub violations
  - `SUB=magic` - Check for magic numbers
  - `SUB=cuda_gates` - Validate CUDA feature gates
  - `SUB=gpu_reservoir` - Check reservoir GPU compliance
  - `SUB=gpu_info` - Verify GPU device info handling

### ✅ PhaseField and Integration Work
- Reviewing PhaseField (f64) construction
- Quantum find_coloring integration with device/stream patterns
- ADP Q-learning and Active Inference config validation
- Synergy/ablation checks (enable/disable components)

### ✅ PR/Change Review
- Any changes touching:
  - `foundation/prct-core/`
  - `foundation/neuromorphic/`
  - `foundation/shared-types/`
  - GPU wiring, correctness, or performance

## How to Invoke (For Claude Code CLI)

When using Claude Code CLI, you can request the prism-gpu-orchestrator agent:

```
"Use the prism-gpu-orchestrator agent to implement the Phase 2 thermodynamic
kernel modifications for FluxNet force profile integration."
```

Or for validation:

```
"Use the prism-gpu-orchestrator agent to run policy checks on the new
FluxNet GPU buffers with SUB=cuda_gates and SUB=gpu_reservoir."
```

## FluxNet RL Integration Points for GPU Orchestrator

### 1. ForceProfile GPU Buffers (Phase A)
**Task:** Implement pinned host memory + device buffer mirrors
**Why GPU Orchestrator:** Ensures proper CudaDevice sharing and stream synchronization

**Key requirements:**
- Use `Arc<CudaDevice>` (Article V compliance)
- Pinned host memory for fast transfers
- Device buffers synchronized before kernel launch
- No fallback CPU paths

### 2. Phase 2 Kernel Modification (Phase C)
**Task:** Extend `thermodynamic.cu` kernel with force parameters
**Why GPU Orchestrator:** CUDA kernel expertise, PTX validation

**Key requirements:**
- Add `float* f_strong`, `float* f_weak` parameters
- Modify phase force formula: `f_strong * repulsion - f_weak * coupling`
- Validate kernel compiles without stubs
- Check magic numbers in force computations

### 3. Per-Temperature Stream Management (Phase D)
**Task:** RL controller per-temp hook with proper stream/event sync
**Why GPU Orchestrator:** Stream orchestration patterns, event synchronization

**Key requirements:**
- Single CudaDevice throughout Phase 2
- Event markers between RL steps and kernel launches
- Stream synchronization before telemetry capture
- No CPU-GPU race conditions

### 4. Neuromorphic Reservoir Integration (Phase B)
**Task:** Wire reservoir difficulty scores into ForceProfile
**Why GPU Orchestrator:** Neuromorphic GPU module expertise

**Key requirements:**
- Extract `conflict_scores` from GPU reservoir
- Transfer to host for band assignment
- Validate Phase 0 → Phase 2 data flow
- No intermediate CPU fallbacks

### 5. Policy Validation (Phase G)
**Task:** Run comprehensive policy checks
**Why GPU Orchestrator:** Knows all policy check SUB flags

**Commands:**
```bash
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh
SUB=stubs ./tools/mcp_policy_checks.sh
SUB=cuda_gates ./tools/mcp_policy_checks.sh
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh
```

## GPU Orchestrator Compliance Standards

When implementing FluxNet, follow these standards:

### ✅ Constitutional Compliance
- **Article V:** Shared CUDA context (`Arc<CudaDevice>`)
- **Article VII:** Kernels compiled in `build.rs`, PTX loaded at runtime
- **Zero stubs:** No `todo!()`, `unimplemented!()`, or CPU fallbacks

### ✅ Memory Management
- **Host pinned:** Use `cuda_device.alloc_host()` for host buffers
- **Device buffers:** Owned by struct, not temporary allocations
- **Synchronization:** Explicit `htod_sync_copy_into()` before kernels

### ✅ Stream Patterns
- **Per-phase streams:** Managed by GPU stream pool
- **Event synchronization:** Use `CudaEvent` for dependencies
- **No blocking:** Prefer async patterns (future cudarc upgrade)

### ✅ Deterministic Mode
- **Config toggle:** `deterministic = true` in TOML
- **Fixed seeds:** RNG seeded consistently
- **No race conditions:** Proper synchronization

## Example: FluxNet Phase C Implementation Request

**Good request for GPU Orchestrator:**

> "Use the prism-gpu-orchestrator agent to implement Phase C of FluxNet:
>
> Modify `foundation/kernels/thermodynamic.cu` to accept ForceProfile parameters:
> - Add `float* f_strong` and `float* f_weak` device pointers
> - Add `float force_strong_gain` and `float force_weak_gain` scalars
> - Update phase force computation to: `f_strong[v] * force_strong_gain * repulsion - f_weak[v] * force_weak_gain * coupling`
> - Ensure kernel signature matches Rust FFI in `gpu_thermodynamic.rs`
> - Validate with `SUB=cuda_gates ./tools/mcp_policy_checks.sh`
> - Confirm no stubs or magic numbers
>
> Requirements:
> - Article V compliance (Arc<CudaDevice>)
> - No CPU fallbacks
> - Proper stream/event sync before kernel launch"

## Policy Check Reference

### Full Policy Check Suite
```bash
# CUDA compilation
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh

# Stub detection
SUB=stubs ./tools/mcp_policy_checks.sh

# Magic number detection
SUB=magic ./tools/mcp_policy_checks.sh

# CUDA feature gates
SUB=cuda_gates ./tools/mcp_policy_checks.sh

# GPU reservoir compliance
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh

# GPU info handling
SUB=gpu_info ./tools/mcp_policy_checks.sh

# NCU profiler view (if profiling data exists)
SUB=view_ncu ./tools/mcp_policy_checks.sh

# Nsys profiler view
SUB=view_nsys ./tools/mcp_policy_checks.sh
```

## FluxNet Implementation Checklist for GPU Orchestrator

Use the GPU Orchestrator agent for these specific FluxNet tasks:

- [ ] **Phase A.1:** ForceProfile device buffer allocation (Article V compliance)
- [ ] **Phase B.1:** Reservoir GPU → ForceProfile data flow validation
- [ ] **Phase C.2:** Thermodynamic CUDA kernel modification
- [ ] **Phase C.3:** Kernel launch site update in `gpu_thermodynamic.rs`
- [ ] **Phase D.5:** Per-temperature stream/event synchronization
- [ ] **Phase G.2:** Policy validation (`SUB=cuda_gates`, `SUB=stubs`, `SUB=gpu_reservoir`)
- [ ] **Phase I.2:** Final policy check before merge

## Integration with Other Agents

The GPU Orchestrator works best when:
- **Explore agent** has already mapped the codebase structure
- **Plan agent** has defined the implementation strategy
- **General-purpose agent** handles non-GPU tasks (configs, documentation)

**Handoff pattern:**
1. Use **Explore** to understand current GPU architecture
2. Use **Plan** to design FluxNet GPU integration
3. Use **prism-gpu-orchestrator** to implement GPU-specific code
4. Use **General-purpose** for testing and documentation

## Tips for Research Preview

If using Claude Code Research Preview (web interface), explicitly request GPU-focused work:

**Good:**
> "I need GPU-first implementation of ForceProfile device buffers with proper CudaDevice sharing and stream synchronization. Follow PRISM constitutional compliance (Article V, VII)."

**Better:**
> "Implement Phase C.2 (thermodynamic kernel modification) using PRISM GPU orchestrator standards: Arc<CudaDevice>, no stubs, proper stream/event sync, validate with policy checks."

## Summary

The prism-gpu-orchestrator agent ensures:
- ✅ GPU-first implementation (no CPU fallbacks)
- ✅ Constitutional compliance (Article V, VII)
- ✅ Stream/event synchronization correctness
- ✅ Policy validation (stubs, magic numbers, CUDA gates)
- ✅ Neuromorphic reservoir integration
- ✅ Phase 2 thermodynamic kernel expertise

Use it for **all GPU-related FluxNet work** to maintain world-record pipeline standards! 🚀
