# World Record Pipeline: Fallback Scenarios & Performance Impact

**Author**: PRISM GPU Pipeline Architect
**Date**: 2025-11-02
**Version**: 1.0.0

## Overview

This document catalogs all fallback scenarios in the World Record Graph Coloring Pipeline, detailing:
- **When** the fallback occurs
- **Why** it triggers (GPU unavailable, error, config disabled)
- **What** alternative path is taken
- **Performance impact** (quantitative estimates)
- **Mitigation** strategies

---

## Global Fallback: CUDA Not Compiled

### Trigger
`#[cfg(not(feature = "cuda"))]` - CUDA feature not enabled at compile time

### Location
`world_record_pipeline.rs:1312-1361` (constructor)

### Impact
**50-80% slower overall pipeline execution**

### Details
```
[PIPELINE][FALLBACK] CUDA feature not compiled - using CPU-only mode
[PIPELINE][FALLBACK] Performance impact: ~50-80% slower (no GPU acceleration)
[PIPELINE][FALLBACK] Affected phases: reservoir (~10-50x slower), thermo (~5x slower), quantum (~3x slower)
```

### Affected Phases
- Phase 0: Reservoir → CPU fallback (~10-50x slower)
- Phase 1: Transfer Entropy → CPU fallback (~2-3x slower)
- Phase 2: Thermodynamic Equilibration → CPU fallback (~5x slower)
- Phase 3: Quantum Solver → CPU fallback (~3x slower)

### Mitigation
- Compile with `--features cuda`
- Ensure CUDA toolkit installed (11.8+)
- Verify GPU drivers (545+)

---

## Phase 0: Reservoir Conflict Prediction

### Fallback 1: GPU Disabled in Config

**Trigger**: `config.gpu.enable_reservoir_gpu = false`
**Location**: `world_record_pipeline.rs:1662-1672`
**Impact**: **10-50x slower** (depends on graph size)

```
[PHASE 0][FALLBACK] GPU reservoir disabled in config → CPU fallback
[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (GPU disabled by user)
```

**Reason**: User explicitly disabled GPU reservoir in config
**Mitigation**: Set `gpu.enable_reservoir_gpu = true` in config

---

### Fallback 2: GPU Initialization Failed

**Trigger**: `GpuReservoirConflictPredictor::predict_gpu()` returns `Err`
**Location**: `world_record_pipeline.rs:1647-1660`
**Impact**: **10-50x slower** + loss of dendritic processing

```
[PHASE 0][FALLBACK] GPU reservoir failed: <error details>
[PHASE 0][FALLBACK] Using CPU reservoir fallback
[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (loses GPU acceleration)
```

**Common Causes**:
- VRAM allocation failure (graph too large)
- Neuromorphic engine initialization error
- CUDA context issues

**Mitigation**:
- Reduce graph size
- Check VRAM availability (`nvidia-smi`)
- Verify GPU is not locked by another process

---

### Fallback 3: CUDA Not Compiled (CPU-Only)

**Trigger**: `#[cfg(not(feature = "cuda"))]`
**Location**: `world_record_pipeline.rs:1675-1693`
**Impact**: **10-50x slower**

```
[PHASE 0][FALLBACK] CUDA not compiled → CPU-only reservoir
[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (no GPU support)
```

**Mitigation**: Compile with `--features cuda`

---

## Phase 1: Transfer Entropy & Active Inference

### Fallback 1: Transfer Entropy GPU Disabled

**Trigger**: `config.gpu.enable_te_gpu = false`
**Location**: Phase checklist logging
**Impact**: **2-3x slower** (CPU TE computation)

```
[PHASE 1] TE active (CPU)
```

**Mitigation**: Set `gpu.enable_te_gpu = true`

---

### Fallback 2: CUDA Not Compiled

**Trigger**: `#[cfg(not(feature = "cuda"))]`
**Impact**: **2-3x slower** (CPU-only TE)

```
[PHASE 1] TE active (CPU only)
```

**Mitigation**: Compile with `--features cuda`

---

## Phase 2: Thermodynamic Equilibration

### Fallback 1: Thermodynamic GPU Disabled

**Trigger**: `config.gpu.enable_thermo_gpu = false`
**Impact**: **5x slower** (loses parallel tempering on GPU)

```
[PHASE 2] Thermodynamic active (CPU)
```

**Mitigation**: Set `gpu.enable_thermo_gpu = true`

---

### Fallback 2: CUDA Not Compiled

**Trigger**: `#[cfg(not(feature = "cuda"))]`
**Impact**: **5x slower**

```
[PHASE 2] Thermodynamic active (CPU only)
```

**Mitigation**: Compile with `--features cuda`

---

### Fallback 3: PIMC Requested But Not Implemented

**Trigger**: `config.use_pimc = true` but PIMC module not ready
**Location**: `world_record_pipeline.rs:558-570`
**Impact**: **None** (feature not yet used)

```
[PIPELINE][FALLBACK] PIMC requested but not implemented, will skip this phase
[PIPELINE][FALLBACK] Performance impact: none (experimental feature)
```

**Mitigation**: Wait for PIMC implementation (future work)

---

### Fallback 4: PIMC GPU Requested But Not Implemented

**Trigger**: `config.use_pimc = true && config.gpu.enable_pimc_gpu = true`
**Location**: `world_record_pipeline.rs:561-565`
**Impact**: **Error (config validation fails)**

```
Error: PIMC GPU requested but not yet implemented (use_pimc=true, enable_pimc_gpu=true)
```

**Mitigation**: Set `gpu.enable_pimc_gpu = false` or disable PIMC entirely

---

## Phase 3: Quantum-Classical Hybrid

### Fallback 1: Quantum Solver Failed

**Trigger**: `quantum_solver.find_coloring()` returns `Err`
**Location**: `world_record_pipeline.rs:1044-1049`
**Impact**: **20-30% slower** (loses quantum exploration)

```
[QUANTUM-CLASSICAL][FALLBACK] Quantum solver failed: <error>
[QUANTUM-CLASSICAL][FALLBACK] Using DSATUR-only refinement instead
[QUANTUM-CLASSICAL][FALLBACK] Performance impact: ~20-30% slower (loses quantum exploration)
```

**Common Causes**:
- PhaseField construction error
- Kuramoto state invalid
- QUBO solver divergence

**Mitigation**:
- Check Kuramoto phases are finite
- Verify graph connectivity
- Reduce `quantum.iterations` if solver unstable

---

### Fallback 2: Entire Phase Failed

**Trigger**: `solve_with_feedback()` returns `Err`
**Location**: `world_record_pipeline.rs:1906-1911`
**Impact**: **30% slower** (loses entire quantum-classical optimization)

```
[PHASE 3][FALLBACK] Quantum-Classical phase failed: <error>
[PHASE 3][FALLBACK] Continuing with best solution from previous phases
[PHASE 3][FALLBACK] Performance impact: ~30% (loses quantum-classical hybrid optimization)
```

**Mitigation**: Pipeline continues with best solution from Phases 0-2

---

### Fallback 3: Quantum GPU Disabled

**Trigger**: `config.gpu.enable_quantum_gpu = false`
**Impact**: **3x slower** (CPU quantum solver)

```
[PHASE 3] Quantum solver active (CPU)
```

**Mitigation**: Set `gpu.enable_quantum_gpu = true`

---

## Phase 5: Ensemble Consensus

### Fallback 1: No Valid Colorings

**Trigger**: All ensemble solutions have conflicts > 0
**Location**: `world_record_pipeline.rs:1158-1175`
**Impact**: **Approximate solution returned** (may have conflicts)

```
[ENSEMBLE][FALLBACK] No valid colorings found in ensemble
[ENSEMBLE][FALLBACK] Using best approximate solution (lowest conflicts)
[ENSEMBLE][FALLBACK] Performance impact: solution may have conflicts
[ENSEMBLE] ℹ️  Best approximate: X colors, Y conflicts
```

**Mitigation**:
- Increase `memetic.generations`
- Boost `thermo.replicas` for better exploration
- Enable more phases (quantum, thermodynamic)

---

### Fallback 2: Empty Solutions List

**Trigger**: `ensemble.solutions.is_empty()`
**Location**: Ensemble voting fails
**Impact**: **Pipeline error** (should never happen)

```
Error: No solutions available for ensemble voting (empty solutions list)
```

**Mitigation**: This is a logic error - report as bug

---

## Experimental Features

### Fallback: TDA Requested But Not Implemented

**Trigger**: `config.use_tda = true`
**Location**: `world_record_pipeline.rs:536-549`
**Impact**: **None** (phase skipped silently)

```
[PIPELINE][FALLBACK] TDA requested but not implemented, will skip this phase
[PIPELINE][FALLBACK] Performance impact: none (feature not used yet)
```

**Mitigation**: Disable `use_tda` until implementation complete

---

### Fallback: TDA GPU Requested

**Trigger**: `config.use_tda = true && config.gpu.enable_tda_gpu = true`
**Impact**: **Error (config validation fails)**

```
Error: TDA GPU requested but not yet implemented (use_tda=true, enable_tda_gpu=true)
```

**Mitigation**: Set `gpu.enable_tda_gpu = false`

---

### Fallback: GNN Screening Requested But Not Implemented

**Trigger**: `config.use_gnn_screening = true`
**Location**: `world_record_pipeline.rs:552-555`
**Impact**: **None** (experimental feature)

```
[PIPELINE][FALLBACK] GNN screening requested but not implemented, will skip this phase
[PIPELINE][FALLBACK] Performance impact: none (experimental feature)
```

**Mitigation**: Disable `use_gnn_screening`

---

## VRAM Guardrails

### Hard Limit: Thermodynamic Replicas Exceed VRAM

**Trigger**: `config.thermo.replicas > 56` (for 8GB devices)
**Location**: `world_record_pipeline.rs:498-512`
**Impact**: **Error (config validation fails)**

```
[VRAM][GUARD] thermo.replicas X exceeds safe limit for 8GB devices (max 56)
Error: thermo.replicas X exceeds VRAM limit (max 56 for 8GB devices)
```

**Mitigation**: Set `thermo.replicas <= 56`

---

### Hard Limit: Thermodynamic Temps Exceed VRAM

**Trigger**: `config.thermo.num_temps > 56` (for 8GB devices)
**Impact**: **Error (config validation fails)**

```
Error: thermo.num_temps X exceeds VRAM limit (max 56 for 8GB devices)
```

**Mitigation**: Set `thermo.num_temps <= 56`

---

### Runtime VRAM Check

**Trigger**: `validate_vram_requirements()` at pipeline start
**Location**: `world_record_pipeline.rs:575-641`
**Impact**: **Early failure** (prevents OOM crashes mid-run)

Checks:
- Thermodynamic VRAM: `per_replica_mb * replicas < 4GB`
- Reservoir VRAM: `reservoir_size^2 * 4 < 2GB`
- Quantum VRAM: `n^2 * 8 < 2GB` (PhaseField coherence matrix)

**Example**:
```
[VRAM][GUARD] Thermodynamic allocation estimate: 1200 MB (56 replicas)
[VRAM][GUARD] Reservoir allocation estimate: 64 MB (size=2000)
[VRAM][GUARD] Quantum solver allocation estimate: 8 MB
[VRAM][GUARD] ✅ VRAM validation passed for all enabled GPU phases
```

**Mitigation**: Reduce replicas, reservoir size, or graph size

---

## Performance Impact Summary

| Fallback Scenario | Performance Impact | Severity | Mitigation Priority |
|-------------------|-------------------|----------|-------------------|
| CUDA not compiled | 50-80% slower | Critical | High - recompile with `--features cuda` |
| GPU reservoir failed | 10-50x slower | High | High - check VRAM, GPU status |
| GPU reservoir disabled | 10-50x slower | High | Medium - enable in config |
| Thermodynamic GPU disabled | 5x slower | Medium | Medium - enable in config |
| Quantum solver failed | 20-30% slower | Medium | Low - pipeline continues |
| No valid colorings | Conflicts may exist | Medium | Medium - tune algorithm params |
| TE GPU disabled | 2-3x slower | Low | Low - CPU TE acceptable |
| PIMC requested | None (skipped) | None | None - experimental |
| TDA requested | None (skipped) | None | None - experimental |
| GNN requested | None (skipped) | None | None - experimental |

---

## Validation Checklist

Before running world record attempt:

1. **Compilation**
   - [ ] Compiled with `--features cuda`
   - [ ] `cargo check --features cuda` passes
   - [ ] No `todo!` or `unimplemented!` in hot paths

2. **Configuration**
   - [ ] `gpu.enable_reservoir_gpu = true`
   - [ ] `gpu.enable_thermo_gpu = true`
   - [ ] `gpu.enable_quantum_gpu = true`
   - [ ] `thermo.replicas <= 56`
   - [ ] `thermo.num_temps <= 56`

3. **GPU Status**
   - [ ] `nvidia-smi` shows GPU available
   - [ ] VRAM > 8GB free (or adjust limits)
   - [ ] Driver version >= 545
   - [ ] CUDA toolkit 11.8+ installed

4. **Fallback Logging**
   - [ ] Run with `2>&1 | tee pipeline.log`
   - [ ] Search for `[FALLBACK]` in logs
   - [ ] Verify no unexpected CPU fallbacks

5. **Policy Checks**
   - [ ] `rg "todo!|unimplemented!" foundation/prct-core/src/world_record_pipeline.rs` → empty
   - [ ] `cargo clippy --features cuda` → no errors
   - [ ] All expect() calls documented with `[DEFAULT][FATAL]` or similar

---

## Debugging Fallbacks

### Enable verbose logging
```bash
RUST_LOG=debug cargo run --features cuda --release
```

### Check GPU availability
```bash
nvidia-smi
# Expected: GPU 0 visible, VRAM free > 4GB
```

### Verify CUDA compilation
```bash
cargo build --features cuda 2>&1 | grep "CUDA"
# Expected: CUDA kernels compiled successfully
```

### Grep for fallback triggers
```bash
cargo run --features cuda 2>&1 | grep "\[FALLBACK\]"
# Expected: 0 fallbacks for optimal run
```

---

## Contact

For questions about fallback scenarios:
- GitHub Issues: `https://github.com/user/prism/issues`
- Discord: PRISM Dev Channel
- Email: dev@prism-graph-coloring.ai

---

**End of Fallback Scenarios Documentation**
