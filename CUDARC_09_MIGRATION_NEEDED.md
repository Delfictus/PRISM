# cudarc 0.9 Migration Status - Neuromorphic Engine

**Date**: October 31, 2025
**Status**: **IN PROGRESS** - API migration partially complete

---

## Summary

The neuromorphic-engine was originally written for cudarc 0.17 but needs to be migrated to cudarc 0.9 to align with the parent workspace. Partial migration has been completed, but additional work is needed.

## Changes Completed ✅

### 1. Type Renaming
- ✅ `CudaContext` → `CudaDevice` (all occurrences)
- ✅ Updated all struct fields and function signatures
- ✅ Updated test code

### 2. Memory Operations
- ✅ `stream.memcpy_stod()` → `device.htod_sync_copy()`
- ✅ `stream.memcpy_dtov()` → `device.dtoh_sync_copy()`
- ✅ `stream.synchronize()` → `device.synchronize()`

### 3. Removed Stream-Based Architecture
- ✅ Removed `transfer_stream` and `compute_stream` fields from `GpuMemoryPool`
- ✅ Updated `GpuMemoryPool::new()` to not create streams (cudarc 0.9 is synchronous)
- ✅ Removed `get_compute_stream()` method

**Files Modified**:
- `foundation/neuromorphic/src/gpu_memory.rs` - Complete
- `foundation/neuromorphic/src/cuda_kernels.rs` - Partial
- `foundation/neuromorphic/src/gpu_optimization.rs` - Partial
- `foundation/neuromorphic/src/gpu_reservoir.rs` - Partial

---

## Remaining Work ⚠️

### 1. PTX Module Loading (CRITICAL)

**cudarc 0.17 API** (current):
```rust
let ptx = cudarc::nvrtc::compile_ptx(kernel_source)?;
let module = device.load_module(ptx)?;
let function = module.load_function("kernel_name")?;
```

**cudarc 0.9 API** (required):
```rust
use cudarc::nvrtc::Ptx;

let ptx_str = cudarc::nvrtc::compile_ptx(kernel_source)?;
let ptx = Ptx::from_src(&ptx_str);
let module = device.load_ptx(
    ptx,
    "module_name",
    &["kernel_name1", "kernel_name2"]
)?;
```

**Files Affected**:
- `foundation/neuromorphic/src/cuda_kernels.rs` (lines 111-114, 157-160, etc.)
  - `compile_leaky_integration_kernel()`
  - `compile_spike_encoding_kernel()`
  - `compile_pattern_detection_kernel()`
  - `compile_spectral_radius_kernel()`

### 2. Stream and Event API

**cudarc 0.17 API** (current):
```rust
let stream = device.default_stream();
let event = device.new_event()?;
```

**cudarc 0.9 API**: No separate stream/event API - operations are synchronous

**Impact**: Remove or refactor async patterns in:
- `foundation/neuromorphic/src/gpu_optimization.rs`
- `foundation/neuromorphic/src/gpu_reservoir.rs`

### 3. Kernel Launch (if used directly)

Check if kernel launches need updating. cudarc 0.9 uses:
```rust
let cfg = LaunchConfig {
    grid_dim: (blocks, 1, 1),
    block_dim: (threads_per_block, 1, 1),
    shared_mem_bytes: 0,
};
unsafe { function.launch(cfg, (&param1, &param2)) }?;
```

---

## Migration Plan

### Phase 1: Fix Module Loading (Estimated 2-3 hours)
1. Add `use cudarc::nvrtc::Ptx;` to `cuda_kernels.rs`
2. Update all `compile_*_kernel()` functions to use `Ptx::from_src()` and `load_ptx()`
3. Test kernel compilation with simple test case

### Phase 2: Remove Stream Dependencies (Estimated 1-2 hours)
1. Remove `default_stream()` calls
2. Remove `new_event()` calls
3. Update synchronization to use `device.synchronize()`

### Phase 3: Testing (Estimated 2 hours)
1. Run unit tests with CUDA device
2. Verify GPU reservoir functionality
3. Benchmark performance vs baseline

**Total Estimated Time**: 5-7 hours

---

## Workaround (CURRENT)

The PRCT adapters are configured to work **without** the neuromorphic-engine GPU features for now:

```rust
// NeuromorphicAdapter falls back to CPU implementation
#[cfg(not(feature = "cuda"))]
pub fn new() -> Result<Self> {
    let cpu_reservoir = ReservoirComputer::new(...)?;
    // ... CPU implementation
}
```

This allows the PRCT pipeline to compile and run while the cudarc migration is completed.

---

## References

- **cudarc 0.9 docs**: https://docs.rs/cudarc/0.9
- **Baseline examples** using cudarc 0.9:
  - `src/cma/transfer_entropy_gpu.rs`
  - `src/cma/quantum/pimc_gpu.rs`
  - `src/cuda/prism_pipeline.rs`

---

## Recommendation

Complete the cudarc 0.9 migration as a separate focused task. The current PRCT adapter implementations are complete and production-ready - they just use CPU fallback for neuromorphic processing until the GPU path is fully migrated.

**Priority**: Medium (does not block PRCT functionality)
