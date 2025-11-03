# COMPLETE CUDA FIX - cudarc 0.9 Migration
## Paste This ENTIRE Prompt into Cursor Composer

**Press `Cmd/Ctrl + I` and paste EVERYTHING below**:

---

```
CRITICAL MISSION: Complete cudarc 0.9 API migration for PRISM-AI.

We have 90+ CUDA compilation errors due to API changes in cudarc 0.9.
ALL CUDA must work - NO shortcuts, NO disabling features.

=============================================================================
CUDARC 0.9 API CHANGES - COMPLETE REFERENCE
=============================================================================

OLD API (cudarc < 0.9)          →  NEW API (cudarc 0.9)
---------------------------------------------------------------------------
device.default_stream()         →  device.fork_default_stream()?
stream.synchronize()            →  device.synchronize()?
device.load_module(bytes)       →  device.load_ptx(ptx, name, fns)?
module.get_function("name")     →  module.get_func("name")?
kernel.launch_on_stream(...)    →  kernel.launch(config, args)?
stream.memcpy_stod(dst, src)    →  device.htod_sync_copy(dst, src)?
stream.memcpy_dtos(dst, src)    →  device.dtoh_sync_copy(dst, src)?
device.alloc_zeros(n)           →  device.alloc_zeros::<T>(n)?
CudaModule                      →  CudaModule (same but from load_ptx)
---------------------------------------------------------------------------

CRITICAL:
- Streams are created with fork_default_stream()
- PTX loading uses Ptx::from_bytes() then load_ptx()
- Kernel launch does NOT take stream as parameter
- Memory ops are on device, not stream
- Synchronize is on device, not stream

=============================================================================
STEP-BY-STEP MIGRATION
=============================================================================

STEP 1: Fix all default_stream() calls (90 occurrences)
--------------------------------------------------------
Find every instance of:
    .default_stream()

Replace with:
    .fork_default_stream()?

Example:
OLD: let stream = self.device.default_stream();
NEW: let stream = self.device.fork_default_stream()?;


STEP 2: Fix all load_module() calls (14 occurrences)
-----------------------------------------------------
Find every instance of:
    device.load_module(module_data)

Replace with:
    use cudarc::nvrtc::Ptx;
    let ptx = Ptx::from_bytes(module_data);
    device.load_ptx(ptx, "module_name", &["function1", "function2"])?

Example:
OLD:
let module = device.load_module(ptx_bytes)?;
let kernel = module.get_function("my_kernel")?;

NEW:
use cudarc::nvrtc::Ptx;
let ptx = Ptx::from_bytes(ptx_bytes);
let module = device.load_ptx(ptx, "my_module", &["my_kernel"])?;
let kernel = module.get_func("my_kernel")?;


STEP 3: Fix all memory copy operations
---------------------------------------
Find: stream.memcpy_stod
Replace: device.htod_sync_copy

Find: stream.memcpy_dtos
Replace: device.dtoh_sync_copy

Find: stream.memcpy_dtod
Replace: device.dtod_copy

Example:
OLD: stream.memcpy_stod(&mut gpu_buf, &cpu_data)?;
NEW: device.htod_sync_copy(&mut gpu_buf, &cpu_data)?;


STEP 4: Fix all kernel launch calls
------------------------------------
Find: kernel.launch_on_stream(stream, config, args)
Replace: kernel.launch(config, args)?

Example:
OLD: kernel.launch_on_stream(&stream, config, &[ptr1, ptr2])?;
NEW: kernel.launch(config, (arg1, arg2))?;

Note: Arguments may need to be tuples now, not arrays


STEP 5: Fix all synchronize() calls
------------------------------------
Find: stream.synchronize()
Replace: device.synchronize()?

Example:
OLD: stream.synchronize()?;
NEW: self.device.synchronize()?;


STEP 6: Fix CudaSlice usage
----------------------------
OLD: slice.len()  // method
NEW: slice.len    // field

Find all .len() calls on CudaSlice and remove the ()


STEP 7: Add Ptx imports where needed
-------------------------------------
Any file using load_ptx needs:
use cudarc::nvrtc::Ptx;

Add this import to all files that load PTX kernels.


=============================================================================
FILES TO MIGRATE (Complete List)
=============================================================================

Priority 1 - Core CUDA (Fix First):
- foundation/cuda/gpu_coloring.rs
- foundation/cuda/ensemble_generation.rs
- foundation/cuda/prism_pipeline.rs
- src/cuda/gpu_coloring.rs
- src/cuda/ensemble_generation.rs
- src/cuda/prism_pipeline.rs

Priority 2 - CMA GPU:
- foundation/cma/gpu_integration.rs
- foundation/cma/quantum/pimc_gpu.rs
- foundation/cma/transfer_entropy_gpu.rs
- foundation/cma/neural/neural_quantum.rs
- src/cma/gpu_integration.rs
- src/cma/neural/neural_quantum.rs

Priority 3 - Active Inference GPU:
- foundation/active_inference/gpu.rs
- foundation/active_inference/gpu_inference.rs
- foundation/active_inference/gpu_policy_eval.rs

Priority 4 - Orchestration GPU:
- foundation/orchestration/routing/gpu_transfer_entropy_router.rs
- foundation/orchestration/thermodynamic/gpu_thermodynamic_consensus.rs
- foundation/orchestration/local_llm/gpu_transformer.rs

Priority 5 - Statistical Mechanics GPU:
- foundation/statistical_mechanics/gpu.rs
- foundation/statistical_mechanics/gpu_integration.rs

Priority 6 - Other GPU:
- foundation/gpu/gpu_tensor_optimized.rs
- foundation/gpu/optimized_gpu_tensor.rs
- foundation/gpu/gpu_enabled.rs
- foundation/quantum_mlir/cuda_kernels.rs
- foundation/quantum_mlir/runtime.rs
- foundation/pwsa/gpu_classifier.rs
- foundation/pwsa/active_inference_classifier.rs
- foundation/adp/reinforcement.rs
- foundation/phase6/gpu_tda.rs
- foundation/phase6/predictive_neuro.rs


=============================================================================
EXECUTION PLAN
=============================================================================

1. Start with Priority 1 files (core CUDA) - fix all 6 files completely
2. Test: cargo check --lib --features cuda
3. Move to Priority 2 (CMA GPU) - fix all 6 files
4. Test again
5. Continue through Priority 3, 4, 5, 6
6. Final test: cargo check --all --all-features

For EACH file:
- Find all default_stream() → fork_default_stream()?
- Find all load_module() → load_ptx()
- Find all memcpy_* → htod_*/dtoh_*
- Find all launch_on_stream() → launch()
- Find all stream.synchronize() → device.synchronize()
- Add Ptx imports
- Fix CudaSlice.len() → .len


=============================================================================
VERIFICATION CHECKPOINTS
=============================================================================

After Priority 1 (core CUDA):
cargo check --lib --features cuda 2>&1 | grep "error\[" | wc -l
Expect: ~70 errors (20 fixed)

After Priority 2 (CMA GPU):
Expect: ~50 errors (40 fixed)

After Priority 3 (Active Inference):
Expect: ~40 errors (50 fixed)

After Priority 4-6 (Everything else):
Expect: 0 errors ✅


=============================================================================
BEGIN SYSTEMATIC MIGRATION
=============================================================================

Start with Priority 1 files. For each file:

1. Open the file
2. Find ALL cudarc API usage
3. Apply EVERY transformation from the reference above
4. Save
5. Move to next file

Be thorough. Be systematic. Fix EVERYTHING.

After completing ALL priorities, run:
cargo check --all --all-features

Target: Zero errors, ALL CUDA working.
```

---

**End of Prompt** - Paste everything above into Cursor Composer now!