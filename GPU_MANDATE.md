# 🚨 CRITICAL: GPU MANDATE - NO EXCEPTIONS 🚨

## ABSOLUTE RULE: NEVER DISABLE GPU

**THIS IS NON-NEGOTIABLE:**

```
❌ NEVER disable CUDA features
❌ NEVER add CPU fallbacks
❌ NEVER comment out GPU code to "fix" bugs
❌ NEVER use #[cfg(not(feature = "cuda"))]
❌ NEVER suggest "run without --features cuda"
❌ NEVER disable GPU modules to bypass compilation errors
```

## THE ONLY ACCEPTABLE RESPONSE TO GPU BUGS

```
✅ DEBUG the GPU code
✅ FIX the CUDA kernel
✅ CORRECT the stream synchronization
✅ REPAIR the device buffer management
✅ RESOLVE the PTX compilation issue
✅ MAINTAIN full GPU acceleration at all times
```

## Why This Matters

PRISM is a **GPU-FIRST, GPU-ONLY** world-record graph coloring system:

1. **Performance:** GPU provides 10-50x speedup over CPU
2. **World Record:** CPU cannot achieve ≤83 colors on DSJC1000
3. **Architecture:** Entire pipeline designed for GPU parallelism
4. **Investment:** Months of GPU optimization would be wasted
5. **Mission:** Breaking world records requires GPU acceleration

## If You Encounter a GPU Bug

### ❌ WRONG Approach (NEVER DO THIS)

```rust
// NEVER DO THIS
#[cfg(feature = "cuda")]
fn some_gpu_function() -> Result<()> {
    // GPU code that has a bug
}

// ❌ NEVER add this fallback
#[cfg(not(feature = "cuda"))]
fn some_gpu_function() -> Result<()> {
    // CPU fallback
    println!("Running on CPU...");
    Ok(())
}
```

### ✅ CORRECT Approach (ALWAYS DO THIS)

```rust
// ✅ ALWAYS fix the bug properly
#[cfg(feature = "cuda")]
fn some_gpu_function() -> Result<()> {
    // DEBUG: What's the actual error?
    // FIX: Correct the root cause
    // VALIDATE: Run with SUB=cuda_gates policy check
    // VERIFY: Ensure GPU still accelerates

    // Properly fixed GPU code here
}
```

## Common GPU Bug Types & Proper Fixes

### Bug Type 1: PTX Compilation Failure

**❌ WRONG:**
```bash
# Remove --features cuda
cargo build --release
```

**✅ CORRECT:**
```bash
# Find the PTX error
cargo build --release --features cuda 2>&1 | grep -A 10 "error"

# Fix the CUDA kernel syntax
# Check foundation/kernels/*.cu files
# Validate kernel signature matches Rust FFI

# Rebuild with GPU
cargo build --release --features cuda
```

### Bug Type 2: CudaDevice Sharing Issue

**❌ WRONG:**
```rust
// Comment out GPU code
// let cuda_device = Arc::new(CudaDevice::new(0)?);
```

**✅ CORRECT:**
```rust
// Ensure Arc<CudaDevice> is shared properly (Article V)
let cuda_device = Arc::new(CudaDevice::new(0)?);

// Pass Arc::clone() to all phases
let phase0_result = run_phase0(Arc::clone(&cuda_device))?;
let phase1_result = run_phase1(Arc::clone(&cuda_device))?;
let phase2_result = run_phase2(Arc::clone(&cuda_device))?;
```

### Bug Type 3: Stream Synchronization Race

**❌ WRONG:**
```rust
// Remove async stream code, go back to synchronous
```

**✅ CORRECT:**
```rust
// Add proper event synchronization
let event = cuda_device.create_event()?;
stream.record_event(&event)?;
event.synchronize()?;

// Or use device.synchronize() for debugging
cuda_device.synchronize()?;
```

### Bug Type 4: Device Buffer Size Mismatch

**❌ WRONG:**
```rust
// Switch to CPU Vec instead of CudaSlice
let data = vec![0.0f32; n];
```

**✅ CORRECT:**
```rust
// Fix the buffer size calculation
let data_host = vec![0.0f32; n];
let data_device = cuda_device.htod_sync_copy(&data_host)?;

// Verify size matches kernel expectations
assert_eq!(data_device.len(), n);
```

### Bug Type 5: Kernel Launch Parameter Mismatch

**❌ WRONG:**
```rust
// Skip the kernel launch
// func.launch(cfg, params)?;
```

**✅ CORRECT:**
```rust
// Fix the parameter types to match kernel signature
// Check foundation/kernels/*.cu for expected types
unsafe {
    func.launch(
        cfg,
        (
            &device_buffer1,  // Correct type: &CudaSlice<T>
            &device_buffer2,
            &n,               // Correct type: &usize
            &temp,            // Correct type: &f64
        ),
    )?;
}
```

## FluxNet-Specific GPU Mandate

### ForceProfile (Phase A)

**MUST have GPU device buffers:**
```rust
pub struct ForceProfile {
    // Host data
    pub f_strong: Vec<f32>,
    pub f_weak: Vec<f32>,

    // GPU device mirrors (REQUIRED)
    pub device_f_strong: CudaSlice<f32>,  // ✅ MANDATORY
    pub device_f_weak: CudaSlice<f32>,    // ✅ MANDATORY

    cuda_device: Arc<CudaDevice>,         // ✅ MANDATORY
}
```

**NEVER do this:**
```rust
pub struct ForceProfile {
    pub f_strong: Vec<f32>,
    pub f_weak: Vec<f32>,
    // ❌ NO device buffers = CPU-only = UNACCEPTABLE
}
```

### Phase 2 Thermodynamic Kernel (Phase C)

**MUST modify CUDA kernel:**
```cuda
__global__ void evolve_oscillators_with_conflicts_kernel(
    float* phases,
    int* colors,
    const int* edge_list,
    const int* edge_offsets,
    const float* f_strong,       // ✅ REQUIRED
    const float* f_weak,         // ✅ REQUIRED
    float force_strong_gain,     // ✅ REQUIRED
    float force_weak_gain,       // ✅ REQUIRED
    ...
)
```

**NEVER do this:**
```cuda
// ❌ NEVER skip force parameters
__global__ void evolve_oscillators_kernel(
    float* phases,
    int* colors,
    // Missing force parameters = CPU fallback logic = FORBIDDEN
    ...
)
```

### RL Controller (Phase D)

**MUST sync GPU before telemetry:**
```rust
// Apply force command to GPU
force_profile.to_device()?;  // ✅ Host → Device

// Launch kernel
unsafe { func.launch(cfg, params)?; }

// Synchronize before reading results
cuda_device.synchronize()?;  // ✅ MANDATORY

// Now safe to capture telemetry
let telemetry_after = capture_telemetry()?;
```

**NEVER do this:**
```rust
// ❌ NO synchronization = race condition
// Apply forces
force_profile.to_device()?;

// Launch kernel
unsafe { func.launch(cfg, params)?; }

// ❌ Read telemetry without sync (WRONG)
let telemetry_after = capture_telemetry()?;
```

## Debugging GPU Issues - Proper Workflow

### Step 1: Identify the Error
```bash
cargo build --release --features cuda 2>&1 | tee build.log
```

### Step 2: Locate the Source
- PTX error? → Check `foundation/kernels/*.cu`
- Rust compile error? → Check FFI signatures
- Runtime error? → Add debug prints, check synchronization

### Step 3: Fix the Root Cause
- Don't mask the error
- Don't add fallbacks
- Fix the actual bug

### Step 4: Validate GPU Still Works
```bash
# Rebuild
cargo build --release --features cuda

# Run policy checks
SUB=cuda_gates ./tools/mcp_policy_checks.sh
SUB=stubs ./tools/mcp_policy_checks.sh
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh

# Quick GPU test
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/quick_test.toml --max-minutes 2
```

### Step 5: Verify Performance
```bash
# GPU should be faster than CPU (10-50x)
# Check nvidia-smi during run
watch -n 0.1 nvidia-smi
```

## Policy Enforcement

If you see ANY of these patterns, **immediately reject and fix:**

```rust
// ❌ REJECT
#[cfg(not(feature = "cuda"))]

// ❌ REJECT
if cfg!(feature = "cuda") {
    // GPU path
} else {
    // CPU fallback
}

// ❌ REJECT
// TODO: Implement GPU version later
// For now, use CPU...

// ❌ REJECT
// GPU has a bug, disabling for now
// #[cfg(feature = "cuda")]
```

## The Bottom Line

**GPU acceleration is not optional. It is the entire point of PRISM.**

- World record: **REQUIRES** GPU
- Performance: **REQUIRES** GPU
- FluxNet RL: **REQUIRES** GPU
- Phase 2 thermodynamic: **REQUIRES** GPU
- Reservoir prediction: **REQUIRES** GPU
- Quantum annealing: **REQUIRES** GPU

**If there's a GPU bug, FIX THE BUG. Never disable the GPU.**

## Emergency Escalation

If you encounter a GPU bug you cannot fix:

1. Document the exact error message
2. Show the CUDA kernel code
3. Show the Rust FFI signature
4. Show the launch parameters
5. Ask for help with **fixing the GPU code**

**DO NOT:**
- Suggest disabling CUDA
- Suggest CPU fallback
- Suggest "testing without GPU first"
- Comment out GPU code
- Skip GPU features

## Summary

```
GPU-FIRST. GPU-ONLY. GPU-ALWAYS.

Fix bugs properly. Never disable acceleration.

This is the PRISM way.
```

---

**THIS DOCUMENT IS LAW. NO EXCEPTIONS. 🚨**
