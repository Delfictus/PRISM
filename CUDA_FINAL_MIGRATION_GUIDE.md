# CUDA API Final Migration Guide
## Quick Reference for Completing the Migration

This guide provides copy-paste examples for the remaining 24 files.

---

## Quick Command to Find Files Needing Work

```bash
# Find files with old API:
grep -r "\.load_module\|\.load_function\|\.launch_builder" \
  --include="*.rs" \
  --exclude-dir=target \
  --exclude-dir=".backup" \
  . | cut -d: -f1 | sort -u
```

---

## Pattern 1: load_module → load_ptx

### Find Pattern
```rust
let module = device.load_module(ptx)?;
// or
let module = context.load_module(ptx)?;
```

### Replace With (Template)
```rust
let module = device.load_ptx(
    ptx,
    "YOUR_MODULE_NAME",  // Give it a descriptive name
    &[
        "kernel_function_1",  // List ALL kernel names from the PTX
        "kernel_function_2",
        // ... add all your kernels here
    ]
)?;
```

### How to Find Kernel Names
1. Look for `load_function("KERNEL_NAME")` calls below
2. Or check your .cu/.ptx file for kernel definitions
3. List them all in the array

---

## Pattern 2: load_function → get_func

### Find Pattern
```rust
let kernel = module.load_function("kernel_name")?;
// or
let kernel = Arc::new(module.load_function("kernel_name")?);
```

### Replace With
```rust
let kernel = module.get_func("kernel_name")
    .ok_or_else(|| anyhow!("Failed to load kernel_name"))?;
// or
let kernel = Arc::new(
    module.get_func("kernel_name")
        .ok_or_else(|| anyhow!("Failed to load kernel_name"))?
);
```

---

## Pattern 3: launch_builder → direct launch

### Find Pattern
```rust
let mut launch = stream.launch_builder(&kernel);
launch.arg(&arg1);
launch.arg(&arg2);
launch.arg(&arg3);
unsafe {
    launch.launch(config)?;
}
```

### Replace With
```rust
unsafe {
    kernel.clone().launch(
        config,
        (
            &arg1,
            &arg2,
            &arg3,
        )
    )?;
}
```

**IMPORTANT NOTES:**
- Args must be in a **tuple**: `(arg1, arg2, arg3)`
- Keep `&` for references, omit for values
- Maintain exact order from old code
- Use `kernel.clone()` if kernel is Arc<CudaFunction>

---

## Complete Example Conversion

### Before (Old API)
```rust
pub fn new() -> Result<Self> {
    let device = CudaDevice::new(0)?;

    let ptx_bytes = include_bytes!("kernels.ptx");
    let ptx = Ptx::from_src(std::str::from_utf8(ptx_bytes)?);

    let module = device.load_module(ptx)?;

    let kernel1 = Arc::new(module.load_function("my_kernel")?);
    let kernel2 = Arc::new(module.load_function("other_kernel")?);

    Ok(Self { device, kernel1, kernel2 })
}

pub fn run(&self, data: &[f32]) -> Result<Vec<f32>> {
    let stream = self.device.default_stream();

    let input_gpu = stream.memcpy_stod(data)?;
    let mut output_gpu = stream.alloc_zeros(data.len())?;

    let config = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&self.kernel1);
    launch.arg(&input_gpu);
    launch.arg(&mut output_gpu);
    launch.arg(&(data.len() as i32));
    unsafe {
        launch.launch(config)?;
    }

    stream.synchronize()?;
    let result = stream.memcpy_dtov(&output_gpu)?;
    Ok(result)
}
```

### After (cudarc 0.9)
```rust
pub fn new() -> Result<Self> {
    let device = CudaDevice::new(0)?;

    let ptx_bytes = include_bytes!("kernels.ptx");
    let ptx = Ptx::from_src(std::str::from_utf8(ptx_bytes)?);

    let module = device.load_ptx(
        ptx,
        "my_kernels",
        &["my_kernel", "other_kernel"]
    )?;

    let kernel1 = Arc::new(
        module.get_func("my_kernel")
            .ok_or_else(|| anyhow!("Failed to load my_kernel"))?
    );
    let kernel2 = Arc::new(
        module.get_func("other_kernel")
            .ok_or_else(|| anyhow!("Failed to load other_kernel"))?
    );

    Ok(Self { device, kernel1, kernel2 })
}

pub fn run(&self, data: &[f32]) -> Result<Vec<f32>> {
    let input_gpu = self.device.htod_sync_copy(data)?;
    let mut output_gpu = self.device.alloc_zeros(data.len())?;

    let config = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        self.kernel1.clone().launch(
            config,
            (
                &input_gpu,
                &mut output_gpu,
                data.len() as i32,
            )
        )?;
    }

    self.device.synchronize()?;
    let result = self.device.dtoh_sync_copy(&output_gpu)?;
    Ok(result)
}
```

---

## File-by-File Checklist

Use this to track your progress:

### CMA Files
- [ ] src/cma/transfer_entropy_gpu.rs
- [ ] src/cma/quantum/pimc_gpu.rs
- [ ] foundation/cma/transfer_entropy_gpu.rs
- [ ] foundation/cma/quantum/pimc_gpu.rs

### Quantum Files
- [ ] foundation/quantum/src/gpu_coloring.rs
- [ ] foundation/quantum/src/gpu_k_opt.rs
- [ ] foundation/quantum/src/gpu_tsp.rs

### Active Inference Files
- [ ] foundation/active_inference/gpu.rs
- [ ] foundation/active_inference/gpu_policy_eval.rs
- [ ] foundation/active_inference/gpu_inference.rs

### Statistical Mechanics Files
- [ ] foundation/statistical_mechanics/gpu.rs
- [ ] foundation/statistical_mechanics/gpu_bindings.rs

### Neuromorphic Files
- [ ] foundation/neuromorphic/src/cuda_kernels.rs
- [ ] foundation/neuromorphic/src/gpu_reservoir.rs

### Information Theory Files
- [ ] foundation/information_theory/gpu.rs

### GPU Utility Files
- [ ] foundation/gpu/kernel_executor.rs
- [ ] foundation/gpu_coloring.rs
- [ ] foundation/gpu/optimized_gpu_tensor.rs
- [ ] foundation/gpu/gpu_tensor_optimized.rs

### Orchestration Files
- [ ] foundation/phase6/gpu_tda.rs
- [ ] foundation/orchestration/thermodynamic/optimized_thermodynamic_consensus.rs
- [ ] foundation/orchestration/thermodynamic/gpu_thermodynamic_consensus.rs
- [ ] foundation/orchestration/local_llm/gpu_transformer.rs

### Other Files
- [ ] foundation/quantum_mlir/cuda_kernels.rs

---

## Quick sed Commands (Use with Caution)

These can handle some simple cases but **REVIEW CHANGES MANUALLY**:

```bash
# Replace load_function with get_func (simple cases):
sed -i 's/\.load_function(\("/\.get_func("/' FILE.rs
sed -i 's/\.load_function("/\.get_func("/' FILE.rs

# Add ok_or_else after get_func (manual review needed):
# This is too complex for sed - do manually
```

---

## Testing After Each File

```bash
# Build and check for errors:
cargo build --all-features 2>&1 | grep -A 3 "error.*FILE_YOU_JUST_FIXED"

# If no errors, you're good!
# If errors remain, check:
# 1. Did you list ALL kernels in load_ptx?
# 2. Are launch args in correct order?
# 3. Did you keep & for references?
```

---

## Common Pitfalls

1. **Forgot to list a kernel in load_ptx array**
   - Symptom: "kernel not found" at runtime
   - Fix: Add kernel name to the array

2. **Wrong argument order in launch tuple**
   - Symptom: Wrong results or GPU errors
   - Fix: Match exact order from old code

3. **Missing & on references in launch**
   - Symptom: Type mismatch errors
   - Fix: Keep & for CudaSlice, remove for primitives

4. **Trying to launch without clone()**
   - Symptom: "cannot move out of Arc"
   - Fix: Use `kernel.clone().launch(...)`

---

## Support

If you get stuck on a particular file:
1. Check the working examples in `src/cuda/gpu_coloring.rs`
2. Review the patterns in this guide
3. Check cudarc 0.9 documentation

---

## Progress Tracking

Run this to see how many files still need work:

```bash
echo "Files with old API remaining:"
grep -r "\.load_module\|\.load_function\|\.launch_builder" \
  --include="*.rs" --exclude-dir=target . | \
  cut -d: -f1 | sort -u | wc -l
```

Target: **0 files**

Current as of 2025-10-26: **24 files**

---

Good luck! The patterns are consistent, so once you do 2-3 files, the rest will be quick! 🚀
