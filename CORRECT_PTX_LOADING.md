# CORRECT PTX Loading for cudarc 0.9

## ⚠️ Critical Understanding

**PTX files are already compiled** - they are NOT CUDA C source code!
- `.cu` files = CUDA C source (needs compilation)
- `.ptx` files = Already compiled PTX assembly (ready to load)

## ❌ WRONG Approaches

```rust
// WRONG - from_src is misleadingly named but this pattern is seen in codebase
let ptx_string = std::fs::read_to_string(ptx_path)?;
let ptx = Ptx::from_src(&ptx_string);  // Confusing but sometimes works

// WRONG - This is for CUDA C source, not PTX
let ptx = Ptx::compile_ptx(cuda_source)?;  // For .cu files only!
```

## ✅ CORRECT Approach for Pre-Compiled PTX (cudarc 0.9)

```rust
use cudarc::driver::{CudaDevice, CudaModule};
use std::ffi::CString;

// Method 1: Direct PTX loading (Most Correct)
let ptx_cstring = CString::new(std::fs::read_to_string(ptx_path)?)?;
let module = unsafe {
    cudarc::driver::sys::cuModuleLoadData(ptx_cstring.as_ptr() as *const _)?
};

// Method 2: Using load_ptx (if available in your cudarc version)
let ptx_bytes = std::fs::read(ptx_path)?;
let ptx_string = String::from_utf8(ptx_bytes)?;
let ptx = cudarc::nvrtc::Ptx {
    ptx: ptx_string  // Direct field assignment
};
let module = device.load_ptx(ptx, "module_name", &["kernel1", "kernel2"])?;

// Method 3: If Ptx::from_file exists in your version
let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
let module = device.load_ptx(ptx, "module_name", &["kernel1", "kernel2"])?;
```

## 🔍 How to Verify Kernel Names

```bash
# Extract actual kernel names from PTX
grep "\.entry" your_file.ptx | sed 's/.*\.entry //' | sed 's/(.*//'
```

## 📝 Complete Example with Validation

```rust
use cudarc::driver::{CudaDevice, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use anyhow::{Result, Context};
use std::sync::Arc;

pub fn load_neuromorphic_kernels(device: Arc<CudaDevice>) -> Result<CudaModule> {
    let ptx_path = "foundation/kernels/ptx/neuromorphic_gemv.ptx";

    // Verify file exists
    if !std::path::Path::new(ptx_path).exists() {
        return Err(anyhow::anyhow!("PTX file not found: {}", ptx_path));
    }

    // Check compute capability
    let props = device.props()?;
    let compute_cap = props.major * 10 + props.minor;
    if compute_cap < 75 {  // Require Turing or newer
        return Err(anyhow::anyhow!(
            "GPU compute capability {}.{} too old (need 7.5+)",
            props.major, props.minor
        ));
    }

    // Read PTX file (it's ASCII text)
    let ptx_string = std::fs::read_to_string(ptx_path)
        .with_context(|| format!("Failed to read PTX from {}", ptx_path))?;

    // Create Ptx struct - the field is just called 'ptx'
    let ptx = Ptx { ptx: ptx_string };

    // Load with actual kernel names from the PTX file
    // These MUST match the .entry names in the PTX!
    let module = device.load_ptx(ptx, "neuromorphic", &[
        "matvec_input_kernel",      // Verified from PTX
        "matvec_reservoir_kernel",  // Verified from PTX
        "leaky_integration_kernel"  // Verified from PTX
    ])?;

    // Verify kernels loaded correctly
    let _test = module.get_func("matvec_input_kernel")
        .context("Failed to get matvec_input_kernel")?;

    println!("✅ Neuromorphic kernels loaded successfully");
    Ok(module)
}
```

## 🚨 Common Pitfalls

1. **Using wrong kernel names** - Must match EXACTLY what's in PTX
2. **Treating PTX like CUDA source** - PTX is already compiled
3. **Missing architecture flags** - PTX may be compiled for specific GPU
4. **Not checking if PTX exists** - Leads to cryptic errors

## 🔧 Fixing the Codebase

All instances of:
```rust
let ptx = Ptx::from_src(&ptx_string);  // Misleading but sometimes works
```

Should potentially be:
```rust
let ptx = Ptx { ptx: ptx_string };  // Direct struct construction
```

Or use the proper loading method for your cudarc version.

## 📊 Verification Commands

```bash
# Check PTX is valid
nvdisasm foundation/kernels/ptx/neuromorphic_gemv.ptx | head

# List all kernels in PTX
grep "\.entry" foundation/kernels/ptx/*.ptx

# Check PTX architecture
grep "\.target" foundation/kernels/ptx/neuromorphic_gemv.ptx
```

## Summary

The key issue is that cudarc's API is confusing:
- `Ptx::from_src()` sounds like it's for source but works with PTX strings
- `compile_ptx()` is for actual CUDA C source
- PTX files are **already compiled** and just need to be loaded

Always verify kernel names match exactly what's in the PTX file!