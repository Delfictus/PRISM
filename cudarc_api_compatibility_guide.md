# cudarc API Compatibility Guide

## Key Differences Across Versions

### Pre-compiled PTX Loading (All Versions)

**Common Pattern (0.7+):**
```rust
// Read PTX as bytes, convert to string
let ptx_bytes = std::fs::read("kernel.ptx")?;
let ptx_string = String::from_utf8(ptx_bytes)?;

// Create Ptx object (confusing name - accepts PTX IR, not CUDA C)
let ptx = cudarc::nvrtc::Ptx::from_src(&ptx_string);

// Load into device
let module = device.load_ptx(ptx, "module_name", &["kernel1", "kernel2"])?;
```

**❌ Common Mistakes:**

1. **Reading as String directly:**
   ```rust
   // Don't do this - may fail on some systems
   let ptx = std::fs::read_to_string("kernel.ptx")?;
   ```

2. **Using wrong Ptx constructor:**
   ```rust
   // Ptx::from_file() doesn't exist in most versions
   let ptx = Ptx::from_file("kernel.ptx")?; // ❌
   ```

3. **Forgetting kernel name verification:**
   ```rust
   // Load without checking if kernels exist
   device.load_ptx(ptx, "module", &["wrong_name"])?; // May fail silently
   ```

## Kernel Launch Patterns

### Basic Launch (All Versions)
```rust
let func = module.get_func("kernel_name")
    .ok_or_else(|| anyhow!("Kernel not found"))?;

let cfg = LaunchConfig {
    grid_dim: (blocks, 1, 1),
    block_dim: (threads, 1, 1),
    shared_mem_bytes: 0,
};

unsafe {
    func.launch(cfg, &[&param1, &param2])?;
}

device.synchronize()?;
```

### Parameter Passing

**Scalar Parameters:**
```rust
let n: u32 = 1024;
func.launch(cfg, &[&n])?;
```

**Device Pointers (CudaSlice):**
```rust
let data: CudaSlice<f32> = device.alloc_zeros(1024)?;
func.launch(cfg, &[&data])?;
```

**Raw Pointers (Advanced):**
```rust
let ptr: *const f32 = data.device_ptr();
func.launch(cfg, &[&ptr])?;
```

## Architecture Validation

**Always check compute capability:**
```rust
let (major, minor) = device.compute_capability()?;
let compute_cap = major * 10 + minor;

// For modern features (Tensor Cores, etc.)
assert!(compute_cap >= 75, "Need sm_75+ (Turing)");

// For basic features
assert!(compute_cap >= 60, "Need sm_60+ (Pascal)");
```

## PTX File Requirements

1. **File Format**: ASCII text (UTF-8)
2. **Must contain**: `.version`, `.target`, `.entry` declarations
3. **Architecture**: Must match or be compatible with device sm_XX

**Example PTX Header:**
```ptx
.version 7.5
.target sm_75
.address_size 64

.visible .entry my_kernel(
    .param .u64 .ptr .align 8 input,
    .param .u64 .ptr .align 8 output,
    .param .u32 n
)
{
    // kernel body
}
```

## Common Issues and Solutions

### Issue: "Kernel not found" after load_ptx

**Cause**: Kernel name mismatch between load_ptx() and .entry declaration

**Solution**: Extract exact kernel names from PTX:
```bash
grep "\.visible \.entry" kernel.ptx | sed 's/.*\.entry //' | sed 's/(.*//'
```

### Issue: "Invalid PTX"

**Cause**: PTX compiled for wrong architecture or corrupted

**Solutions**:
1. Verify PTX is text: `file kernel.ptx` should show "ASCII text"
2. Check architecture: `grep "\.target" kernel.ptx`
3. Recompile with correct sm_XX target

### Issue: Launch fails silently

**Cause**: Missing synchronization or parameter type mismatch

**Solution**:
```rust
// Always synchronize to catch errors
device.synchronize()
    .context("Kernel execution failed")?;

// Check CUDA error status
if let Err(e) = device.synchronize() {
    eprintln!("CUDA Error: {:?}", e);
}
```

## Best Practices

1. **Always validate compute capability before loading PTX**
2. **Verify kernel names exist in PTX before load_ptx()**
3. **Use proper error context with anyhow**
4. **Synchronize after kernel launches to catch errors**
5. **Read PTX as bytes, then convert to String**
6. **Store loaded modules for reuse (don't reload every launch)**

## Testing PTX Loading

```rust
#[test]
fn test_ptx_kernels() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Load PTX
    let module = load_ptx_module(
        &device,
        "path/to/kernel.ptx",
        "module_name",
        &["kernel1", "kernel2"]
    )?;
    
    // Verify kernels exist
    assert!(module.get_func("kernel1").is_some());
    assert!(module.get_func("kernel2").is_some());
    
    // Test launch with dummy data
    let input: CudaSlice<f32> = device.alloc_zeros(1024)?;
    let output: CudaSlice<f32> = device.alloc_zeros(1024)?;
    
    let func = module.get_func("kernel1").unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    
    unsafe {
        func.launch(cfg, &[&input, &output])?;
    }
    
    device.synchronize()?;
    
    Ok(())
}
```
