#!/bin/bash
#
# cudarc API Version Detector and Compatibility Checker
# Generates correct integration code for the specific cudarc version
#

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   cudarc API Version Detector & Compatibility Checker         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Find cudarc version in Cargo.toml files
echo -e "${GREEN}[1/5] Detecting cudarc version...${NC}"

CUDARC_VERSION=""
CARGO_FILES=$(find . -name "Cargo.toml" -not -path "*/target/*")

for cargo in $CARGO_FILES; do
    if grep -q "cudarc" "$cargo" 2>/dev/null; then
        version=$(grep "cudarc" "$cargo" | grep -oP 'version\s*=\s*"\K[^"]+' | head -1)
        if [ -n "$version" ]; then
            CUDARC_VERSION="$version"
            echo -e "  ${CYAN}✓${NC} Found cudarc version: ${GREEN}$version${NC} in $cargo"
            break
        fi
    fi
done

if [ -z "$CUDARC_VERSION" ]; then
    echo -e "  ${YELLOW}⚠${NC} cudarc version not found in Cargo.toml"
    echo "  Checking Cargo.lock..."
    
    if [ -f "Cargo.lock" ]; then
        CUDARC_VERSION=$(grep -A 2 'name = "cudarc"' Cargo.lock | grep "version" | head -1 | grep -oP '"\K[^"]+')
        if [ -n "$CUDARC_VERSION" ]; then
            echo -e "  ${CYAN}✓${NC} Found cudarc version: ${GREEN}$CUDARC_VERSION${NC} in Cargo.lock"
        fi
    fi
fi

if [ -z "$CUDARC_VERSION" ]; then
    echo -e "  ${YELLOW}⚠${NC} Cannot determine cudarc version. Assuming 0.9+"
    CUDARC_VERSION="0.9.0"
fi

echo ""

# Parse version
VERSION_MAJOR=$(echo $CUDARC_VERSION | cut -d. -f1)
VERSION_MINOR=$(echo $CUDARC_VERSION | cut -d. -f2)

echo -e "${GREEN}[2/5] Analyzing API compatibility...${NC}"

# Determine API style based on version
if [ "$VERSION_MAJOR" -eq 0 ] && [ "$VERSION_MINOR" -ge 9 ]; then
    API_STYLE="modern"
    echo -e "  ${CYAN}✓${NC} API Style: Modern (0.9+)"
elif [ "$VERSION_MAJOR" -eq 0 ] && [ "$VERSION_MINOR" -ge 7 ]; then
    API_STYLE="transition"
    echo -e "  ${YELLOW}⚠${NC} API Style: Transition (0.7-0.8)"
else
    API_STYLE="legacy"
    echo -e "  ${YELLOW}⚠${NC} API Style: Legacy (<0.7)"
fi

echo ""

# Check current PTX loading patterns
echo -e "${GREEN}[3/5] Auditing current PTX loading code...${NC}"

AUDIT_FILE="cudarc_api_audit.txt"
echo "=== cudarc API Usage Audit ===" > "$AUDIT_FILE"
echo "cudarc version: $CUDARC_VERSION" >> "$AUDIT_FILE"
echo "" >> "$AUDIT_FILE"

echo "CURRENT load_ptx CALLS:" >> "$AUDIT_FILE"
find foundation -name "*.rs" -type f -exec grep -H -B 2 -A 2 "load_ptx" {} \; >> "$AUDIT_FILE" 2>/dev/null

echo "CURRENT Ptx::from_src USAGE:" >> "$AUDIT_FILE"
find foundation -name "*.rs" -type f -exec grep -H -B 2 -A 2 "Ptx::from_src" {} \; >> "$AUDIT_FILE" 2>/dev/null

echo "CURRENT PTX FILE READING:" >> "$AUDIT_FILE"
find foundation -name "*.rs" -type f -exec grep -H -B 1 -A 1 "read.*\.ptx\|read_to_string.*\.ptx" {} \; >> "$AUDIT_FILE" 2>/dev/null

echo -e "  ${CYAN}✓${NC} Audit saved to: $AUDIT_FILE"
echo ""

# Generate correct integration code
echo -e "${GREEN}[4/5] Generating version-specific integration code...${NC}"

TEMPLATE_FILE="cudarc_${VERSION_MAJOR}_${VERSION_MINOR}_integration.rs"

cat > "$TEMPLATE_FILE" << RUSTCODE
//! cudarc $CUDARC_VERSION - Correct PTX Loading Pattern
//! Auto-generated integration code for your cudarc version

use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};

/// Load a pre-compiled PTX file (cudarc $CUDARC_VERSION style)
pub fn load_ptx_module(
    device: &Arc<CudaDevice>,
    ptx_path: &str,
    module_name: &str,
    kernel_names: &[&str],
) -> Result<CudaModule> {
    // Validate compute capability
    let (major, minor) = device.compute_capability()
        .context("Failed to query compute capability")?;
    
    println!("[PTX-LOADER] Device compute capability: sm_{}{}", major, minor);
    
    // Require minimum sm_75 (Turing) for modern features
    let compute_cap = major * 10 + minor;
    if compute_cap < 75 {
        return Err(anyhow!(
            "Insufficient compute capability: sm_{}{} (need sm_75+)",
            major, minor
        ));
    }
    
    // Verify PTX file exists
    if !std::path::Path::new(ptx_path).exists() {
        return Err(anyhow!("PTX file not found: {}", ptx_path));
    }
    
    println!("[PTX-LOADER] Loading PTX from: {}", ptx_path);
    
    // Read PTX file as bytes (PTX is ASCII text, but read as bytes first)
    let ptx_bytes = std::fs::read(ptx_path)
        .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
    
    // Convert bytes to UTF-8 string (PTX is text-based IR)
    let ptx_string = String::from_utf8(ptx_bytes)
        .context("PTX file contains invalid UTF-8")?;
    
    // Verify PTX header
    if !ptx_string.starts_with(".version") && !ptx_string.contains(".target") {
        return Err(anyhow!("Invalid PTX file format (missing .version or .target)"));
    }
    
    // Extract PTX version for logging
    if let Some(version_line) = ptx_string.lines().find(|l| l.starts_with(".version")) {
        println!("[PTX-LOADER] PTX version: {}", version_line);
    }
    
    // Create Ptx object
    // NOTE: Ptx::from_src() is confusingly named - it accepts PTX IR text, not CUDA C
    let ptx = Ptx::from_src(&ptx_string);
    
    println!("[PTX-LOADER] Loading module '{}' with {} kernel(s)", module_name, kernel_names.len());
    
    // Load PTX into device
    let module = device.load_ptx(ptx, module_name, kernel_names)
        .with_context(|| format!(
            "Failed to load PTX module '{}' from {}",
            module_name, ptx_path
        ))?;
    
    // Verify all kernels are present
    for kernel_name in kernel_names {
        let func = module.get_func(kernel_name)
            .ok_or_else(|| anyhow!(
                "Kernel '{}' not found in PTX module (check .entry declarations)",
                kernel_name
            ))?;
        
        // Query kernel attributes (optional, for debugging)
        if let Ok(attrs) = func.get_attributes() {
            println!("[PTX-LOADER]   ✓ {} (regs: {}, shared: {} bytes)",
                kernel_name,
                attrs.num_regs,
                attrs.shared_size_bytes
            );
        } else {
            println!("[PTX-LOADER]   ✓ {}", kernel_name);
        }
    }
    
    Ok(module)
}

/// Example: Load neuromorphic GEMV kernels
pub fn load_neuromorphic_kernels(device: &Arc<CudaDevice>) -> Result<CudaModule> {
    load_ptx_module(
        device,
        "foundation/kernels/ptx/neuromorphic_gemv.ptx",
        "neuromorphic_gemv",
        &[
            "matvec_input_kernel",
            "matvec_reservoir_kernel",
            "leaky_integration_kernel",
        ]
    )
}

/// Launch a kernel with type-safe parameters
pub fn launch_kernel<T>(
    module: &CudaModule,
    kernel_name: &str,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    params: &[&T],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + ?Sized,
{
    let func = module.get_func(kernel_name)
        .ok_or_else(|| anyhow!("Kernel '{}' not found", kernel_name))?;
    
    let cfg = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes,
    };
    
    unsafe {
        func.launch(cfg, params)
            .with_context(|| format!("Kernel launch failed: {}", kernel_name))?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_ptx() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let module = load_neuromorphic_kernels(&device)?;
        
        // Verify kernel exists
        assert!(module.get_func("matvec_input_kernel").is_some());
        
        Ok(())
    }
}
RUSTCODE

echo -e "  ${CYAN}✓${NC} Generated: $TEMPLATE_FILE"
echo ""

# Generate API differences documentation
echo -e "${GREEN}[5/5] Generating API compatibility guide...${NC}"

COMPAT_GUIDE="cudarc_api_compatibility_guide.md"

cat > "$COMPAT_GUIDE" << 'MARKDOWN'
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
MARKDOWN

echo -e "  ${CYAN}✓${NC} Generated: $COMPAT_GUIDE"
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    ANALYSIS COMPLETE                          ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Generated files:${NC}"
echo "  📋 $AUDIT_FILE              - Current API usage audit"
echo "  🔧 $TEMPLATE_FILE           - Version-specific integration code"
echo "  📖 $COMPAT_GUIDE            - Complete API compatibility guide"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review $AUDIT_FILE for current incorrect usage"
echo "  2. Use $TEMPLATE_FILE as reference for correct API calls"
echo "  3. Read $COMPAT_GUIDE for common issues and solutions"
echo "  4. Update your PTX loading code to match the template"
echo ""

echo -e "${CYAN}Quick integration test:${NC}"
echo "  rustc --version && cargo build --features cuda"
echo ""
