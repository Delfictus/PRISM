//! cudarc 0.9 - Correct PTX Loading Pattern
//! Auto-generated integration code for your cudarc version

use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};

/// Load a pre-compiled PTX file (cudarc 0.9 style)
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
