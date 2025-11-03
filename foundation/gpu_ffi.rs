//! FFI Interface to GPU Runtime Library
//!
//! This module provides direct access to compiled GPU kernels
//! through a shared library, bypassing cudarc API complexity.

use std::sync::Once;
use anyhow::{Result, anyhow};
use ndarray::Array1;

// FFI declarations for GPU runtime library
#[link(name = "gpu_runtime", kind = "dylib")]
extern "C" {
    fn launch_transfer_entropy(source: *const f64, target: *const f64, n: i32) -> f32;
    fn launch_thermodynamic(phases: *mut f64, velocities: *mut f64, n_osc: i32, n_steps: i32);
    fn gpu_available() -> i32;
}

static INIT: Once = Once::new();
static mut GPU_READY: bool = false;

/// Initialize GPU FFI and check availability
pub fn initialize_gpu_ffi() {
    INIT.call_once(|| {
        unsafe {
            GPU_READY = gpu_available() != 0;
            if GPU_READY {
                println!("[GPU-FFI] ✅ GPU runtime library loaded successfully!");
            } else {
                println!("[GPU-FFI] ❌ GPU not available via runtime library");
            }
        }
    });
}

/// Check if GPU is available through FFI
pub fn is_gpu_available() -> bool {
    initialize_gpu_ffi();
    unsafe { GPU_READY }
}

/// Launch Transfer Entropy computation on GPU via FFI
pub fn compute_transfer_entropy_gpu(source: &Array1<f64>, target: &Array1<f64>) -> Result<f64> {
    if !is_gpu_available() {
        return Err(anyhow!("GPU not available"));
    }

    let n = source.len() as i32;
    if n != target.len() as i32 {
        return Err(anyhow!("Source and target must have same length"));
    }

    unsafe {
        let result = launch_transfer_entropy(
            source.as_ptr(),
            target.as_ptr(),
            n
        );
        Ok(result as f64)
    }
}

/// Launch Thermodynamic evolution on GPU via FFI
pub fn evolve_thermodynamic_gpu(
    phases: &mut Array1<f64>,
    velocities: &mut Array1<f64>,
    n_steps: usize
) -> Result<()> {
    if !is_gpu_available() {
        return Err(anyhow!("GPU not available"));
    }

    let n_osc = phases.len() as i32;
    if n_osc != velocities.len() as i32 {
        return Err(anyhow!("Phases and velocities must have same length"));
    }

    unsafe {
        launch_thermodynamic(
            phases.as_mut_ptr(),
            velocities.as_mut_ptr(),
            n_osc,
            n_steps as i32
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        initialize_gpu_ffi();
        println!("GPU available via FFI: {}", is_gpu_available());
    }

    #[test]
    fn test_transfer_entropy_gpu() {
        if !is_gpu_available() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        let source = Array1::linspace(0.0, 10.0, 1000);
        let target = source.mapv(|x| x.sin());

        match compute_transfer_entropy_gpu(&source, &target) {
            Ok(te) => println!("Transfer Entropy (GPU): {}", te),
            Err(e) => println!("GPU computation failed: {}", e),
        }
    }
}