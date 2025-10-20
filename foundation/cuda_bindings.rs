//! FFI Bindings for CUDA Kernels
//!
//! This module provides safe Rust wrappers around the CUDA kernels
//! for quantum evolution and double-double arithmetic.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use parking_lot::Mutex;
use num_complex::Complex64;
use ndarray::{Array1, Array2};
use anyhow::{Result, anyhow};

// ============================================================================
// CUDA Runtime FFI
// ============================================================================

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, size: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: i32) -> i32;
}

// Memory copy directions
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

#[repr(C)]
struct CudaDeviceProp {
    name: [i8; 256],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    warp_size: i32,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    clock_rate: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
    major: i32,
    minor: i32,
    // ... other fields omitted for brevity
}

// ============================================================================
// Quantum Evolution Kernel FFI
// ============================================================================
// NOTE: Object files linked directly by build.rs - no library needed
// ============================================================================

// Object files linked by build.rs:
//   quantum_evolution.o
//   quantum_mlir.o
// No #[link] attribute needed - prevents "cannot find -lquantum_kernels" error

extern "C" {
    // Initialize quantum evolution system
    fn quantum_evolution_init(system_size: i32) -> *mut c_void;

    // Evolve quantum state
    fn evolve_quantum_state(
        h_real: *const f64,
        h_imag: *const f64,
        psi_real: *mut f64,
        psi_imag: *mut f64,
        time: f64,
        dim: i32,
    ) -> i32;

    // Cleanup
    fn quantum_evolution_cleanup(handle: *mut c_void);

    // High-precision evolution with double-double
    fn evolve_quantum_state_dd(
        h_real_hi: *const f64,
        h_real_lo: *const f64,
        h_imag_hi: *const f64,
        h_imag_lo: *const f64,
        psi_real_hi: *mut f64,
        psi_real_lo: *mut f64,
        psi_imag_hi: *mut f64,
        psi_imag_lo: *mut f64,
        time: f64,
        dim: i32,
    ) -> i32;

    // Build Hamiltonians
    fn build_tight_binding_hamiltonian_gpu(
        edges: *const i32,
        weights: *const f64,
        num_vertices: i32,
        num_edges: i32,
        hopping_strength: f64,
    ) -> *mut c_void;

    fn build_ising_hamiltonian_gpu(
        j_matrix: *const f64,
        h_field: *const f64,
        n_spins: i32,
    ) -> *mut c_void;

    // VQE expectation value
    fn vqe_expectation_value_gpu(
        state_real: *const f64,
        state_imag: *const f64,
        hamiltonian_handle: *mut c_void,
        dim: i32,
    ) -> f64;

    // Measurement
    fn measure_probability_distribution_gpu(
        state_real: *const f64,
        state_imag: *const f64,
        probabilities: *mut f64,
        dim: i32,
    ) -> i32;
}

// ============================================================================
// Double-Double Arithmetic FFI
// ============================================================================
// NOTE: Object file linked directly by build.rs - no library needed
// ============================================================================

// Object file linked by build.rs:
//   double_double.o
// No #[link] attribute needed - prevents "cannot find -ldd_kernels" error

extern "C" {
    // Test double-double arithmetic
    fn run_dd_test();

    // Array operations
    fn dd_array_add_gpu(
        result_hi: *mut f64,
        result_lo: *mut f64,
        a_hi: *const f64,
        a_lo: *const f64,
        b_hi: *const f64,
        b_lo: *const f64,
        n: i32,
    ) -> i32;

    fn dd_matrix_vector_mul_gpu(
        y_real_hi: *mut f64,
        y_real_lo: *mut f64,
        y_imag_hi: *mut f64,
        y_imag_lo: *mut f64,
        a_real_hi: *const f64,
        a_real_lo: *const f64,
        a_imag_hi: *const f64,
        a_imag_lo: *const f64,
        x_real_hi: *const f64,
        x_real_lo: *const f64,
        x_imag_hi: *const f64,
        x_imag_lo: *const f64,
        n: i32,
    ) -> i32;

    fn dd_deterministic_reduce_gpu(
        output_hi: *mut f64,
        output_lo: *mut f64,
        input_hi: *const f64,
        input_lo: *const f64,
        n: i32,
    ) -> i32;
}

// ============================================================================
// Safe Rust Wrappers
// ============================================================================

/// Check CUDA error and convert to Result
fn check_cuda_error(error: i32) -> Result<()> {
    if error != 0 {
        unsafe {
            let err_str = std::ffi::CStr::from_ptr(cudaGetErrorString(error));
            Err(anyhow!("CUDA error {}: {:?}", error, err_str))
        }
    } else {
        Ok(())
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub memory_gb: f64,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

impl GpuDevice {
    /// Get information about available GPU devices
    pub fn enumerate() -> Result<Vec<GpuDevice>> {
        let mut count = 0;
        unsafe {
            check_cuda_error(cudaGetDeviceCount(&mut count))?;
        }

        let mut devices = Vec::new();
        for i in 0..count {
            let mut prop: CudaDeviceProp = unsafe { std::mem::zeroed() };
            unsafe {
                check_cuda_error(cudaGetDeviceProperties(&mut prop, i))?;
            }

            let name = unsafe {
                std::ffi::CStr::from_ptr(prop.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };

            devices.push(GpuDevice {
                name,
                compute_capability: (prop.major, prop.minor),
                memory_gb: prop.total_global_mem as f64 / (1024.0 * 1024.0 * 1024.0),
                max_threads_per_block: prop.max_threads_per_block,
                warp_size: prop.warp_size,
            });
        }

        Ok(devices)
    }
}

/// GPU memory buffer
pub struct GpuBuffer<T> {
    ptr: *mut c_void,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    /// Allocate GPU memory
    pub fn new(size: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let byte_size = size * std::mem::size_of::<T>();

        unsafe {
            check_cuda_error(cudaMalloc(&mut ptr, byte_size))?;
        }

        Ok(GpuBuffer {
            ptr,
            size,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.size {
            return Err(anyhow!("Size mismatch: {} vs {}", data.len(), self.size));
        }

        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            check_cuda_error(cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ))?;
        }

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.size {
            return Err(anyhow!("Size mismatch: {} vs {}", data.len(), self.size));
        }

        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            check_cuda_error(cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ))?;
        }

        Ok(())
    }

    /// Get raw pointer for FFI
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get mutable raw pointer for FFI
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as *mut T
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

/// Quantum evolution on GPU
pub struct QuantumEvolutionGpu {
    handle: *mut c_void,
    dimension: usize,
}

impl QuantumEvolutionGpu {
    /// Create new quantum evolution system
    pub fn new(dimension: usize) -> Result<Self> {
        let handle = unsafe { quantum_evolution_init(dimension as i32) };

        if handle.is_null() {
            return Err(anyhow!("Failed to initialize quantum evolution"));
        }

        Ok(QuantumEvolutionGpu { handle, dimension })
    }

    /// Evolve quantum state with standard precision
    pub fn evolve(&self, hamiltonian: &Array2<Complex64>, state: &Array1<Complex64>, time: f64) -> Result<Array1<Complex64>> {
        if hamiltonian.nrows() != self.dimension || hamiltonian.ncols() != self.dimension {
            return Err(anyhow!("Hamiltonian dimension mismatch"));
        }

        if state.len() != self.dimension {
            return Err(anyhow!("State dimension mismatch"));
        }

        // Separate real and imaginary parts
        let h_real: Vec<f64> = hamiltonian.iter().map(|c| c.re).collect();
        let h_imag: Vec<f64> = hamiltonian.iter().map(|c| c.im).collect();
        let mut psi_real: Vec<f64> = state.iter().map(|c| c.re).collect();
        let mut psi_imag: Vec<f64> = state.iter().map(|c| c.im).collect();

        // Call GPU kernel
        let result = unsafe {
            evolve_quantum_state(
                h_real.as_ptr(),
                h_imag.as_ptr(),
                psi_real.as_mut_ptr(),
                psi_imag.as_mut_ptr(),
                time,
                self.dimension as i32,
            )
        };

        check_cuda_error(result)?;

        // Combine back into complex array
        let evolved: Vec<Complex64> = psi_real
            .iter()
            .zip(psi_imag.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();

        Ok(Array1::from(evolved))
    }

    /// Evolve with double-double precision
    pub fn evolve_dd(&self, hamiltonian: &Array2<Complex64>, state: &Array1<Complex64>, time: f64) -> Result<Array1<Complex64>> {
        // Convert to double-double representation
        let h_real_hi: Vec<f64> = hamiltonian.iter().map(|c| c.re).collect();
        let h_real_lo = vec![0.0; hamiltonian.len()];
        let h_imag_hi: Vec<f64> = hamiltonian.iter().map(|c| c.im).collect();
        let h_imag_lo = vec![0.0; hamiltonian.len()];

        let mut psi_real_hi: Vec<f64> = state.iter().map(|c| c.re).collect();
        let mut psi_real_lo = vec![0.0; state.len()];
        let mut psi_imag_hi: Vec<f64> = state.iter().map(|c| c.im).collect();
        let mut psi_imag_lo = vec![0.0; state.len()];

        // Call high-precision GPU kernel
        let result = unsafe {
            evolve_quantum_state_dd(
                h_real_hi.as_ptr(),
                h_real_lo.as_ptr(),
                h_imag_hi.as_ptr(),
                h_imag_lo.as_ptr(),
                psi_real_hi.as_mut_ptr(),
                psi_real_lo.as_mut_ptr(),
                psi_imag_hi.as_mut_ptr(),
                psi_imag_lo.as_mut_ptr(),
                time,
                self.dimension as i32,
            )
        };

        check_cuda_error(result)?;

        // Combine high and low parts (for now just use high part)
        let evolved: Vec<Complex64> = psi_real_hi
            .iter()
            .zip(psi_imag_hi.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();

        Ok(Array1::from(evolved))
    }

    /// Compute VQE expectation value
    pub fn vqe_expectation(&self, state: &Array1<Complex64>, hamiltonian_handle: *mut c_void) -> Result<f64> {
        let state_real: Vec<f64> = state.iter().map(|c| c.re).collect();
        let state_imag: Vec<f64> = state.iter().map(|c| c.im).collect();

        let expectation = unsafe {
            vqe_expectation_value_gpu(
                state_real.as_ptr(),
                state_imag.as_ptr(),
                hamiltonian_handle,
                self.dimension as i32,
            )
        };

        Ok(expectation)
    }

    /// Measure probability distribution
    pub fn measure(&self, state: &Array1<Complex64>) -> Result<Vec<f64>> {
        let state_real: Vec<f64> = state.iter().map(|c| c.re).collect();
        let state_imag: Vec<f64> = state.iter().map(|c| c.im).collect();
        let mut probabilities = vec![0.0; self.dimension];

        let result = unsafe {
            measure_probability_distribution_gpu(
                state_real.as_ptr(),
                state_imag.as_ptr(),
                probabilities.as_mut_ptr(),
                self.dimension as i32,
            )
        };

        check_cuda_error(result)?;
        Ok(probabilities)
    }
}

impl Drop for QuantumEvolutionGpu {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                quantum_evolution_cleanup(self.handle);
            }
        }
    }
}

/// Hamiltonian builder for GPU
pub struct HamiltonianBuilder;

impl HamiltonianBuilder {
    /// Build tight-binding Hamiltonian on GPU
    pub fn tight_binding(edges: &[(usize, usize)], weights: &[f64], num_vertices: usize, hopping: f64) -> Result<*mut c_void> {
        let edge_array: Vec<i32> = edges
            .iter()
            .flat_map(|(i, j)| vec![*i as i32, *j as i32])
            .collect();

        let handle = unsafe {
            build_tight_binding_hamiltonian_gpu(
                edge_array.as_ptr(),
                weights.as_ptr(),
                num_vertices as i32,
                edges.len() as i32,
                hopping,
            )
        };

        if handle.is_null() {
            return Err(anyhow!("Failed to build Hamiltonian on GPU"));
        }

        Ok(handle)
    }

    /// Build Ising Hamiltonian on GPU
    pub fn ising(j_matrix: &Array2<f64>, h_field: &Array1<f64>) -> Result<*mut c_void> {
        let n = h_field.len();

        let handle = unsafe {
            build_ising_hamiltonian_gpu(
                j_matrix.as_ptr(),
                h_field.as_ptr(),
                n as i32,
            )
        };

        if handle.is_null() {
            return Err(anyhow!("Failed to build Ising Hamiltonian on GPU"));
        }

        Ok(handle)
    }
}

/// Double-double arithmetic operations on GPU
pub struct DoublDoubleGpu;

impl DoublDoubleGpu {
    /// Test double-double arithmetic
    pub fn test() {
        unsafe {
            run_dd_test();
        }
    }

    /// Add arrays with double-double precision
    pub fn add_arrays(a: &[(f64, f64)], b: &[(f64, f64)]) -> Result<Vec<(f64, f64)>> {
        if a.len() != b.len() {
            return Err(anyhow!("Array size mismatch"));
        }

        let n = a.len();
        let a_hi: Vec<f64> = a.iter().map(|(hi, _)| *hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|(_, lo)| *lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|(hi, _)| *hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|(_, lo)| *lo).collect();

        let mut result_hi = vec![0.0; n];
        let mut result_lo = vec![0.0; n];

        let error = unsafe {
            dd_array_add_gpu(
                result_hi.as_mut_ptr(),
                result_lo.as_mut_ptr(),
                a_hi.as_ptr(),
                a_lo.as_ptr(),
                b_hi.as_ptr(),
                b_lo.as_ptr(),
                n as i32,
            )
        };

        check_cuda_error(error)?;

        Ok(result_hi
            .iter()
            .zip(result_lo.iter())
            .map(|(&hi, &lo)| (hi, lo))
            .collect())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_enumeration() {
        let devices = GpuDevice::enumerate().unwrap();
        println!("Found {} GPU devices:", devices.len());

        for (i, device) in devices.iter().enumerate() {
            println!("GPU {}: {}", i, device.name);
            println!("  Compute capability: {}.{}", device.compute_capability.0, device.compute_capability.1);
            println!("  Memory: {:.2} GB", device.memory_gb);
        }

        assert!(!devices.is_empty(), "No GPU devices found");
    }

    #[test]
    fn test_gpu_buffer() {
        let mut buffer = GpuBuffer::<f64>::new(1000).unwrap();
        let data = vec![1.0; 1000];

        buffer.copy_from_host(&data).unwrap();

        let mut result = vec![0.0; 1000];
        buffer.copy_to_host(&mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_double_double() {
        DoublDoubleGpu::test();
    }

    #[test]
    fn test_quantum_evolution() {
        let dim = 10;
        let evolution = QuantumEvolutionGpu::new(dim).unwrap();

        let h = Array2::eye(dim);
        let psi = Array1::zeros(dim);

        let result = evolution.evolve(&h, &psi, 1.0).unwrap();
        assert_eq!(result.len(), dim);
    }
}