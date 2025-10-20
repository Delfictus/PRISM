//! GPU Memory Management for Quantum MLIR
//!
//! Handles GPU memory allocation and data transfer using cudarc

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use cudarc::driver::result::DriverError;
use std::sync::Arc;
use anyhow::Result;

use super::cuda_kernels::CudaComplex;
use super::Complex64;

/// GPU memory manager for quantum states
pub struct GpuMemoryManager {
    pub context: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl GpuMemoryManager {
    /// Create new GPU memory manager with shared CUDA context
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (CudaContext::new already returns Arc)
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        let stream = context.new_stream()
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA stream: {}", e))?;

        Ok(Self {
            context,
            stream,
        })
    }

    /// Allocate GPU memory for quantum state
    pub fn allocate_state(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        let stream = self.context.default_stream();

        // Initialize to |00...0> state
        let mut init = vec![CudaComplex::zero(); dimension];
        init[0] = CudaComplex::one();

        // Upload initial state directly
        let state = stream.memcpy_stod(&init)
            .map_err(|e| anyhow::anyhow!("Failed to allocate and initialize quantum state on GPU: {}", e))?;

        Ok(state)
    }

    /// Allocate GPU memory for Hamiltonian matrix
    pub fn allocate_hamiltonian(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        let size = dimension * dimension;
        let stream = self.context.default_stream();
        stream.alloc_zeros::<CudaComplex>(size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate GPU memory for Hamiltonian: {}", e))
    }

    /// Copy quantum state from host to device
    pub fn upload_state(&self, host_state: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_state: Vec<CudaComplex> = host_state.iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        let stream = self.context.default_stream();
        stream.memcpy_stod(&cuda_state)
            .map_err(|e| anyhow::anyhow!("Failed to upload quantum state to GPU: {}", e))
    }

    /// Copy quantum state from device to host
    pub fn download_state(&self, device_state: &CudaSlice<CudaComplex>) -> Result<Vec<Complex64>> {
        let stream = self.context.default_stream();
        let cuda_state = stream.memcpy_dtov(device_state)
            .map_err(|e| anyhow::anyhow!("Failed to download quantum state from GPU: {}", e))?;

        Ok(cuda_state.into_iter()
            .map(|c| Complex64 { real: c.real, imag: c.imag })
            .collect())
    }

    /// Upload Hamiltonian matrix to GPU
    pub fn upload_hamiltonian(&self, hamiltonian: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_ham: Vec<CudaComplex> = hamiltonian.iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        let stream = self.context.default_stream();
        stream.memcpy_stod(&cuda_ham)
            .map_err(|e| anyhow::anyhow!("Failed to upload Hamiltonian to GPU: {}", e))
    }

    /// Allocate GPU memory for measurement probabilities
    pub fn allocate_probabilities(&self, dimension: usize) -> Result<CudaSlice<f64>> {
        let stream = self.context.default_stream();
        stream.alloc_zeros::<f64>(dimension)
            .map_err(|e| anyhow::anyhow!("Failed to allocate GPU memory for probabilities: {}", e))
    }

    /// Download probabilities from GPU
    pub fn download_probabilities(&self, device_probs: &CudaSlice<f64>) -> Result<Vec<f64>> {
        let stream = self.context.default_stream();
        stream.memcpy_dtov(device_probs)
            .map_err(|e| anyhow::anyhow!("Failed to download probabilities from GPU: {}", e))
    }

    /// Get raw pointer to GPU memory (for FFI)
    pub fn get_ptr<T>(&self, slice: &CudaSlice<T>) -> *mut T {
        slice.device_ptr(&self.stream).0 as *mut T
    }

    /// Get const pointer to GPU memory (for FFI)
    pub fn get_const_ptr<T>(&self, slice: &CudaSlice<T>) -> *const T {
        slice.device_ptr(&self.stream).0 as *const T
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<()> {
        self.context.default_stream().synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to synchronize GPU: {}", e))
    }

    /// Get device properties
    pub fn get_device_info(&self) -> String {
        format!("CUDA Device 0")  // Simplified for now
    }

    /// Check available memory
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        // Placeholder - cudarc doesn't expose memory info easily
        Ok((8_000_000_000, 16_000_000_000))  // 8GB free, 16GB total placeholder
    }
}