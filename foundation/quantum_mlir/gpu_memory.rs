//! GPU Memory Management for Quantum MLIR
//!
//! Handles GPU memory allocation and data transfer using cudarc

use anyhow::Result;
use cudarc::driver::result::DriverError;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

use super::cuda_kernels::CudaComplex;
use super::Complex64;

/// GPU memory manager for quantum states
pub struct GpuMemoryManager {
    pub context: Arc<CudaDevice>,
}

impl GpuMemoryManager {
    /// Create new GPU memory manager with shared CUDA context
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (CudaDevice::new already returns Arc)
    pub fn new(context: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { context })
    }

    /// Allocate GPU memory for quantum state
    pub fn allocate_state(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        // Initialize to |00...0> state
        let mut init = vec![CudaComplex::zero(); dimension];
        init[0] = CudaComplex::one();

        // Upload initial state directly
        let state = self.context.htod_sync_copy(&init).map_err(|e| {
            anyhow::anyhow!(
                "Failed to allocate and initialize quantum state on GPU: {}",
                e
            )
        })?;

        Ok(state)
    }

    /// Allocate GPU memory for Hamiltonian matrix
    pub fn allocate_hamiltonian(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        let size = dimension * dimension;
        self.context
            .alloc_zeros::<CudaComplex>(size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate GPU memory for Hamiltonian: {}", e))
    }

    /// Copy quantum state from host to device
    pub fn upload_state(&self, host_state: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_state: Vec<CudaComplex> = host_state
            .iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        self.context
            .htod_sync_copy(&cuda_state)
            .map_err(|e| anyhow::anyhow!("Failed to upload quantum state to GPU: {}", e))
    }

    /// Copy quantum state from device to host
    pub fn download_state(&self, device_state: &CudaSlice<CudaComplex>) -> Result<Vec<Complex64>> {
        let cuda_state = self
            .context
            .dtoh_sync_copy(device_state)
            .map_err(|e| anyhow::anyhow!("Failed to download quantum state from GPU: {}", e))?;

        Ok(cuda_state
            .into_iter()
            .map(|c| Complex64 {
                real: c.real,
                imag: c.imag,
            })
            .collect())
    }

    /// Upload Hamiltonian matrix to GPU
    pub fn upload_hamiltonian(&self, hamiltonian: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_ham: Vec<CudaComplex> = hamiltonian
            .iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        self.context
            .htod_sync_copy(&cuda_ham)
            .map_err(|e| anyhow::anyhow!("Failed to upload Hamiltonian to GPU: {}", e))
    }

    /// Allocate GPU memory for measurement probabilities
    pub fn allocate_probabilities(&self, dimension: usize) -> Result<CudaSlice<f64>> {
        self.context
            .alloc_zeros::<f64>(dimension)
            .map_err(|e| anyhow::anyhow!("Failed to allocate GPU memory for probabilities: {}", e))
    }

    /// Download probabilities from GPU
    pub fn download_probabilities(&self, device_probs: &CudaSlice<f64>) -> Result<Vec<f64>> {
        self.context
            .dtoh_sync_copy(device_probs)
            .map_err(|e| anyhow::anyhow!("Failed to download probabilities from GPU: {}", e))
    }

    /// Get raw pointer to GPU memory (for FFI)
    pub fn get_ptr<T>(&self, slice: &CudaSlice<T>) -> *mut T {
        slice.device_ptr() as *mut T
    }

    /// Get const pointer to GPU memory (for FFI)
    pub fn get_const_ptr<T>(&self, slice: &CudaSlice<T>) -> *const T {
        slice.device_ptr() as *const T
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to synchronize GPU: {}", e))
    }

    /// Get device properties
    pub fn get_device_info(&self) -> String {
        format!("CUDA Device 0") // Simplified for now
    }

    /// Check available memory
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        // Placeholder - cudarc doesn't expose memory info easily
        Ok((8_000_000_000, 16_000_000_000)) // 8GB free, 16GB total placeholder
    }
}
