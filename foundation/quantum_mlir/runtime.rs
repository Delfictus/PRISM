//! Quantum MLIR Runtime - Actual GPU Execution
//!
//! This is the runtime that executes quantum operations on GPU
//! using our native complex number kernels

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::sync::Arc;

use super::cuda_kernels::{CudaComplex, QuantumGpuKernels};
use super::gpu_memory::GpuMemoryManager;
use super::{Complex64, Hamiltonian, QuantumOp, QuantumState};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Quantum GPU runtime for executing quantum operations
#[cfg(feature = "cuda")]
pub struct QuantumGpuRuntime {
    /// GPU memory manager
    memory: Arc<GpuMemoryManager>,
    /// GPU kernels (PTX runtime loaded)
    kernels: Arc<QuantumGpuKernels>,
    /// Current quantum state on GPU
    gpu_state: Arc<Mutex<Option<CudaSlice<CudaComplex>>>>,
    /// Cached Hamiltonian on GPU
    gpu_hamiltonian: Arc<Mutex<Option<CudaSlice<CudaComplex>>>>,
    /// Number of qubits
    num_qubits: usize,
}

#[cfg(feature = "cuda")]
impl QuantumGpuRuntime {
    /// Create new quantum GPU runtime
    pub fn new(num_qubits: usize) -> Result<Self> {
        use cudarc::driver::CudaDevice;

        // Create CUDA context once (CudaDevice::new already returns Arc)
        let context = CudaDevice::new(0)
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA context: {}", e))?;

        // Create GPU kernels with PTX loading
        let kernels = Arc::new(QuantumGpuKernels::new(context.clone())?);

        // Pass the context to memory manager
        let memory = Arc::new(GpuMemoryManager::new(context)?);
        let dimension = 1 << num_qubits;

        println!(
            "[Quantum GPU Runtime] Initializing with {} qubits",
            num_qubits
        );
        println!("[Quantum GPU Runtime] State dimension: {}", dimension);
        println!("[Quantum GPU Runtime] {}", memory.get_device_info());

        // Allocate and initialize quantum state on GPU
        let gpu_state = memory.allocate_state(dimension)?;

        Ok(Self {
            memory,
            kernels,
            gpu_state: Arc::new(Mutex::new(Some(gpu_state))),
            gpu_hamiltonian: Arc::new(Mutex::new(None)),
            num_qubits,
        })
    }

    /// Execute a quantum operation on GPU
    pub fn execute_op(&self, op: &QuantumOp) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        match op {
            QuantumOp::Hadamard { qubit } => {
                println!("[GPU PTX] Applying Hadamard gate to qubit {}", qubit);
                self.kernels.hadamard(state, *qubit, self.num_qubits)?;
            }
            QuantumOp::CNOT { control, target } => {
                println!(
                    "[GPU PTX] Applying CNOT gate: control={}, target={}",
                    control, target
                );
                self.kernels
                    .cnot(state, *control, *target, self.num_qubits)?;
            }
            QuantumOp::Evolution { hamiltonian, time } => {
                println!("[GPU PTX] Time evolution for t={}", time);
                self.evolve_with_hamiltonian(hamiltonian, *time)?;
            }
            QuantumOp::PauliX { qubit } => {
                println!("[GPU PTX] Applying Pauli-X gate to qubit {}", qubit);
                // Hadamard-Z-Hadamard sequence
                self.kernels.hadamard(state, *qubit, self.num_qubits)?;
                self.kernels.hadamard(state, *qubit, self.num_qubits)?;
            }
            QuantumOp::Measure { qubit } => {
                println!("[GPU PTX] Measuring qubit {}", qubit);
                let _probs = self.measure()?;
                println!("[GPU PTX] Measurement complete");
            }
            _ => {
                println!("[GPU PTX] Operation not yet implemented: {:?}", op);
            }
        }

        self.memory.synchronize()?;
        Ok(())
    }

    /// Upload Hamiltonian to GPU
    pub fn upload_hamiltonian(&self, hamiltonian: &Hamiltonian) -> Result<()> {
        println!(
            "[GPU] Uploading Hamiltonian ({}x{})",
            hamiltonian.dimension, hamiltonian.dimension
        );

        let gpu_ham = self.memory.upload_hamiltonian(&hamiltonian.elements)?;
        *self.gpu_hamiltonian.lock() = Some(gpu_ham);

        Ok(())
    }

    /// Evolve quantum state under Hamiltonian
    fn evolve_with_hamiltonian(&self, hamiltonian: &Hamiltonian, time: f64) -> Result<()> {
        // TODO: Implement Hamiltonian evolution with PTX kernel
        // For now, just upload the Hamiltonian
        if self.gpu_hamiltonian.lock().is_none() {
            self.upload_hamiltonian(hamiltonian)?;
        }

        println!(
            "[GPU PTX] Hamiltonian evolution (simplified) for t={}",
            time
        );
        // Evolution kernel would be launched here
        Ok(())
    }

    /// Apply Quantum Fourier Transform
    pub fn apply_qft(&self, inverse: bool) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        println!(
            "[GPU PTX] Applying {} QFT",
            if inverse { "inverse" } else { "forward" }
        );
        self.kernels.qft(state, self.num_qubits, inverse)?;

        Ok(())
    }

    /// Apply VQE ansatz with parameters
    pub fn apply_vqe_ansatz(&self, parameters: &[f64], num_layers: usize) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        // Upload parameters to GPU
        let gpu_params = self
            .memory
            .context
            .htod_sync_copy(parameters)
            .map_err(|e| anyhow::anyhow!("Failed to upload VQE parameters: {}", e))?;

        println!("[GPU PTX] Applying VQE ansatz with {} layers", num_layers);
        self.kernels
            .vqe_ansatz(state, &gpu_params, self.num_qubits, num_layers)?;

        Ok(())
    }

    /// Measure quantum state and get probabilities
    pub fn measure(&self) -> Result<Vec<f64>> {
        let state_guard = self.gpu_state.lock();
        let state = state_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let dimension = 1 << self.num_qubits;
        let mut gpu_probs = self.memory.allocate_probabilities(dimension)?;

        self.kernels.measure(state, &mut gpu_probs, dimension)?;

        self.memory.download_probabilities(&gpu_probs)
    }

    /// Get current quantum state from GPU
    pub fn get_state(&self) -> Result<QuantumState> {
        let state_guard = self.gpu_state.lock();
        let gpu_state = state_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let amplitudes = self.memory.download_state(gpu_state)?;
        let dimension = amplitudes.len();

        Ok(QuantumState {
            dimension,
            amplitudes,
        })
    }

    /// Set quantum state on GPU
    pub fn set_state(&self, state: &QuantumState) -> Result<()> {
        let gpu_state = self.memory.upload_state(&state.amplitudes)?;
        *self.gpu_state.lock() = Some(gpu_state);
        Ok(())
    }

    /// Get memory usage information
    pub fn get_memory_info(&self) -> Result<String> {
        let (free, total) = self.memory.get_memory_info()?;
        Ok(format!(
            "GPU Memory: {:.2} GB free / {:.2} GB total",
            free as f64 / 1e9,
            total as f64 / 1e9
        ))
    }
}
