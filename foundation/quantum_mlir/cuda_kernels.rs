//! CUDA Kernel Runtime Loading via PTX
//!
//! Uses cudarc's PTX runtime loading instead of FFI to .o files
//! This solves ALL linking issues - kernels loaded at runtime from PTX files

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx_with_opts;
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA complex number type matching cuDoubleComplex
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaComplex {
    pub real: f64,
    pub imag: f64,
}

impl CudaComplex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }
}

// Implement required traits for CudaComplex to work with cudarc
unsafe impl cudarc::driver::DeviceRepr for CudaComplex {}
unsafe impl cudarc::driver::ValidAsZeroBits for CudaComplex {}

/// Quantum GPU Kernels using PTX runtime loading
pub struct QuantumGpuKernels {
    device: Arc<CudaDevice>,
    kernels: HashMap<String, Arc<CudaFunction>>,
}

impl QuantumGpuKernels {
    /// Create new quantum GPU kernels with PTX runtime loading
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Load PTX module at runtime
        println!("[Quantum PTX] Loading quantum_mlir.ptx...");

        // Try multiple possible PTX locations
        let ptx_paths = vec![
            "target/ptx/quantum_mlir.ptx",
            "../target/ptx/quantum_mlir.ptx",
            "../../target/ptx/quantum_mlir.ptx",
            concat!(env!("OUT_DIR"), "/quantum_mlir.ptx"),
        ];

        let mut ptx_path = None;
        for path in &ptx_paths {
            if std::path::Path::new(path).exists() {
                ptx_path = Some(*path);
                println!("[Quantum PTX] Found PTX at: {}", path);
                break;
            }
        }

        let ptx_path = ptx_path.ok_or_else(|| {
            anyhow::anyhow!("quantum_mlir.ptx not found in any expected location")
        })?;

        // Load PTX module
        let ptx = std::fs::read_to_string(ptx_path).context("Failed to read PTX file")?;

        // Load PTX and kernel functions
        let kernel_names = vec![
            "hadamard_gate_kernel",
            "cnot_gate_kernel",
            "qft_kernel",
            "vqe_ansatz_kernel",
            "measurement_kernel", // Note: it's "measurement" not "measure" in PTX
        ];

        device
            .load_ptx(ptx.into(), "quantum_module", &kernel_names)
            .context("Failed to load PTX module")?;

        println!("[Quantum PTX] ✓ PTX module loaded");

        // Get kernel functions
        let mut kernels = HashMap::new();

        for name in kernel_names {
            match device.get_func("quantum_module", name) {
                Ok(func) => {
                    println!("[Quantum PTX] ✓ Loaded: {}", name);
                    kernels.insert(name.to_string(), Arc::new(func));
                }
                Err(e) => {
                    println!("[Quantum PTX] ⚠ Failed to load {}: {}", name, e);
                }
            }
        }

        println!("[Quantum PTX] ✓ Native cuDoubleComplex support ready");

        Ok(Self { device, kernels })
    }

    /// Apply Hadamard gate on GPU
    pub fn hadamard(
        &self,
        state: &mut CudaSlice<CudaComplex>,
        qubit: usize,
        num_qubits: usize,
    ) -> Result<()> {
        let dimension = 1 << num_qubits;
        let num_blocks = ((dimension / 2) + 255) / 256;
        let num_threads = 256;

        let func = self
            .kernels
            .get("hadamard_gate_kernel")
            .ok_or_else(|| anyhow::anyhow!("Hadamard kernel not loaded"))?;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let qubit_i32 = qubit as i32;
        let num_qubits_i32 = num_qubits as i32;

        unsafe {
            func.clone()
                .launch(config, (state, &qubit_i32, &num_qubits_i32))?;
        }

        Ok(())
    }

    /// Apply CNOT gate on GPU
    pub fn cnot(
        &self,
        state: &mut CudaSlice<CudaComplex>,
        control: usize,
        target: usize,
        num_qubits: usize,
    ) -> Result<()> {
        let dimension = 1 << num_qubits;
        let num_blocks = ((dimension / 4) + 255) / 256;
        let num_threads = 256;

        let func = self
            .kernels
            .get("cnot_gate_kernel")
            .ok_or_else(|| anyhow::anyhow!("CNOT kernel not loaded"))?;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let control_i32 = control as i32;
        let target_i32 = target as i32;
        let num_qubits_i32 = num_qubits as i32;

        unsafe {
            func.clone()
                .launch(config, (state, &control_i32, &target_i32, &num_qubits_i32))?;
        }

        Ok(())
    }

    /// Apply Quantum Fourier Transform on GPU
    pub fn qft(
        &self,
        state: &mut CudaSlice<CudaComplex>,
        num_qubits: usize,
        inverse: bool,
    ) -> Result<()> {
        let dimension = 1 << num_qubits;
        let num_blocks = (dimension + 255) / 256;
        let num_threads = 256;

        let func = self
            .kernels
            .get("qft_kernel")
            .ok_or_else(|| anyhow::anyhow!("QFT kernel not loaded"))?;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_qubits_i32 = num_qubits as i32;

        unsafe {
            func.clone()
                .launch(config, (state, &num_qubits_i32, &inverse))?;
        }

        Ok(())
    }

    /// VQE ansatz application
    pub fn vqe_ansatz(
        &self,
        state: &mut CudaSlice<CudaComplex>,
        parameters: &CudaSlice<f64>,
        num_qubits: usize,
        num_layers: usize,
    ) -> Result<()> {
        let dimension = 1 << num_qubits;
        let num_blocks = (dimension + 255) / 256;
        let num_threads = 256;

        let func = self
            .kernels
            .get("vqe_ansatz_kernel")
            .ok_or_else(|| anyhow::anyhow!("VQE kernel not loaded"))?;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_qubits_i32 = num_qubits as i32;
        let num_layers_i32 = num_layers as i32;

        unsafe {
            func.clone().launch(
                config,
                (state, parameters, &num_qubits_i32, &num_layers_i32),
            )?;
        }

        Ok(())
    }

    /// Measure quantum state
    pub fn measure(
        &self,
        state: &CudaSlice<CudaComplex>,
        probabilities: &mut CudaSlice<f64>,
        dimension: usize,
    ) -> Result<()> {
        let num_blocks = (dimension + 255) / 256;
        let num_threads = 256;

        let func = self
            .kernels
            .get("measurement_kernel")
            .ok_or_else(|| anyhow::anyhow!("Measurement kernel not loaded"))?;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dimension_i32 = dimension as i32;

        unsafe {
            func.clone()
                .launch(config, (state, probabilities, &dimension_i32))?;
        }

        Ok(())
    }
}
