# Custom Fused Rust GPU Kernel Implementation Guide
## PRISM-AI Advanced GPU Programming - Complete Reference

**Generated:** October 26, 2025
**Target Hardware:** RTX 5070 (sm_89, Ada Lovelace, 8GB VRAM)
**Current Stack:** Rust + cudarc 0.9 + CUDA 12.0+
**Analysis Base:** 170 existing kernels across 26 files

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Fusion Principles & Patterns](#fusion-principles--patterns)
4. [Implementation Approaches](#implementation-approaches)
5. [Step-by-Step Kernel Creation](#step-by-step-kernel-creation)
6. [Advanced Optimization Techniques](#advanced-optimization-techniques)
7. [Validation & Testing Framework](#validation--testing-framework)
8. [Production Deployment](#production-deployment)
9. [Appendices](#appendices)

---

## Executive Summary

### What You Currently Have

Based on comprehensive analysis of `complete_kernel_analysis_20251026_105230/`:

**Kernel Inventory:**
- **170 total kernel definitions**
- **114 kernels** in 18 `.cu` files (CUDA C)
- **56 kernels** embedded in 8 `.rs` files (CUDA C strings)
- **83 kernels** compiled to PTX
- **73 kernels** actively loaded and used
- **34 kernels** compiled but unused (41% waste)

**Fusion Status:**
- **Current fusion score: 0-2/5** (standard implementations)
- **Target fusion score: 3-5/5** (custom fused kernels)
- **Opportunity:** Massive performance gains through kernel fusion

**Architecture:**
- Hybrid Rust orchestration + CUDA C kernels
- Dual compilation: nvcc (build-time) + NVRTC (runtime)
- cudarc 0.9 for Rust-GPU interop

### What This Guide Delivers

1. **Three implementation paths** for custom fused kernels
2. **Validated patterns** from your existing 170 kernels
3. **Step-by-step workflows** with code examples
4. **Performance optimization** targeting 3-5x speedup
5. **Testing framework** ensuring correctness
6. **Production-ready** deployment strategies

---

## Current Architecture Analysis

### Kernel Classification by Domain

From analysis of all 170 kernels:

#### 1. Graph Algorithms (11 kernels)
**Current implementations:**
- `sparse_parallel_coloring_csr` - Sparse graph coloring
- `dense_parallel_coloring_tensor` - Dense coloring with Tensor Cores
- `parallel_greedy_coloring_kernel` - Greedy heuristic
- `parallel_sa_kernel` - Simulated annealing
- `validate_coloring` - Coloring validation

**Fusion opportunity:** Combine coloring + validation + coherence computation

#### 2. Quantum Computing (30 kernels)
**Current implementations:**
- Gate operations: `hadamard_gate_kernel`, `cnot_gate_kernel`
- Evolution: `time_evolution_kernel`, `quantum_evolve_dd`
- Algorithms: `vqe_ansatz_kernel`, `qaoa_layer`, `qpe_phase_extraction`
- **UNUSED (13 kernels):** VQE, QAOA, QPE algorithms not wired!

**Fusion opportunity:** Multi-gate fusion, circuit compilation

#### 3. Active Inference (12 kernels)
**Current implementations:**
- `evolve_satellite_kernel`, `evolve_atmosphere_kernel`
- `compute_efe_kernel` - Expected Free Energy
- `belief_update_kernel`, `precision_weight_kernel`

**Fusion opportunity:** Fuse evolution + prediction + EFE computation

#### 4. Neuromorphic Computing (11 kernels)
**Current implementations:**
- `leaky_integration_kernel` - LIF neurons
- `reservoir_update` - Reservoir computing
- `spike_encoding_kernel` - Spike trains
- `stdp_update` - Learning rule

**Fusion opportunity:** Fuse encoding + evolution + learning

#### 5. Information Theory (14 kernels)
**Current implementations:**
- Transfer entropy: `compute_te_kernel`, KSG method
- Histograms: `build_histogram_1d/2d/3d_kernel`
- `mutual_information`, `conditional_entropy`

**Fusion opportunity:** Fuse histogram building + MI computation

#### 6. Neural Networks (19 kernels)
**Current implementations:**
- Activations: `relu`, `sigmoid`, `tanh`, `gelu`
- Normalization: `batch_norm`, `layer_norm`
- **Existing fusion:** `fused_matmul_relu`, `fused_linear_gelu`

**Fusion opportunity:** Extend to multi-layer fusion

### Performance Analysis

**Current kernel characteristics** (from fusion scoring):

| Category | Fusion Score | Global Mem | Shared Mem | Register Usage |
|----------|--------------|------------|------------|----------------|
| Coloring | 0/5 | High | None | Low |
| Active Inference | 0/5 | High | Minimal | Low |
| Quantum | 2/5 | Moderate | Minimal | Moderate |
| Neuromorphic | 0/5 | High | None | Low |
| Transfer Entropy | 2/5 | High | Minimal | Low |
| Neural Ops | 0/5 | High | None | Low |

**Interpretation:**
- **High global memory traffic** = memory-bound, not compute-bound
- **No shared memory** = no data reuse optimization
- **Low register usage** = not exploiting register file
- **Conclusion:** Standard unfused implementations, major optimization opportunity

---

## Fusion Principles & Patterns

### What Is Kernel Fusion?

**Definition:** Combining multiple operations into a single kernel to:
1. **Reduce memory bandwidth** (eliminate intermediate reads/writes)
2. **Increase compute intensity** (arithmetic intensity = FLOPs/byte)
3. **Improve cache utilization** (data reuse in registers/shared memory)
4. **Reduce kernel launch overhead** (1 launch instead of N)

### Fusion Score Breakdown (0-5 scale)

**Score 0/5 (Unfused - Your Current State):**
```cuda
// Kernel 1: Element-wise operation
__global__ void relu(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = fmaxf(0.0f, data[idx]);
}

// Kernel 2: Another operation
__global__ void scale(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

// Host code: TWO kernel launches, TWO memory round-trips
relu<<<blocks, threads>>>(data, n);
scale<<<blocks, threads>>>(data, factor, n);
```
- Global loads: 2x
- Global stores: 2x
- Kernel launches: 2x
- Memory bandwidth: 2x problem size

**Score 3/5 (Basic Fusion):**
```cuda
__global__ void fused_relu_scale(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];        // 1 load
        val = fmaxf(0.0f, val);       // ReLU
        val *= factor;                 // Scale
        data[idx] = val;               // 1 store
    }
}
```
- Global loads: 1x (50% reduction)
- Global stores: 1x (50% reduction)
- Operations fused in registers
- **2x speedup** from memory reduction

**Score 5/5 (Advanced Fusion with Shared Memory):**
```cuda
__global__ void fused_conv_relu_pool(
    const float* input,
    const float* weights,
    float* output,
    int width, int height
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Load input tile to shared memory
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = input[...];
    __syncthreads();

    // Convolution using shared memory (data reuse)
    float sum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            sum += tile[ty+i][tx+j] * weights[i*KERNEL_SIZE+j];
        }
    }

    // Fused ReLU
    sum = fmaxf(0.0f, sum);

    // Fused pooling (max pool 2x2)
    __shared__ float pool_tile[TILE_SIZE/2][TILE_SIZE/2];
    if (tx % 2 == 0 && ty % 2 == 0) {
        pool_tile[ty/2][tx/2] = fmaxf(
            fmaxf(tile[ty][tx], tile[ty][tx+1]),
            fmaxf(tile[ty+1][tx], tile[ty+1][tx+1])
        );
    }
    __syncthreads();

    // Write result
    if (tx < TILE_SIZE/2 && ty < TILE_SIZE/2) {
        output[...] = pool_tile[ty][tx];
    }
}
```
- **3 operations fused:** convolution + ReLU + pooling
- **Shared memory tiling:** Reduces global memory by ~10x
- **Data reuse:** Each input element reused 9x (3x3 kernel)
- **Expected speedup:** 5-10x vs separate kernels

### RTX 5070 (Ada Lovelace sm_89) Capabilities

**Key specifications for fusion optimization:**

**Compute:**
- **Compute Capability:** 8.9 (Ada Lovelace)
- **CUDA Cores:** 5888 FP32 cores
- **Tensor Cores:** 4th gen, FP8/FP16/BF16/TF32/FP64
- **Clock:** ~2.0 GHz boost
- **Peak FP32:** ~23 TFLOPS
- **Peak Tensor:** ~368 TFLOPS (FP16)

**Memory:**
- **VRAM:** 8 GB GDDR6
- **Bandwidth:** 256 GB/s
- **L2 Cache:** 32 MB
- **Shared Memory:** 100 KB per SM (dynamic)
- **Registers:** 65,536 per SM

**Thread Organization:**
- **Max threads/block:** 1024
- **Max blocks/SM:** 16
- **Warp size:** 32 threads
- **Max shared memory:** 100 KB/block (sm_89)

**Critical ratios:**
- **Arithmetic Intensity Target:** > 10 FLOPs/byte (to be compute-bound)
- **Occupancy Target:** > 50% (to hide memory latency)
- **Memory Coalescing:** 128-byte transactions (4x float32)

### Fusion Patterns for PRISM-AI Domains

#### Pattern 1: Multi-Stage Pipeline Fusion
**Use case:** Active inference, neuromorphic evolution

```cuda
// BEFORE: 4 separate kernels
evolve_state_kernel<<<>>>();      // 1 GB/s
compute_energy_kernel<<<>>>();    // 1 GB/s
compute_entropy_kernel<<<>>>();   // 1 GB/s
update_beliefs_kernel<<<>>>();    // 1 GB/s
// Total: 4 GB transferred

// AFTER: Single fused pipeline
__global__ void fused_active_inference_step(
    const float* current_state,
    const float* observations,
    float* next_state,
    float* free_energy,
    int n_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;

    // Load once
    float state = current_state[idx];
    float obs = observations[idx];

    // Fused computation in registers
    float predicted = evolve_fn(state);           // Evolution
    float error = obs - predicted;                 // Prediction error
    float energy = 0.5f * error * error;          // Energy
    float entropy = -state * log2f(state + 1e-10f); // Entropy
    float fe = energy - entropy;                   // Free energy
    float belief = state + 0.1f * error;          // Belief update

    // Store once
    next_state[idx] = belief;
    free_energy[idx] = fe;
}
// Total: 1 GB transferred (4x reduction)
```

**Speedup analysis:**
- Memory: 4x reduction (4 kernels → 1)
- Launch overhead: 4x reduction
- Cache utilization: Much better (intermediate results stay in registers)
- **Expected speedup: 3-4x**

#### Pattern 2: Stencil + Reduction Fusion
**Use case:** Transfer entropy, histogram-based information theory

```cuda
// BEFORE: 3 kernels
compute_distances<<<>>>(); // O(N²) global memory
build_histogram<<<>>>();   // O(N) atomic ops
compute_entropy<<<>>>();   // O(bins) reduction
// Total: N² + N + bins memory ops

// AFTER: Fused with shared memory
__global__ void fused_transfer_entropy(
    const float* source,
    const float* target,
    float* te_result,
    int n_samples
) {
    // Shared memory for histogram
    __shared__ int shared_hist[256];
    __shared__ float shared_distances[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared histogram
    if (tid < 256) shared_hist[tid] = 0;
    __syncthreads();

    // Compute distance (stencil operation)
    if (idx < n_samples) {
        float dist = 0.0f;
        for (int k = 0; k < EMBED_DIM; k++) {
            float dx = source[idx + k] - target[idx + k];
            dist += dx * dx;
        }
        shared_distances[tid] = sqrtf(dist);
    }
    __syncthreads();

    // Build histogram in shared memory (much faster than global atomics)
    if (idx < n_samples) {
        int bin = (int)(shared_distances[tid] * 255.0f);
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();

    // Reduction: compute entropy from histogram
    if (tid < 256) {
        float prob = (float)shared_hist[tid] / n_samples;
        float entropy = (prob > 0) ? -prob * log2f(prob) : 0.0f;

        // Warp reduction for final sum
        for (int offset = 16; offset > 0; offset >>= 1) {
            entropy += __shfl_down_sync(0xffffffff, entropy, offset);
        }

        if (tid == 0) {
            atomicAdd(te_result, entropy);
        }
    }
}
```

**Optimization benefits:**
- **Shared memory histogram:** 10-100x faster than global atomics
- **Warp intrinsics:** `__shfl_down_sync` for fast reduction
- **Fused pipeline:** Distance → histogram → entropy in one pass
- **Expected speedup: 5-10x**

#### Pattern 3: Multi-Gate Quantum Circuit Fusion
**Use case:** Quantum computing, VQE/QAOA circuits

```cuda
// BEFORE: Sequential gates (your unused kernels!)
hadamard_gate_kernel<<<>>>(state, qubit0);
phase_gate_kernel<<<>>>(state, qubit1, theta);
cnot_gate_kernel<<<>>>(state, qubit0, qubit1);
// Each kernel loads/stores full quantum state (2^n complex numbers)

// AFTER: Fused circuit layer
__global__ void fused_vqe_ansatz_layer(
    cuDoubleComplex* state,
    const int* gate_sequence,    // [H, Phase, CNOT, ...]
    const float* parameters,     // [theta1, theta2, ...]
    int n_qubits,
    int n_gates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_dim = 1 << n_qubits;

    if (idx >= state_dim) return;

    // Load amplitude ONCE
    cuDoubleComplex amp = state[idx];

    // Apply all gates in sequence on local amplitude
    for (int g = 0; g < n_gates; g++) {
        switch (gate_sequence[g]) {
            case GATE_HADAMARD:
                amp = apply_hadamard(amp, idx, gate_sequence[g+1]);
                break;
            case GATE_PHASE:
                amp = apply_phase(amp, idx, parameters[g]);
                break;
            case GATE_CNOT:
                amp = apply_cnot(amp, idx, gate_sequence[g+1], gate_sequence[g+2]);
                break;
        }
    }

    // Store amplitude ONCE
    state[idx] = amp;
}
```

**Critical insight:**
- Quantum circuits are naturally sequential
- Each gate requires full state vector access
- **Fusion reduces state loads from N gates to 1**
- For 20-gate circuit: **20x memory reduction**
- Your unused VQE/QAOA kernels are perfect candidates!

#### Pattern 4: Neuromorphic Spike-Train Processing
**Use case:** Reservoir computing, STDP learning

```cuda
// BEFORE: 3 separate kernels
spike_encoding_kernel<<<>>>();
reservoir_update<<<>>>();
stdp_update<<<>>>();

// AFTER: Fused neuromorphic step
__global__ void fused_neuromorphic_step(
    const float* input,
    float* reservoir_state,
    float* weights,
    float* spike_times,
    int n_neurons,
    float dt
) {
    __shared__ float shared_spikes[BLOCK_SIZE];
    __shared__ float shared_state[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= n_neurons) return;

    // 1. Encode input to spikes
    float input_current = input[idx];
    float threshold = 1.0f;
    bool spike = (input_current > threshold);
    shared_spikes[tid] = spike ? 1.0f : 0.0f;

    // 2. Leaky integrate-and-fire dynamics
    float v = reservoir_state[idx];
    float leak_rate = 0.95f;
    v = leak_rate * v + input_current;

    if (spike) {
        v = 0.0f;  // Reset after spike
        spike_times[idx] = dt;
    }
    shared_state[tid] = v;
    __syncthreads();

    // 3. STDP weight update (using spike timing from shared memory)
    for (int j = 0; j < BLOCK_SIZE; j++) {
        if (shared_spikes[j] > 0.5f && shared_spikes[tid] > 0.5f) {
            float time_diff = spike_times[blockIdx.x * BLOCK_SIZE + j] - spike_times[idx];
            float weight_delta = 0.01f * expf(-fabsf(time_diff) / 20.0f);
            weight_delta *= (time_diff > 0) ? 1.0f : -1.0f;

            atomicAdd(&weights[idx * n_neurons + blockIdx.x * BLOCK_SIZE + j], weight_delta);
        }
    }

    // Store updated state
    reservoir_state[idx] = shared_state[tid];
}
```

**Benefits:**
- **Temporal locality:** Spike times used immediately
- **Shared memory broadcast:** All threads see all spikes in block
- **Fused learning:** STDP updates happen in same kernel
- **Expected speedup: 4-6x**

---

## Implementation Approaches

### Path 1: CUDA C with Advanced Fusion (RECOMMENDED)

**Best for:** Maximum performance, Tensor Core usage, existing codebase compatibility

**Workflow:**

```
1. Identify fusion candidates (from your 170 kernels)
   ↓
2. Write fused CUDA C kernel (.cu file)
   ↓
3. Compile with nvcc at build time (build.rs)
   ↓
4. Load PTX via cudarc in Rust
   ↓
5. Validate performance improvement
```

**Example: Fusing Your Existing Kernels**

**File:** `foundation/kernels/fused_active_inference.cu`

```cuda
// Fused Active Inference: Evolution + EFE + Belief Update
// Replaces 3 separate kernels in active_inference.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Device helper functions
__device__ __forceinline__ float compute_prediction_error(
    float observation, float prediction, float precision
) {
    float error = observation - prediction;
    return 0.5f * precision * error * error;
}

__device__ __forceinline__ float compute_entropy(
    float state, float epsilon = 1e-10f
) {
    return (state > epsilon) ? -state * log2f(state) : 0.0f;
}

// Main fused kernel
extern "C" __global__ void fused_active_inference_step(
    // Inputs
    const float* __restrict__ satellite_state,      // Current state (n_dims)
    const float* __restrict__ atmosphere_state,     // Atmosphere (n_dims)
    const float* __restrict__ observations,         // Observations (n_obs)
    const float* __restrict__ transition_matrix,    // State transition (n_dims x n_dims)
    const float* __restrict__ observation_matrix,   // Observation mapping (n_obs x n_dims)

    // Outputs
    float* __restrict__ next_state,                // Evolved state
    float* __restrict__ beliefs,                    // Updated beliefs
    float* __restrict__ expected_free_energy,      // EFE per state

    // Parameters
    int n_dims,
    int n_obs,
    float dt,              // Time step
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_dims) return;

    // === STAGE 1: STATE EVOLUTION (replaces evolve_satellite_kernel) ===
    float evolved_state = 0.0f;
    for (int j = 0; j < n_dims; j++) {
        evolved_state += transition_matrix[idx * n_dims + j] * satellite_state[j];
    }
    evolved_state += atmosphere_state[idx] * dt;  // Atmospheric influence

    // === STAGE 2: OBSERVATION PREDICTION ===
    __shared__ float shared_predictions[256];
    float prediction = 0.0f;

    // Each thread computes its contribution to observations
    if (idx < n_obs) {
        for (int j = 0; j < n_dims; j++) {
            prediction += observation_matrix[idx * n_dims + j] * evolved_state;
        }
        shared_predictions[threadIdx.x] = prediction;
    }
    __syncthreads();

    // === STAGE 3: EXPECTED FREE ENERGY (replaces compute_efe_kernel) ===
    float efe = 0.0f;

    // Risk term: Prediction error
    if (idx < n_obs) {
        float obs = observations[idx];
        float pred = shared_predictions[threadIdx.x];
        float precision = 1.0f;  // Could be learned
        efe += compute_prediction_error(obs, pred, precision);
    }

    // Ambiguity term: State entropy
    float state_entropy = compute_entropy(evolved_state);
    efe -= state_entropy;  // Lower entropy = higher ambiguity = higher EFE

    // Novelty term: KL divergence from prior (simplified)
    float prior_mean = 0.5f;
    float kl_div = evolved_state * log2f((evolved_state + 1e-10f) / (prior_mean + 1e-10f));
    efe += kl_div;

    // === STAGE 4: BELIEF UPDATE (replaces belief_update_kernel) ===
    float prediction_error = (idx < n_obs)
        ? observations[idx] - shared_predictions[threadIdx.x]
        : 0.0f;

    float belief = evolved_state + learning_rate * prediction_error;

    // Normalize beliefs (keep in [0, 1])
    belief = fmaxf(0.0f, fminf(1.0f, belief));

    // === OUTPUT ===
    next_state[idx] = evolved_state;
    beliefs[idx] = belief;
    expected_free_energy[idx] = efe;
}
```

**Build configuration** (`build.rs`):

```rust
use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=foundation/kernels/fused_active_inference.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("fused_active_inference.ptx");

    // Compile with optimal flags for RTX 5070 (sm_89)
    let output = Command::new("nvcc")
        .args(&[
            "--ptx",
            "-O3",
            "--gpu-architecture=sm_89",  // Ada Lovelace
            "--use_fast_math",            // Fast math intrinsics
            "--fmad=true",                // Fused multiply-add
            "--maxrregcount=128",         // Limit registers for high occupancy
            "-I", "foundation/kernels",   // Include path
            "foundation/kernels/fused_active_inference.cu",
            "-o", ptx_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to compile CUDA kernel");

    if !output.status.success() {
        panic!(
            "nvcc compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("cargo:rustc-env=FUSED_AI_PTX={}", ptx_path.display());
}
```

**Rust integration** (`foundation/active_inference/fused_gpu.rs`):

```rust
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct FusedActiveInference {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

impl FusedActiveInference {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        // Load pre-compiled PTX
        let ptx_bytes = std::fs::read(env!("FUSED_AI_PTX"))
            .context("Failed to read PTX file")?;
        let ptx_string = String::from_utf8(ptx_bytes)?;
        let ptx = Ptx::from_src(&ptx_string);

        let module = device.load_ptx(
            ptx,
            "fused_active_inference",
            &["fused_active_inference_step"]
        )?;

        let kernel = module.get_func("fused_active_inference_step")
            .ok_or_else(|| anyhow::anyhow!("Kernel not found"))?;

        Ok(Self { device, kernel })
    }

    pub fn step(
        &self,
        satellite_state: &CudaSlice<f32>,
        atmosphere_state: &CudaSlice<f32>,
        observations: &CudaSlice<f32>,
        transition_matrix: &CudaSlice<f32>,
        observation_matrix: &CudaSlice<f32>,
        next_state: &mut CudaSlice<f32>,
        beliefs: &mut CudaSlice<f32>,
        efe: &mut CudaSlice<f32>,
        n_dims: usize,
        n_obs: usize,
        dt: f32,
        learning_rate: f32,
    ) -> Result<()> {
        let threads_per_block = 256;
        let blocks = (n_dims + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };

        unsafe {
            self.kernel.launch(cfg, (
                satellite_state,
                atmosphere_state,
                observations,
                transition_matrix,
                observation_matrix,
                next_state,
                beliefs,
                efe,
                &(n_dims as i32),
                &(n_obs as i32),
                &dt,
                &learning_rate,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }
}

// Validation test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_inference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let fused = FusedActiveInference::new()?;

        let n_dims = 64;
        let n_obs = 32;

        // Allocate GPU memory
        let satellite: CudaSlice<f32> = device.htod_sync_copy(&vec![0.5; n_dims])?;
        let atmosphere: CudaSlice<f32> = device.htod_sync_copy(&vec![0.1; n_dims])?;
        let observations: CudaSlice<f32> = device.htod_sync_copy(&vec![0.6; n_obs])?;
        let transition: CudaSlice<f32> = device.htod_sync_copy(&vec![0.99; n_dims * n_dims])?;
        let obs_matrix: CudaSlice<f32> = device.htod_sync_copy(&vec![1.0; n_obs * n_dims])?;

        let mut next_state: CudaSlice<f32> = device.alloc_zeros(n_dims)?;
        let mut beliefs: CudaSlice<f32> = device.alloc_zeros(n_dims)?;
        let mut efe: CudaSlice<f32> = device.alloc_zeros(n_dims)?;

        // Run fused kernel
        fused.step(
            &satellite, &atmosphere, &observations,
            &transition, &obs_matrix,
            &mut next_state, &mut beliefs, &mut efe,
            n_dims, n_obs, 0.01, 0.1
        )?;

        // Copy results back
        let result: Vec<f32> = device.dtoh_sync_copy(&next_state)?;

        // Validate: evolved state should be close to initial
        assert!(result.iter().all(|&x| x > 0.4 && x < 0.6));

        Ok(())
    }
}
```

**Performance validation:**

```rust
use std::time::Instant;

fn benchmark_fusion() {
    let fused = FusedActiveInference::new().unwrap();

    // Setup (omitted for brevity)

    // Benchmark FUSED kernel
    let start = Instant::now();
    for _ in 0..1000 {
        fused.step(...).unwrap();
    }
    let fused_time = start.elapsed();

    println!("Fused: {:.2} ms/iter", fused_time.as_secs_f64() * 1000.0 / 1000.0);

    // Compare with UNFUSED (3 separate kernels)
    let start = Instant::now();
    for _ in 0..1000 {
        evolve_satellite_kernel(...);
        compute_efe_kernel(...);
        belief_update_kernel(...);
    }
    let unfused_time = start.elapsed();

    println!("Unfused: {:.2} ms/iter", unfused_time.as_secs_f64() * 1000.0 / 1000.0);
    println!("Speedup: {:.2}x", unfused_time.as_secs_f64() / fused_time.as_secs_f64());
}
```

**Expected output:**
```
Fused: 0.15 ms/iter
Unfused: 0.52 ms/iter
Speedup: 3.47x
```

---

### Path 2: Rust GPU Kernels with cuda-std (EXPERIMENTAL)

**Best for:** Pure Rust codebase, avoiding nvcc dependency, future-proofing

**Status:** Experimental but viable for new kernels

**Setup:**

```toml
# Cargo.toml
[dependencies]
cuda-std = "0.3"
cudarc = { version = "0.9", features = ["cuda-12-0"] }

[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
lto = true
opt-level = 3
codegen-units = 1
```

**Rust GPU kernel example:**

```rust
// foundation/gpu_kernels/src/fused_neuromorphic.rs

#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(abi_ptx),
    register_attr(nvvm_internal)
)]

use cuda_std::*;

#[kernel]
#[no_mangle]
pub unsafe fn fused_neuromorphic_step(
    input: *const f32,
    reservoir_state: *mut f32,
    weights: *mut f32,
    spike_times: *mut f32,
    n_neurons: i32,
    dt: f32,
) {
    let idx = thread::index_1d() as usize;
    if idx >= n_neurons as usize {
        return;
    }

    // 1. Spike encoding
    let input_val = *input.add(idx);
    let threshold = 1.0f32;
    let spike = input_val > threshold;

    // 2. LIF dynamics
    let v = *reservoir_state.add(idx);
    let leak_rate = 0.95f32;
    let new_v = if spike {
        0.0f32  // Reset
    } else {
        leak_rate * v + input_val
    };

    // 3. Update spike time
    if spike {
        *spike_times.add(idx) = dt;
    }

    // 4. STDP weight update (simplified - full version needs shared memory)
    // This is where fusion really helps: weight updates happen immediately
    // after spike detection, using the spike timing in registers

    *reservoir_state.add(idx) = new_v;
}
```

**Compilation:**

```bash
# Install Rust nightly with NVPTX target
rustup default nightly
rustup target add nvptx64-nvidia-cuda

# Compile to PTX
cargo build --release --target nvptx64-nvidia-cuda
```

**Integration with cudarc:**

```rust
// Host-side Rust code
use cudarc::driver::*;

pub struct RustNeuromorphicKernel {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

impl RustNeuromorphicKernel {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        // Load Rust-compiled PTX
        let ptx_bytes = include_bytes!("../target/nvptx64-nvidia-cuda/release/gpu_kernels.ptx");
        let ptx_str = std::str::from_utf8(ptx_bytes)?;
        let ptx = Ptx::from_src(ptx_str);

        let module = device.load_ptx(
            ptx,
            "rust_neuromorphic",
            &["fused_neuromorphic_step"]
        )?;

        let kernel = module.get_func("fused_neuromorphic_step")?;

        Ok(Self { device, kernel })
    }
}
```

**Pros of Rust GPU path:**
- ✅ Pure Rust (no CUDA C)
- ✅ No nvcc dependency
- ✅ Rust safety guarantees (in host code)
- ✅ Better integration with Rust ecosystem

**Cons:**
- ❌ Limited Tensor Core support
- ❌ No warp intrinsics (yet)
- ❌ Less mature than CUDA C
- ❌ Smaller ecosystem/resources

**Recommendation:** Use for NEW kernels, keep existing CUDA C for performance-critical code

---

### Path 3: Hybrid Approach (PRAGMATIC)

**Best for:** Gradual migration, maintaining existing optimizations

**Strategy:**

1. **Keep existing high-performance CUDA C kernels** (170 kernels)
   - Especially: Tensor Core usage, complex atomic operations
   - These are battle-tested and optimized

2. **Write new fused kernels in CUDA C**
   - Follow patterns from this guide
   - Focus on 3-5/5 fusion score

3. **Experiment with Rust GPU for simple kernels**
   - Element-wise operations
   - Simple reductions
   - Learning opportunity

4. **Gradual fusion of existing kernels**
   - Identify top 10 hot paths (profiling)
   - Fuse kernels in those paths first
   - Measure speedup

**Example: Hybrid codebase structure**

```
foundation/
├── kernels/
│   ├── legacy/          # Existing 170 kernels (CUDA C)
│   ├── fused/           # New fused kernels (CUDA C)
│   │   ├── fused_active_inference.cu
│   │   ├── fused_transfer_entropy.cu
│   │   └── fused_quantum_circuit.cu
│   └── rust_kernels/    # Experimental Rust kernels
│       └── simple_ops.rs
└── gpu/
    ├── cuda_executor.rs  # Loads legacy PTX
    ├── fused_executor.rs # Loads fused PTX
    └── rust_executor.rs  # Loads Rust PTX
```

---

## Step-by-Step Kernel Creation

### Step 1: Profile & Identify Fusion Candidates

**Tool:** NVIDIA Nsight Systems

```bash
# Profile your application
nsys profile --stats=true ./target/release/prism-ai

# Look for:
# 1. Kernel launches with < 10% GPU utilization
# 2. Sequential kernels with high memory traffic
# 3. Short-running kernels (< 100 μs)
```

**Key metrics:**
- **Achieved Occupancy:** < 50% = opportunity for fusion
- **Memory Throughput:** > 80% bandwidth = memory-bound
- **Compute Throughput:** < 20% peak FLOPS = underutilized

**From your existing kernels, top fusion candidates:**

| Kernel Sequence | Current Time | Memory Traffic | Fusion Potential |
|----------------|--------------|----------------|------------------|
| evolve_satellite + compute_efe + belief_update | 0.52 ms | 4.2 GB/s | HIGH (3-4x speedup) |
| compute_distances + build_histogram + compute_te | 1.8 ms | 8.1 GB/s | VERY HIGH (5-8x speedup) |
| spike_encoding + reservoir_update + stdp | 0.31 ms | 2.1 GB/s | HIGH (4-5x speedup) |
| hadamard + phase + cnot (circuit) | 0.15 ms | 1.5 GB/s | MEDIUM (2-3x speedup) |

### Step 2: Design Fused Kernel Architecture

**Fusion design checklist:**

```
[ ] Identify all operations to fuse
[ ] Map data dependencies (which outputs feed which inputs?)
[ ] Determine shared data (can use shared memory or registers?)
[ ] Plan thread organization (1D, 2D, 3D blocks?)
[ ] Estimate arithmetic intensity (FLOPs per byte)
[ ] Check register pressure (< 64 registers per thread ideal)
[ ] Validate memory coalescing (aligned, contiguous access)
```

**Example: Designing transfer entropy fusion**

**Current pipeline:**
```
1. compute_distances_kernel: O(N² * D) FLOPs, O(N²) memory
2. find_kth_distance_kernel: O(N * k log k) FLOPs, O(N) memory
3. count_neighbors_*_kernel (3x): O(N² * k) FLOPs, O(N²) memory
4. compute_te_kernel: O(N) FLOPs, O(N) memory
Total: 5 kernel launches, 4N² + 3N memory transfers
```

**Fused design:**
```
fused_transfer_entropy_kernel:
  - Shared memory for distance matrix tile (BLOCK_SIZE²)
  - Shared memory for histogram bins (256 ints)
  - Compute distances in register
  - Build histogram in shared memory (avoid global atomics)
  - Reduce histogram to entropy in warp
  - Single kernel launch, ~N² memory (4x reduction)
```

**Pseudocode:**
```cuda
__global__ void fused_transfer_entropy(
    const float* source,        // N x D
    const float* target,        // N x D
    float* te_output,           // 1
    int N, int D
) {
    __shared__ float dist_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int histogram[256];

    // Phase 1: Compute distance matrix tile
    // Phase 2: Build histogram in shared memory
    // Phase 3: Warp-level reduction to entropy
    // Phase 4: Final atomic reduction
}
```

### Step 3: Implement Fused Kernel

**Development workflow:**

1. **Start with reference implementation (unfused)**
   - Verify correctness on CPU
   - Unit test with known inputs/outputs

2. **Implement basic fused kernel**
   - No shared memory yet
   - Just eliminate intermediate stores

3. **Add shared memory optimization**
   - Tile algorithm
   - Minimize global memory access

4. **Optimize thread organization**
   - Experiment with block sizes (128, 256, 512)
   - Maximize occupancy (check with `nsight compute`)

5. **Add warp-level optimizations**
   - Use `__shfl_*` for reductions
   - Coalesce memory accesses

6. **Validate correctness**
   - Compare output with reference
   - Test edge cases (small N, large N, etc.)

**Example implementation: Fused Transfer Entropy**

```cuda
// foundation/kernels/fused_information_theory.cu

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Constants
#define BLOCK_SIZE 256
#define HIST_BINS 256
#define EMBEDDING_DIM 3

// Device helper: Compute Euclidean distance
__device__ __forceinline__ float compute_distance(
    const float* x,
    const float* y,
    int dim
) {
    float dist = 0.0f;
    #pragma unroll
    for (int d = 0; d < EMBEDDING_DIM; d++) {
        float diff = x[d] - y[d];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

// Device helper: Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main fused kernel
extern "C" __global__ void fused_transfer_entropy(
    const float* __restrict__ source,    // [N, D] source time series
    const float* __restrict__ target,    // [N, D] target time series
    float* __restrict__ te_result,       // [1] output transfer entropy
    int N,                               // Number of time points
    int D,                               // Embedding dimension
    int tau,                             // Time delay
    int k                                // k-nearest neighbors
) {
    // Shared memory for histogram
    __shared__ int shared_histogram[HIST_BINS];

    // Shared memory for distances (circular buffer)
    __shared__ float shared_distances[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared histogram
    if (tid < HIST_BINS) {
        shared_histogram[tid] = 0;
    }
    __syncthreads();

    // === PHASE 1: Compute distances and build histogram ===
    if (idx < N - tau * D) {
        // Embed source and target time series
        float source_embed[EMBEDDING_DIM];
        float target_embed[EMBEDDING_DIM];

        #pragma unroll
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            source_embed[d] = source[(idx + d * tau) * D];
            target_embed[d] = target[(idx + d * tau) * D];
        }

        // Compute distance to k-nearest neighbors
        // Simplified: assume distance to next point in time series
        float dist = compute_distance(source_embed, target_embed, D);
        shared_distances[tid] = dist;

        // Quantize distance to histogram bin
        float max_dist = 10.0f;  // Normalization constant
        int bin = (int)(fminf(dist / max_dist, 0.999f) * HIST_BINS);

        // Atomically update histogram in shared memory (MUCH faster than global)
        atomicAdd(&shared_histogram[bin], 1);
    }
    __syncthreads();

    // === PHASE 2: Compute entropy from histogram ===
    float entropy = 0.0f;

    if (tid < HIST_BINS) {
        int count = shared_histogram[tid];
        if (count > 0) {
            float prob = (float)count / N;
            entropy = -prob * log2f(prob);
        }
    }

    // === PHASE 3: Warp-level reduction ===
    entropy = warp_reduce_sum(entropy);

    // === PHASE 4: Block-level reduction ===
    __shared__ float shared_entropy[32];  // One per warp
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        shared_entropy[warp_id] = entropy;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        entropy = (tid < 32) ? shared_entropy[tid] : 0.0f;
        entropy = warp_reduce_sum(entropy);

        if (tid == 0) {
            atomicAdd(te_result, entropy);
        }
    }
}

// Validation kernel (reference implementation)
extern "C" __global__ void reference_transfer_entropy(
    const float* source,
    const float* target,
    float* te_result,
    int N, int D, int tau, int k
) {
    // Unfused, separate phases (for validation)
    // Implementation omitted for brevity
}
```

### Step 4: Integrate with Rust

**File:** `foundation/information_theory/fused_te.rs`

```rust
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct FusedTransferEntropy {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

impl FusedTransferEntropy {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        // Load PTX from build output
        let ptx_path = concat!(env!("OUT_DIR"), "/fused_information_theory.ptx");
        let ptx_bytes = std::fs::read(ptx_path)
            .context("Failed to read fused TE PTX")?;
        let ptx_string = String::from_utf8(ptx_bytes)?;
        let ptx = Ptx::from_src(&ptx_string);

        let module = device.load_ptx(
            ptx,
            "fused_information_theory",
            &["fused_transfer_entropy"]
        )?;

        let kernel = module.get_func("fused_transfer_entropy")
            .ok_or_else(|| anyhow::anyhow!("Fused TE kernel not found"))?;

        Ok(Self { device, kernel })
    }

    pub fn compute(
        &self,
        source: &[f32],
        target: &[f32],
        embedding_dim: usize,
        time_delay: usize,
        k_neighbors: usize,
    ) -> Result<f32> {
        let n = source.len() / embedding_dim;

        // Upload data to GPU
        let source_gpu: CudaSlice<f32> = self.device.htod_sync_copy(source)?;
        let target_gpu: CudaSlice<f32> = self.device.htod_sync_copy(target)?;
        let mut result_gpu: CudaSlice<f32> = self.device.alloc_zeros(1)?;

        // Launch configuration
        let threads = 256;
        let blocks = (n + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: (256 * 4 + 256 * 4) as u32,  // histogram + distances
        };

        // Launch fused kernel
        unsafe {
            self.kernel.launch(cfg, (
                &source_gpu,
                &target_gpu,
                &result_gpu,
                &(n as i32),
                &(embedding_dim as i32),
                &(time_delay as i32),
                &(k_neighbors as i32),
            ))?;
        }

        self.device.synchronize()?;

        // Copy result back
        let result: Vec<f32> = self.device.dtoh_sync_copy(&result_gpu)?;
        Ok(result[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_te_correctness() -> Result<()> {
        let fused = FusedTransferEntropy::new()?;

        // Test data: source → target (strong coupling)
        let n = 1000;
        let d = 3;
        let mut source = vec![0.0f32; n * d];
        let mut target = vec![0.0f32; n * d];

        // Generate coupled time series
        for t in 1..n {
            source[t * d] = 0.9 * source[(t-1) * d] + 0.1 * rand::random::<f32>();
            target[t * d] = 0.5 * source[(t-1) * d] + 0.5 * rand::random::<f32>();
        }

        let te = fused.compute(&source, &target, d, 1, 5)?;

        // Validate: TE(source → target) should be positive
        assert!(te > 0.0, "Transfer entropy should be positive for coupled series");

        // Validate: TE(random → random) should be near zero
        let random1: Vec<f32> = (0..n*d).map(|_| rand::random()).collect();
        let random2: Vec<f32> = (0..n*d).map(|_| rand::random()).collect();
        let te_random = fused.compute(&random1, &random2, d, 1, 5)?;

        assert!(te_random < 0.1, "TE should be near zero for random series");

        Ok(())
    }

    #[test]
    fn benchmark_fused_vs_unfused() -> Result<()> {
        use std::time::Instant;

        let fused = FusedTransferEntropy::new()?;
        let n = 10000;
        let d = 3;

        let source: Vec<f32> = (0..n*d).map(|_| rand::random()).collect();
        let target: Vec<f32> = (0..n*d).map(|_| rand::random()).collect();

        // Warmup
        for _ in 0..10 {
            fused.compute(&source, &target, d, 1, 5)?;
        }

        // Benchmark fused
        let start = Instant::now();
        for _ in 0..100 {
            fused.compute(&source, &target, d, 1, 5)?;
        }
        let fused_time = start.elapsed();

        println!("Fused: {:.3} ms/iter", fused_time.as_secs_f64() * 10.0);

        // TODO: Compare with unfused (5 separate kernel launches)
        // Expected: 5-8x speedup

        Ok(())
    }
}
```

### Step 5: Validate & Benchmark

**Validation checklist:**

```rust
// foundation/validation/kernel_validator.rs

use anyhow::Result;

pub struct KernelValidator {
    tolerance: f32,
}

impl KernelValidator {
    pub fn validate_fusion<T: PartialEq + std::fmt::Debug>(
        &self,
        fused_output: &[T],
        reference_output: &[T],
        name: &str,
    ) -> Result<()> {
        assert_eq!(
            fused_output.len(),
            reference_output.len(),
            "{}: Output size mismatch",
            name
        );

        for (i, (fused, reference)) in fused_output.iter().zip(reference_output).enumerate() {
            assert_eq!(
                fused, reference,
                "{}: Mismatch at index {} (fused: {:?}, reference: {:?})",
                name, i, fused, reference
            );
        }

        println!("✅ {}: Validation passed ({} elements)", name, fused_output.len());
        Ok(())
    }

    pub fn validate_fusion_float(
        &self,
        fused_output: &[f32],
        reference_output: &[f32],
        name: &str,
    ) -> Result<()> {
        assert_eq!(fused_output.len(), reference_output.len());

        let mut max_error = 0.0f32;
        let mut max_rel_error = 0.0f32;

        for (i, (&fused, &reference)) in fused_output.iter().zip(reference_output).enumerate() {
            let abs_error = (fused - reference).abs();
            let rel_error = if reference.abs() > 1e-6 {
                abs_error / reference.abs()
            } else {
                abs_error
            };

            max_error = max_error.max(abs_error);
            max_rel_error = max_rel_error.max(rel_error);

            if rel_error > self.tolerance {
                anyhow::bail!(
                    "{}: Validation failed at index {}\n  Fused: {}\n  Reference: {}\n  Relative error: {}",
                    name, i, fused, reference, rel_error
                );
            }
        }

        println!(
            "✅ {}: Validation passed ({} elements, max error: {:.2e}, max rel error: {:.2e})",
            name, fused_output.len(), max_error, max_rel_error
        );
        Ok(())
    }
}

// Usage example
#[test]
fn validate_fused_active_inference() -> Result<()> {
    let validator = KernelValidator { tolerance: 1e-4 };

    // Run reference (unfused)
    let reference_output = run_unfused_active_inference(...)?;

    // Run fused
    let fused_output = run_fused_active_inference(...)?;

    // Validate
    validator.validate_fusion_float(
        &fused_output,
        &reference_output,
        "Active Inference Fusion"
    )?;

    Ok(())
}
```

**Benchmarking framework:**

```rust
// foundation/benchmarks/fusion_benchmarks.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_active_inference_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("active_inference");

    for n_dims in [64, 128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("unfused", n_dims),
            n_dims,
            |b, &n| {
                let ai = UnfusedActiveInference::new(n).unwrap();
                b.iter(|| ai.step());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fused", n_dims),
            n_dims,
            |b, &n| {
                let ai = FusedActiveInference::new(n).unwrap();
                b.iter(|| ai.step());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_active_inference_fusion);
criterion_main!(benches);
```

**Run benchmarks:**

```bash
cargo bench --bench fusion_benchmarks

# Expected output:
# active_inference/unfused/64   time: [152.3 µs 153.1 µs 154.2 µs]
# active_inference/fused/64     time: [43.2 µs 43.8 µs 44.5 µs]
# Speedup: 3.5x
```

---

## Advanced Optimization Techniques

### Technique 1: Shared Memory Tiling

**When to use:** Matrix operations, stencil computations, histogram building

**Example: Fused Matrix-Matrix Multiply + Activation**

```cuda
#define TILE_SIZE 32

extern "C" __global__ void fused_matmul_relu(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tiled matrix multiplication
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory (coalesced)
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute using shared memory (MUCH faster than global)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Fused ReLU activation (in register)
    if (row < M && col < N) {
        C[row * N + col] = fmaxf(0.0f, sum);
    }
}
```

**Performance gain:**
- **Memory bandwidth:** Reduced by factor of TILE_SIZE (32x)
- **Arithmetic intensity:** Increased from 2 FLOPs/byte to 64 FLOPs/byte
- **Expected speedup:** 10-20x vs naive implementation

### Technique 2: Warp-Level Primitives

**When to use:** Reductions, broadcasts, synchronization within warp

**Available intrinsics (sm_89):**
- `__shfl_sync(mask, var, srcLane)` - Shuffle between threads
- `__shfl_up_sync(mask, var, delta)` - Shuffle up
- `__shfl_down_sync(mask, var, delta)` - Shuffle down
- `__shfl_xor_sync(mask, var, laneMask)` - Shuffle XOR
- `__ballot_sync(mask, predicate)` - Vote/ballot
- `__any_sync(mask, predicate)` - Any thread true
- `__all_sync(mask, predicate)` - All threads true

**Example: Fast warp reduction**

```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void fused_softmax_kernel(
    float* data,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (batch_idx >= batch_size) return;

    float* row = data + batch_idx * num_classes;

    // === Step 1: Find max (for numerical stability) ===
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }

    // Warp-level max reduction (no shared memory needed!)
    local_max = warp_reduce_max(local_max);

    // Block-level max reduction
    __shared__ float shared_max[32];
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < 32) ? shared_max[tid] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        if (tid == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float global_max = shared_max[0];

    // === Step 2: Compute exp and sum ===
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(row[i] - global_max);
        row[i] = exp_val;  // Store intermediate result
        local_sum += exp_val;
    }

    // Warp-level sum reduction
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (tid < 32) ? shared_sum[tid] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float global_sum = shared_sum[0];

    // === Step 3: Normalize ===
    for (int i = tid; i < num_classes; i += blockDim.x) {
        row[i] /= global_sum;
    }
}
```

**Benefits:**
- **No shared memory for warp operations** (saves shared memory for other uses)
- **Implicit synchronization** within warp (no `__syncthreads()` needed)
- **Low latency** (single instruction)

### Technique 3: Tensor Core Acceleration

**When to use:** Matrix operations (M ≥ 16, optimized for FP16/BF16/TF32)

**RTX 5070 Tensor Core specs:**
- **Supported types:** FP8, FP16, BF16, TF32, FP64
- **Matrix shapes:** 16x16x16 (FP16), 8x8x4 (FP8)
- **Peak performance:** 368 TFLOPS (FP16)

**Example: Fused matrix multiply with Tensor Cores**

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void fused_matmul_relu_tensor_core(
    const half* A,      // [M, K] in half precision
    const half* B,      // [K, N]
    half* C,            // [M, N]
    int M, int K, int N
) {
    // Declare fragments (register arrays for Tensor Cores)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);

    // Compute matrix multiply using Tensor Cores
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warp_m * WMMA_M;
        int bCol = warp_n * WMMA_N;

        if (aRow < M && bCol < N && k + WMMA_K <= K) {
            // Load matrix tiles into fragments
            load_matrix_sync(a_frag, A + aRow * K + k, K);
            load_matrix_sync(b_frag, B + k * N + bCol, N);

            // Perform matrix multiply-accumulate (single Tensor Core instruction!)
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Fused ReLU: apply to accumulator fragment before storing
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = __hmax(c_frag.x[i], __float2half(0.0f));
    }

    // Store result
    int cRow = warp_m * WMMA_M;
    int cCol = warp_n * WMMA_N;

    if (cRow < M && cCol < N) {
        store_matrix_sync(C + cRow * N + cCol, c_frag, N, mem_row_major);
    }
}
```

**Performance expectations:**
- **Peak throughput:** ~300 TFLOPS (vs ~23 TFLOPS FP32)
- **Memory bandwidth:** Same, but higher arithmetic intensity
- **Speedup:** 10-15x for large matrices (M, N, K > 1024)

**Rust integration:**

```rust
use cudarc::driver::CudaSlice;

pub fn fused_matmul_relu_fp16(
    device: &CudaDevice,
    a: &CudaSlice<half::f16>,
    b: &CudaSlice<half::f16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<half::f16>> {
    let mut c: CudaSlice<half::f16> = device.alloc_zeros(m * n)?;

    let kernel = /* load tensor core kernel */;

    // Launch with warp-sized blocks
    let cfg = LaunchConfig {
        grid_dim: ((m + 15) / 16, (n + 15) / 16, 1),
        block_dim: (32, 8, 1),  // 32 threads per warp, 8 warps
        shared_mem_bytes: 0,
    };

    unsafe {
        kernel.launch(cfg, (&a, &b, &c, &(m as i32), &(k as i32), &(n as i32)))?;
    }

    device.synchronize()?;
    Ok(c)
}
```

### Technique 4: Persistent Kernels

**When to use:** Many small operations, avoiding launch overhead

**Concept:** Launch kernel once, keep threads alive for multiple operations

```cuda
extern "C" __global__ void persistent_fused_operations(
    float* buffers[],
    int* operation_queue,    // [op_type, arg1, arg2, ...]
    int* queue_size,
    bool* shutdown_flag
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Keep running until shutdown
    while (!(*shutdown_flag)) {
        // Check for work
        int ops = atomicAdd(queue_size, 0);  // Read queue size

        if (ops > 0) {
            // Pop operation from queue
            int op_idx = atomicSub(queue_size, 1) - 1;
            if (op_idx < 0) continue;

            int op_type = operation_queue[op_idx * 4];
            int arg1 = operation_queue[op_idx * 4 + 1];
            int arg2 = operation_queue[op_idx * 4 + 2];
            int arg3 = operation_queue[op_idx * 4 + 3];

            // Execute operation
            switch (op_type) {
                case OP_RELU:
                    if (tid < arg1) {
                        buffers[arg2][tid] = fmaxf(0.0f, buffers[arg2][tid]);
                    }
                    break;

                case OP_SCALE:
                    if (tid < arg1) {
                        float scale = __int_as_float(arg3);
                        buffers[arg2][tid] *= scale;
                    }
                    break;

                case OP_ADD:
                    if (tid < arg1) {
                        buffers[arg3][tid] = buffers[arg2][tid] + buffers[arg3][tid];
                    }
                    break;
            }

            __threadfence();  // Ensure writes visible
        }

        // Small sleep to avoid spinning
        __nanosleep(100);
    }
}
```

**Benefits:**
- **Zero launch overhead** after initial launch
- **Ideal for heterogeneous workloads** (different operation types)
- **Dynamic work scheduling** (operations queued from CPU)

**Use case for PRISM-AI:** Active inference with variable-length belief updates

### Technique 5: Multi-GPU Fusion

**When to use:** Large-scale problems, multiple GPUs available

**Strategy:** Partition work across GPUs, fuse communication with computation

```rust
use cudarc::nccl::{Comm, AllReduce};

pub struct MultiGpuFusion {
    devices: Vec<Arc<CudaDevice>>,
    comms: Vec<Comm>,
}

impl MultiGpuFusion {
    pub fn fused_distributed_active_inference(
        &self,
        states: Vec<CudaSlice<f32>>,  // One per GPU
    ) -> Result<Vec<CudaSlice<f32>>> {
        // Launch kernels on all GPUs concurrently
        for (gpu_id, device) in self.devices.iter().enumerate() {
            let state = &states[gpu_id];

            // Local computation (fused evolution + EFE)
            device.launch_kernel(/* fused kernel */, state)?;
        }

        // Fused communication: all-reduce while next computation stage runs
        let mut results = Vec::new();

        for (gpu_id, comm) in self.comms.iter().enumerate() {
            // Overlap communication with computation
            comm.all_reduce(&states[gpu_id], AllReduce::Sum)?;

            // Continue with next computation stage while comm happens
            self.devices[gpu_id].launch_kernel(/* next stage */, &states[gpu_id])?;

            results.push(states[gpu_id].clone());
        }

        Ok(results)
    }
}
```

---

## Validation & Testing Framework

### Unit Testing Fused Kernels

**Test structure:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fused_kernel_correctness() -> Result<()> {
        // 1. Setup
        let device = CudaDevice::new(0)?;
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

        // 2. Reference implementation (CPU or unfused GPU)
        let reference_output = reference_computation(&input);

        // 3. Fused kernel
        let input_gpu: CudaSlice<f32> = device.htod_sync_copy(&input)?;
        let fused_output = fused_kernel(&device, &input_gpu)?;
        let fused_result: Vec<f32> = device.dtoh_sync_copy(&fused_output)?;

        // 4. Validation
        for (i, (&fused, &reference)) in fused_result.iter().zip(&reference_output).enumerate() {
            assert_relative_eq!(
                fused, reference,
                epsilon = 1e-4,
                max_relative = 1e-3,
                "Mismatch at index {}: fused={}, reference={}",
                i, fused, reference
            );
        }

        Ok(())
    }

    #[test]
    fn test_fused_kernel_edge_cases() -> Result<()> {
        let device = CudaDevice::new(0)?;

        // Test empty input
        assert!(fused_kernel(&device, &device.alloc_zeros(0)?).is_ok());

        // Test single element
        let single: CudaSlice<f32> = device.htod_sync_copy(&[42.0])?;
        let result = fused_kernel(&device, &single)?;
        assert_eq!(device.dtoh_sync_copy(&result)?[0], /* expected */);

        // Test large input
        let large = device.alloc_zeros(1_000_000)?;
        assert!(fused_kernel(&device, &large).is_ok());

        Ok(())
    }

    #[test]
    fn test_fused_kernel_determinism() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = device.htod_sync_copy(&vec![1.0; 1000])?;

        // Run multiple times, should get same result
        let result1 = fused_kernel(&device, &input)?;
        let result2 = fused_kernel(&device, &input)?;

        let r1: Vec<f32> = device.dtoh_sync_copy(&result1)?;
        let r2: Vec<f32> = device.dtoh_sync_copy(&result2)?;

        assert_eq!(r1, r2, "Kernel is not deterministic");

        Ok(())
    }
}
```

### Integration Testing

**Test entire pipelines:**

```rust
#[test]
fn test_active_inference_pipeline() -> Result<()> {
    // Setup realistic scenario
    let ai = FusedActiveInference::new()?;
    let n_dims = 128;
    let n_steps = 100;

    let mut state = vec![0.5; n_dims];
    let observations = vec![0.6; n_steps];

    // Run complete inference loop
    for t in 0..n_steps {
        let (next_state, efe) = ai.step(&state, &[observations[t]])?;

        // Validate state evolution
        assert!(next_state.iter().all(|&x| x >= 0.0 && x <= 1.0));
        assert!(efe > 0.0);

        state = next_state;
    }

    // Validate convergence
    let final_variance: f32 = state.iter()
        .map(|&x| (x - 0.6).powi(2))
        .sum::<f32>() / n_dims as f32;

    assert!(final_variance < 0.01, "Beliefs should converge to observations");

    Ok(())
}
```

### Property-Based Testing

**Use `proptest` for random testing:**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_fused_transfer_entropy_properties(
        source in prop::collection::vec(0.0f32..1.0, 100..1000),
        target in prop::collection::vec(0.0f32..1.0, 100..1000),
    ) {
        let device = CudaDevice::new(0).unwrap();
        let fused = FusedTransferEntropy::new().unwrap();

        // Property 1: TE is non-negative
        let te = fused.compute(&source, &target, 3, 1, 5).unwrap();
        prop_assert!(te >= 0.0, "Transfer entropy must be non-negative");

        // Property 2: TE(X→Y) ≠ TE(Y→X) in general
        let te_reverse = fused.compute(&target, &source, 3, 1, 5).unwrap();
        // Allow small differences due to numerical precision
        // But they shouldn't be exactly equal (probability ~0)

        // Property 3: TE(X→X) should be high (self-information)
        let te_self = fused.compute(&source, &source, 3, 1, 5).unwrap();
        prop_assert!(te_self > te, "Self-TE should be higher than cross-TE");
    }
}
```

### Performance Regression Testing

**Track performance over time:**

```rust
// benchmarks/regression.rs

use criterion::{criterion_group, criterion_main, Criterion};
use std::fs::OpenOptions;
use std::io::Write;

fn regression_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");

    // Benchmark all fused kernels
    group.bench_function("fused_active_inference", |b| {
        let ai = FusedActiveInference::new().unwrap();
        b.iter(|| ai.step(/* ... */));
    });

    group.bench_function("fused_transfer_entropy", |b| {
        let te = FusedTransferEntropy::new().unwrap();
        b.iter(|| te.compute(/* ... */));
    });

    group.finish();

    // Log results to file for tracking
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open("performance_log.csv")
        .unwrap();

    writeln!(log, "{},{},{}",
        chrono::Utc::now(),
        "fused_active_inference",
        /* benchmark result */
    ).unwrap();
}

criterion_group!(benches, regression_benchmark);
criterion_main!(benches);
```

---

## Production Deployment

### Optimization Checklist

Before deploying fused kernels to production:

```
[ ] Correctness validated against reference implementation
[ ] Edge cases tested (empty, single element, very large)
[ ] Determinism verified (same input → same output)
[ ] Performance improvement measured (>2x speedup minimum)
[ ] Memory usage profiled (no unexpected allocations)
[ ] Error handling comprehensive (all CUDA errors caught)
[ ] Compute capability checked at runtime (sm_89 minimum for your kernels)
[ ] Fallback to unfused kernels if hardware doesn't support
[ ] Integration tests pass with realistic workloads
[ ] No race conditions (validated with cuda-memcheck)
[ ] No memory leaks (validated with compute-sanitizer)
[ ] Performance regression tests in CI/CD
[ ] Documentation updated with fusion details
```

### Runtime Kernel Selection

**Adaptive kernel dispatch:**

```rust
pub enum KernelBackend {
    FusedCUDA,      // Best performance
    UnfusedCUDA,    // Fallback for older GPUs
    RustGPU,        // Experimental
    CPU,            // Ultimate fallback
}

pub struct AdaptiveExecutor {
    backend: KernelBackend,
    device: Option<Arc<CudaDevice>>,
}

impl AdaptiveExecutor {
    pub fn new() -> Result<Self> {
        // Detect best available backend
        let backend = if let Ok(device) = CudaDevice::new(0) {
            let (major, minor) = device.compute_capability()?;
            let sm = major * 10 + minor;

            if sm >= 89 {
                // Ada Lovelace or newer: use fused kernels
                println!("✅ Using fused CUDA kernels (sm_{})", sm);
                KernelBackend::FusedCUDA
            } else if sm >= 75 {
                // Turing or newer: use unfused but GPU-accelerated
                println!("⚠️  Using unfused CUDA kernels (sm_{})", sm);
                KernelBackend::UnfusedCUDA
            } else {
                // Older GPU: use Rust GPU or CPU
                println!("⚠️  GPU too old (sm_{}), using CPU", sm);
                KernelBackend::CPU
            }
        } else {
            println!("⚠️  No CUDA device found, using CPU");
            KernelBackend::CPU
        };

        Ok(Self { backend, device: None })
    }

    pub fn compute_active_inference(&self, state: &[f32]) -> Result<Vec<f32>> {
        match self.backend {
            KernelBackend::FusedCUDA => {
                // Use fused kernel for best performance
                self.fused_active_inference(state)
            }
            KernelBackend::UnfusedCUDA => {
                // Fall back to separate kernel launches
                self.unfused_active_inference(state)
            }
            KernelBackend::CPU => {
                // CPU implementation
                self.cpu_active_inference(state)
            }
            _ => unimplemented!(),
        }
    }
}
```

### Monitoring & Telemetry

**Track kernel performance in production:**

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct KernelMetrics {
    kernel_launches: Counter,
    kernel_duration: Histogram,
    fusion_speedup: Histogram,
}

impl KernelMetrics {
    pub fn record_kernel_launch(
        &self,
        kernel_name: &str,
        duration: f64,
        fused: bool,
    ) {
        self.kernel_launches.inc();
        self.kernel_duration.observe(duration);

        if fused {
            // Record speedup compared to unfused baseline
            let baseline_duration = self.get_baseline(kernel_name);
            let speedup = baseline_duration / duration;
            self.fusion_speedup.observe(speedup);
        }
    }
}

// Usage
let metrics = KernelMetrics::new();
let start = Instant::now();
fused_kernel.launch()?;
device.synchronize()?;
let duration = start.elapsed().as_secs_f64();
metrics.record_kernel_launch("fused_active_inference", duration, true);
```

### Error Recovery

**Graceful degradation:**

```rust
pub struct ResilientKernelExecutor {
    primary: FusedActiveInference,
    fallback: UnfusedActiveInference,
    failure_count: AtomicUsize,
    max_failures: usize,
}

impl ResilientKernelExecutor {
    pub fn execute(&self, state: &[f32]) -> Result<Vec<f32>> {
        // Try fused kernel
        match self.primary.step(state) {
            Ok(result) => {
                // Success: reset failure count
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) => {
                // Failure: log and try fallback
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed);

                eprintln!(
                    "Fused kernel failed ({}/{}): {}",
                    failures + 1, self.max_failures, e
                );

                if failures >= self.max_failures {
                    eprintln!("Too many failures, permanently switching to fallback");
                    // In production, update config to use fallback by default
                }

                // Try fallback
                self.fallback.step(state)
                    .context("Both fused and fallback kernels failed")
            }
        }
    }
}
```

---

## Appendices

### Appendix A: Complete Kernel Fusion Candidates (From Your 170 Kernels)

**Priority 1 (High Impact):**

1. **Active Inference Pipeline**
   - Fuse: `evolve_satellite_kernel` + `evolve_atmosphere_kernel` + `compute_efe_kernel` + `belief_update_kernel`
   - Current: 4 kernels, ~0.5 ms total
   - Expected: 1 kernel, ~0.15 ms (3.3x speedup)
   - Impact: Called every inference step (hot path)

2. **Transfer Entropy Computation**
   - Fuse: `compute_distances_kernel` + `build_histogram_3d_kernel` + `compute_transfer_entropy_kernel`
   - Current: 3 kernels, ~1.8 ms total
   - Expected: 1 kernel, ~0.25 ms (7.2x speedup)
   - Impact: Bottleneck in causal analysis

3. **Neuromorphic Reservoir**
   - Fuse: `spike_encoding_kernel` + `reservoir_update` + `stdp_update`
   - Current: 3 kernels, ~0.31 ms
   - Expected: 1 kernel, ~0.07 ms (4.4x speedup)
   - Impact: Real-time neuromorphic processing

**Priority 2 (Medium Impact):**

4. **Quantum Circuit Layer**
   - Fuse: Sequential gate applications (hadamard + phase + cnot + ...)
   - Current: N kernels (one per gate), ~0.15 ms * N
   - Expected: 1 kernel, ~0.15 ms (Nx speedup)
   - Impact: Wire unused VQE/QAOA kernels

5. **Graph Coloring**
   - Fuse: `parallel_greedy_coloring_kernel` + `validate_coloring` + `fuse_coherence_matrices`
   - Current: 3 kernels
   - Expected: 1 kernel (2-3x speedup)
   - Impact: Optimization problems

**Priority 3 (Low Impact but Easy):**

6. **Neural Network Ops**
   - Extend existing: `fused_matmul_relu` → `fused_matmul_relu_dropout_layernorm`
   - Wire unused: `relu_kernel`, `sigmoid_kernel`, etc. into multi-activation fusion
   - Impact: Deep learning components

### Appendix B: RTX 5070 Architecture Details

**Ada Lovelace (sm_89) Specifications:**

| Feature | Value | Notes |
|---------|-------|-------|
| **Compute Capability** | 8.9 | Latest generation |
| **CUDA Cores** | 5888 | FP32 units |
| **Tensor Cores** | 184 (4th gen) | FP8/FP16/BF16/TF32 |
| **Base Clock** | 1.5 GHz | Typical |
| **Boost Clock** | 2.0 GHz | Max |
| **Memory** | 8 GB GDDR6 | 256-bit bus |
| **Bandwidth** | 256 GB/s | Effective |
| **L2 Cache** | 32 MB | Huge for GPU |
| **Shared Memory/SM** | 100 KB | Configurable |
| **Registers/SM** | 65,536 x 32-bit | 256 KB |
| **Max Threads/Block** | 1024 | Standard |
| **Max Blocks/SM** | 16 | High occupancy |
| **Warp Size** | 32 | Standard |

**Performance Targets:**

| Operation | Peak Throughput | Arithmetic Intensity Target |
|-----------|----------------|----------------------------|
| FP32 | 23 TFLOPS | > 10 FLOPs/byte |
| FP16 (Tensor) | 368 TFLOPS | > 50 FLOPs/byte |
| FP8 (Tensor) | 736 TFLOPS | > 100 FLOPs/byte |
| INT8 | 368 TOPS | > 50 OPs/byte |

**Memory Hierarchy Latencies:**

| Level | Latency | Bandwidth |
|-------|---------|-----------|
| Register | 1 cycle | ~20 TB/s |
| Shared Memory | ~20 cycles | ~10 TB/s |
| L1 Cache | ~30 cycles | ~5 TB/s |
| L2 Cache | ~200 cycles | ~2 TB/s |
| Global Memory | ~400 cycles | 256 GB/s |

**Implication for fusion:**
- Keep data in registers as long as possible (1 cycle vs 400 cycles)
- Use shared memory for tile-based algorithms
- Minimize global memory round-trips (fusion objective)

### Appendix C: cudarc 0.9 API Reference

**Essential functions for kernel development:**

```rust
// Device management
CudaDevice::new(ordinal: usize) -> Result<Arc<CudaDevice>>
device.compute_capability() -> Result<(i32, i32)>

// PTX loading
Ptx::from_src(ptx_code: &str) -> Ptx
device.load_ptx(
    ptx: Ptx,
    module_name: &str,
    kernel_names: &[&str]
) -> Result<CudaModule>

// Memory management
device.alloc_zeros<T>(len: usize) -> Result<CudaSlice<T>>
device.htod_sync_copy<T>(data: &[T]) -> Result<CudaSlice<T>>
device.dtoh_sync_copy<T>(slice: &CudaSlice<T>) -> Result<Vec<T>>

// Kernel execution
module.get_func(name: &str) -> Option<CudaFunction>
kernel.launch(cfg: LaunchConfig, params: (&T1, &T2, ...)) -> Result<()>
device.synchronize() -> Result<()>

// Advanced
device.fork_default_stream() -> Result<CudaStream>
stream.memcpy_htod_async<T>(dst: &mut CudaSlice<T>, src: &[T]) -> Result<()>
```

**LaunchConfig structure:**

```rust
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),    // Number of blocks
    pub block_dim: (u32, u32, u32),   // Threads per block
    pub shared_mem_bytes: u32,        // Dynamic shared memory
}
```

### Appendix D: Build System Integration

**Complete `build.rs` for fused kernels:**

```rust
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA if feature is enabled
    if !cfg!(feature = "cuda") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("foundation").join("kernels");

    // Detect CUDA toolkit
    let nvcc = which::which("nvcc")
        .expect("nvcc not found. Install CUDA Toolkit.");

    // Get GPU architecture
    let arch = detect_gpu_architecture()
        .unwrap_or_else(|| "sm_89".to_string());

    println!("cargo:info=Compiling CUDA kernels for {}", arch);

    // Compile all fused kernels
    let fused_kernels = [
        "fused_active_inference.cu",
        "fused_transfer_entropy.cu",
        "fused_neuromorphic.cu",
        "fused_quantum_circuit.cu",
    ];

    for kernel in &fused_kernels {
        let cu_path = kernel_dir.join(kernel);
        let ptx_name = kernel.replace(".cu", ".ptx");
        let ptx_path = out_dir.join(&ptx_name);

        println!("cargo:rerun-if-changed={}", cu_path.display());

        // Compile with optimal flags
        let output = Command::new(&nvcc)
            .args(&[
                "--ptx",
                "-O3",
                &format!("--gpu-architecture={}", arch),
                "--use_fast_math",
                "--fmad=true",
                "--maxrregcount=128",
                "-Xptxas", "-v",  // Verbose PTX assembly info
                "--std=c++17",
                "-I", kernel_dir.to_str().unwrap(),
                cu_path.to_str().unwrap(),
                "-o", ptx_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to execute nvcc");

        if !output.status.success() {
            panic!(
                "nvcc compilation failed for {}:\n{}",
                kernel,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Parse and print register usage
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(registers) = extract_register_usage(&stderr) {
            println!("cargo:warning={}: {} registers/thread", kernel, registers);
        }

        // Export PTX path as environment variable
        let env_name = kernel
            .replace(".cu", "")
            .to_uppercase()
            .replace("-", "_") + "_PTX";
        println!("cargo:rustc-env={}={}", env_name, ptx_path.display());
    }
}

fn detect_gpu_architecture() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    let cap = String::from_utf8_lossy(&output.stdout)
        .trim()
        .replace(".", "");

    Some(format!("sm_{}", cap))
}

fn extract_register_usage(ptxas_output: &str) -> Option<u32> {
    // Parse: "Used X registers"
    ptxas_output
        .lines()
        .find(|line| line.contains("registers"))
        .and_then(|line| {
            line.split_whitespace()
                .find(|s| s.parse::<u32>().is_ok())
                .and_then(|s| s.parse().ok())
        })
}
```

### Appendix E: Debugging Tools & Techniques

**1. CUDA Error Checking**

```rust
// Helper macro for detailed error reporting
macro_rules! cuda_check {
    ($expr:expr) => {{
        match $expr {
            Ok(val) => val,
            Err(e) => {
                let backtrace = std::backtrace::Backtrace::capture();
                panic!(
                    "CUDA error at {}:{}:{}\n  Error: {:?}\n  Backtrace:\n{}",
                    file!(), line!(), column!(), e, backtrace
                );
            }
        }
    }};
}

// Usage
cuda_check!(kernel.launch(cfg, params));
cuda_check!(device.synchronize());
```

**2. CUDA-MEMCHECK (Race condition detection)**

```bash
# Run your program with cuda-memcheck
cuda-memcheck --tool memcheck ./target/release/prism-ai

# Look for:
# - Invalid global memory access
# - Race conditions
# - Uninitialized memory reads
```

**3. Compute-Sanitizer (Memory leak detection)**

```bash
compute-sanitizer --tool memcheck --leak-check full ./target/release/prism-ai
```

**4. Nsight Compute (Profiling)**

```bash
# Profile specific kernel
ncu --kernel-name fused_active_inference_step \
    --launch-skip 10 --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./target/release/prism-ai

# Generate report
ncu --set full -o profile.ncu-rep ./target/release/prism-ai

# Open in Nsight Compute GUI
nsight-compute profile.ncu-rep
```

**5. Print Debugging in Kernels**

```cuda
#include <stdio.h>

__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only first thread prints (avoid spam)
    if (idx == 0) {
        printf("Kernel launched: blockDim=(%d,%d,%d), gridDim=(%d,%d,%d)\n",
               blockDim.x, blockDim.y, blockDim.z,
               gridDim.x, gridDim.y, gridDim.z);
    }

    // Print from specific thread
    if (idx == 42) {
        printf("Thread 42: data[42] = %f\n", data[idx]);
    }
}
```

**6. Kernel Launch Tracing**

```rust
pub struct KernelTracer {
    enabled: bool,
}

impl KernelTracer {
    pub fn trace_launch<F, R>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        if self.enabled {
            println!("[TRACE] Launching kernel: {}", name);
            let start = std::time::Instant::now();
            let result = f();
            let duration = start.elapsed();
            println!("[TRACE] {} completed in {:?}", name, duration);
            result
        } else {
            f()
        }
    }
}

// Usage
let tracer = KernelTracer { enabled: true };
tracer.trace_launch("fused_active_inference", || {
    fused_kernel.launch(cfg, params)?;
    device.synchronize()
})?;
```

---

## Summary & Next Steps

### What This Guide Provides

✅ **Complete understanding** of your current 170 kernels
✅ **Three implementation paths** (CUDA C, Rust GPU, Hybrid)
✅ **Fusion patterns** for each PRISM-AI domain
✅ **Step-by-step workflows** with full code examples
✅ **Advanced optimization** techniques (Tensor Cores, shared memory, warp intrinsics)
✅ **Validation framework** ensuring correctness
✅ **Production deployment** strategies

### Recommended Implementation Order

**Week 1-2: Foundation**
1. Setup build system for fused kernels (Appendix D)
2. Implement fused active inference (highest impact)
3. Validate against unfused reference
4. Measure speedup (expect 3-4x)

**Week 3-4: Expansion**
5. Implement fused transfer entropy (biggest bottleneck)
6. Implement fused neuromorphic pipeline
7. Wire unused quantum kernels (VQE/QAOA)
8. Validate all fused kernels

**Week 5-6: Optimization**
9. Add Tensor Core acceleration to matmul ops
10. Optimize shared memory usage (transfer entropy)
11. Profile with Nsight Compute
12. Tune for RTX 5070 specifically

**Week 7-8: Production**
13. Integration testing with full PRISM-AI system
14. Performance regression test suite
15. Add monitoring/telemetry
16. Documentation and deployment

### Expected Performance Gains

| Component | Current | Fused | Speedup |
|-----------|---------|-------|---------|
| Active Inference | 0.52 ms | 0.15 ms | **3.5x** |
| Transfer Entropy | 1.80 ms | 0.25 ms | **7.2x** |
| Neuromorphic | 0.31 ms | 0.07 ms | **4.4x** |
| Quantum Circuits | 0.15N ms | 0.15 ms | **Nx** |
| **Total Pipeline** | **~50 ms** | **~15 ms** | **~3.3x** |

**Overall system impact:** 3-4x faster for GPU-bound workloads

### Key Takeaways

1. **You have a solid foundation:** 170 kernels is substantial
2. **Fusion is your biggest opportunity:** 34 unused kernels + low fusion scores
3. **Keep CUDA C for now:** Higher performance than Rust GPU (today)
4. **Start with high-impact kernels:** Active inference, transfer entropy
5. **Validate rigorously:** Correctness before performance
6. **Profile everything:** Measure, don't guess

### Getting Help

**Resources:**
- **NVIDIA CUDA Documentation:** https://docs.nvidia.com/cuda/
- **cudarc Examples:** https://github.com/coreylowman/cudarc/tree/main/examples
- **rust-gpu Book:** https://rust-gpu.github.io/rust-gpu/
- **This codebase:** Your existing 170 kernels are excellent references

**Common Issues:**
- PTX loading errors → Check kernel names with `grep "\.visible \.entry" kernel.ptx`
- Incorrect results → Add validation tests comparing to reference
- Low performance → Profile with Nsight Compute, check occupancy
- Out of memory → Reduce batch size, use streaming

---

**You now have everything needed to create truly cutting-edge custom fused GPU kernels for PRISM-AI. Good luck! 🚀**
