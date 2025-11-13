# Phase 3 (Quantum Coloring) - Proper GPU Implementation Plan

## Current Status

**File**: `foundation/prct-core/src/quantum_coloring.rs`

**Current Implementation** (Lines 88-113):
```rust
fn find_coloring_gpu(...) -> Result<ColoringSolution> {
    println!("[QUANTUM-GPU] Using hybrid CPU/GPU approach");
    println!("[QUANTUM-GPU] Main optimization on CPU, energy evaluation on GPU");

    // ❌ STUB: Just calls CPU implementation
    let result = self.find_coloring_cpu(...)?;

    Ok(result)
}
```

**Problem**: This is a stub that claims GPU but runs CPU.

---

## Available GPU Kernels

**File**: `foundation/kernels/quantum_evolution.cu` (494 lines)

### **Core Kernels Available**:
1. **`apply_diagonal_evolution`** - Apply potential evolution: e^(-iV*dt/ħ)
2. **`apply_kinetic_evolution_momentum`** - Apply kinetic evolution in momentum space
3. **`trotter_suzuki_step`** - Full quantum time evolution (FFI wrapper)
4. **`build_tight_binding_hamiltonian`** - Build Hamiltonian from graph edges
5. **`build_ising_hamiltonian`** - Build Ising/QUBO Hamiltonian
6. Additional kernels for measuring observables

---

## Proper Implementation Plan

### **Option A: Full Quantum Evolution on GPU** (Complex, ~20-30 hours)

This would implement true quantum annealing using Trotter-Suzuki decomposition:

**Algorithm**:
1. Convert graph coloring to QUBO/Ising model
2. Create quantum state on GPU (complex wavefunction)
3. Build Hamiltonian on GPU
4. Run quantum evolution using Trotter-Suzuki
5. Measure quantum state to extract coloring
6. Repeat with simulated annealing schedule

**Problems**:
- Requires cuFFT for Fourier transforms
- State space grows exponentially (2^n - impossible for n=1000)
- Would need clever approximations or subspace methods
- Very complex to implement correctly
- May not be faster than CPU for graph coloring

**Verdict**: ❌ **Not practical for graph coloring**

---

### **Option B: Hybrid GPU-Accelerated QUBO Solver** (Practical, ~8-12 hours)

Use GPU for expensive operations within the existing algorithm:

**What the CPU Implementation Does** (`find_coloring_cpu`):
1. Compute TDA bounds (CPU - fast)
2. Generate initial greedy solution (CPU - fast)
3. **Iterative color reduction loop** (CPU - slow):
   - Build sparse QUBO matrix (CPU)
   - Run simulated annealing (CPU - BOTTLENECK!)
   - Validate solution (CPU)
   - Repeat for multiple targets

**GPU Acceleration Strategy**:
Move the simulated annealing to GPU (the bottleneck).

**Implementation**:

#### **Files to Create**:

1. **`foundation/prct-core/src/gpu_quantum_annealing.rs`** (NEW, ~400 lines)
   - `pub fn gpu_qubo_simulated_annealing(cuda_device, qubo_matrix, num_vars, iterations) -> Result<Vec<bool>>`
   - Load `quantum_evolution.ptx`
   - Implement GPU-based simulated annealing for QUBO
   - Use sparse matrix kernels for energy evaluation
   - Return binary solution vector

#### **Files to Modify**:

2. **`foundation/prct-core/src/quantum_coloring.rs`** - Replace stub
   - **Lines to modify**: 88-113 (`find_coloring_gpu` method)
   - **Change**: Instead of calling CPU, call `gpu_quantum_annealing::gpu_qubo_simulated_annealing()`
   - **Keep**: TDA bounds computation (CPU is fine)
   - **Keep**: Initial solution generation (CPU is fine)
   - **Replace**: The annealing loop at lines 176-230 with GPU version

3. **`foundation/prct-core/src/lib.rs`** - Add module
   - **Add**: `pub mod gpu_quantum_annealing;`
   - **Add**: Re-exports if needed

4. **`foundation/kernels/quantum_evolution.cu`** - Add QUBO kernel (OPTIONAL)
   - **Add**: `__global__ void qubo_energy_kernel(...)` for sparse QUBO energy evaluation
   - **Add**: `__global__ void qubo_flip_kernel(...)` for single-variable flips
   - **Lines to add**: ~50-100 new lines for QUBO-specific operations

---

## Detailed Implementation Specification

### **1. Create `gpu_quantum_annealing.rs`**

```rust
//! GPU-Accelerated QUBO Simulated Annealing
//!
//! Implements simulated annealing for sparse QUBO problems on GPU.

use crate::errors::*;
use crate::sparse_qubo::SparseQUBO;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// GPU-based simulated annealing for QUBO optimization
///
/// # Arguments
/// * `cuda_device` - Shared CUDA context
/// * `qubo` - Sparse QUBO problem definition
/// * `initial_state` - Starting binary vector
/// * `iterations` - Number of annealing steps
/// * `t_initial` - Initial temperature
/// * `t_final` - Final temperature
///
/// # Returns
/// Best binary solution found (maps to graph coloring)
#[cfg(feature = "cuda")]
pub fn gpu_qubo_simulated_annealing(
    cuda_device: &Arc<CudaDevice>,
    qubo: &SparseQUBO,
    initial_state: &[bool],
    iterations: usize,
    t_initial: f64,
    t_final: f64,
) -> Result<Vec<bool>> {
    println!("[QUANTUM-GPU] Starting GPU QUBO annealing: {} variables, {} iterations",
             qubo.num_variables, iterations);

    // Step 1: Load PTX kernels
    let ptx = Ptx::from_file("target/ptx/quantum_evolution.ptx");
    cuda_device.load_ptx(
        ptx,
        "quantum_module",
        &["qubo_energy_kernel", "qubo_propose_flip_kernel", "qubo_accept_kernel"],
    ).map_err(|e| PRCTError::GpuError(format!("Failed to load quantum kernels: {}", e)))?;

    // Step 2: Upload QUBO to GPU (sparse format: row_indices, col_indices, values)
    let (rows, cols, vals) = qubo.to_sparse_csr();
    let d_rows = cuda_device.htod_copy(rows)?;
    let d_cols = cuda_device.htod_copy(cols)?;
    let d_vals = cuda_device.htod_copy(vals)?;

    // Step 3: Upload initial state to GPU
    let state_u8: Vec<u8> = initial_state.iter().map(|&b| b as u8).collect();
    let mut d_state = cuda_device.htod_copy(state_u8)?;

    // Step 4: Allocate GPU buffers
    let mut d_energy = cuda_device.alloc_zeros::<f64>(1)?;
    let mut d_best_state = cuda_device.htod_copy(state_u8.clone())?;
    let mut d_best_energy = cuda_device.alloc_zeros::<f64>(1)?;

    // Step 5: Get kernel function handles
    let energy_kernel = cuda_device.get_func("quantum_module", "qubo_energy_kernel")
        .ok_or_else(|| PRCTError::GpuError("qubo_energy_kernel not found".into()))?;
    let flip_kernel = cuda_device.get_func("quantum_module", "qubo_propose_flip_kernel")
        .ok_or_else(|| PRCTError::GpuError("qubo_propose_flip_kernel not found".into()))?;

    // Step 6: Run simulated annealing on GPU
    let temp_schedule: Vec<f64> = (0..iterations)
        .map(|i| {
            let frac = i as f64 / iterations as f64;
            t_initial * (t_final / t_initial).powf(frac)
        })
        .collect();

    for (iter, &temp) in temp_schedule.iter().enumerate() {
        // Launch energy evaluation kernel
        // Launch flip proposal kernel
        // Launch accept/reject kernel based on Metropolis criterion
        // Update best if improved

        if iter % 1000 == 0 {
            println!("[QUANTUM-GPU] Iteration {}/{}, T={:.4}", iter, iterations, temp);
        }
    }

    // Step 7: Download best solution
    let best_state_u8 = cuda_device.dtoh_sync_copy(&d_best_state)?;
    let best_state: Vec<bool> = best_state_u8.iter().map(|&b| b != 0).collect();

    println!("[QUANTUM-GPU] GPU annealing complete");

    Ok(best_state)
}

// Helper: Convert QUBO binary solution to graph coloring
pub fn qubo_solution_to_coloring(
    qubo_solution: &[bool],
    num_vertices: usize,
    num_colors: usize,
) -> Vec<usize> {
    // Decode binary variables: x[v*k + c] = 1 means vertex v gets color c
    let mut coloring = vec![0; num_vertices];
    for v in 0..num_vertices {
        for c in 0..num_colors {
            let var_idx = v * num_colors + c;
            if var_idx < qubo_solution.len() && qubo_solution[var_idx] {
                coloring[v] = c;
                break;
            }
        }
    }
    coloring
}
```

**Estimated lines**: ~400 lines of new code

---

### **2. Modify `quantum_coloring.rs::find_coloring_gpu()`**

**File**: `foundation/prct-core/src/quantum_coloring.rs`
**Lines to replace**: 88-113

**New implementation**:
```rust
#[cfg(feature = "cuda")]
fn find_coloring_gpu(
    &mut self,
    cuda_device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    graph: &Graph,
    _phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
    initial_estimate: usize,
) -> Result<ColoringSolution> {
    let start = std::time::Instant::now();
    let n = graph.num_vertices;

    println!("[QUANTUM-GPU] Starting GPU-accelerated quantum coloring for {} vertices", n);

    // Step 1: Compute TDA bounds (keep on CPU - fast)
    let bounds = ChromaticBounds::from_graph_tda(graph)?;
    println!("[QUANTUM-GPU] TDA bounds: [{}, {}]", bounds.lower, bounds.upper);

    // Step 2: Generate initial solution (keep on CPU - fast)
    let (initial_solution, actual_target) = self.adaptive_initial_solution(
        graph,
        _phase_field,
        kuramoto_state,
        bounds.lower,
        initial_estimate.min(bounds.upper),
    )?;

    println!("[QUANTUM-GPU] Initial greedy: {} colors (target: {})",
             initial_solution.chromatic_number, actual_target);

    let mut best_solution = initial_solution.clone();
    let mut current_target = best_solution.chromatic_number;
    let target_min = bounds.lower;

    // Step 3: Iterative color reduction with GPU QUBO annealing
    while current_target > target_min {
        let new_target = current_target.saturating_sub(1);
        if new_target < target_min {
            break;
        }

        println!("[QUANTUM-GPU] Attempting {} colors with GPU annealing", new_target);

        // Build QUBO for this target
        let qubo = SparseQUBO::from_graph_coloring(graph, new_target)?;

        // Convert current coloring to QUBO binary variables
        let mut initial_qubo_state = vec![false; n * new_target];
        for (v, &color) in best_solution.colors.iter().enumerate() {
            if color < new_target {
                initial_qubo_state[v * new_target + color] = true;
            }
        }

        // ✅ RUN GPU ANNEALING (this is the actual GPU work!)
        match gpu_quantum_annealing::gpu_qubo_simulated_annealing(
            cuda_device,
            &qubo,
            &initial_qubo_state,
            10000,  // iterations (configurable)
            1.0,    // T_initial
            0.01,   // T_final
        ) {
            Ok(qubo_solution) => {
                // Convert QUBO solution back to coloring
                let coloring = gpu_quantum_annealing::qubo_solution_to_coloring(
                    &qubo_solution,
                    n,
                    new_target,
                );

                // Validate and update if better
                let conflicts = count_conflicts_fast(graph, &coloring);
                if conflicts == 0 && new_target < best_solution.chromatic_number {
                    println!("[QUANTUM-GPU] ✅ GPU annealing found {}-coloring!", new_target);
                    best_solution = ColoringSolution {
                        colors: coloring,
                        chromatic_number: new_target,
                        conflicts: 0,
                    };
                    current_target = new_target;
                } else {
                    println!("[QUANTUM-GPU] No improvement at {} colors", new_target);
                    break;
                }
            }
            Err(e) => {
                println!("[QUANTUM-GPU][FALLBACK] GPU annealing failed: {}", e);
                break;  // Fall back to best found so far
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("[QUANTUM-GPU] GPU quantum coloring completed in {:.2}ms", elapsed);

    Ok(best_solution)
}
```

**Estimated changes**: ~80 lines modified

---

### **3. Add QUBO Kernels to `quantum_evolution.cu`**

**File**: `foundation/kernels/quantum_evolution.cu`
**Lines to add**: After line 494 (append to file)

```cuda
// ============================================================================
// QUBO Simulated Annealing Kernels
// ============================================================================

// Kernel: Evaluate QUBO energy for sparse Q matrix
// E(x) = x^T Q x = Σ_ij Q[i,j] * x[i] * x[j]
__global__ void qubo_energy_kernel(
    const int* row_ptr,      // CSR format: row pointers
    const int* col_idx,      // CSR format: column indices
    const double* values,    // CSR format: QUBO matrix values
    const unsigned char* x,  // Binary solution vector
    double* energy_out,      // Output: total energy
    int num_vars
) {
    __shared__ double shared_energy[256];

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    shared_energy[tid] = 0.0;

    if (row < num_vars && x[row] == 1) {
        // Sum Q[row, col] * x[col] for all cols in this row
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int col = col_idx[idx];
            if (x[col] == 1) {
                shared_energy[tid] += values[idx];
            }
        }
    }

    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_energy[tid] += shared_energy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(energy_out, shared_energy[0]);
    }
}

// Kernel: Propose single-variable flip and compute delta energy
__global__ void qubo_propose_flip_kernel(
    const int* row_ptr,
    const int* col_idx,
    const double* values,
    const unsigned char* x_current,
    unsigned char* x_proposed,
    double* delta_energy,
    int flip_var,
    int num_vars
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy current state
    if (tid < num_vars) {
        x_proposed[tid] = x_current[tid];
    }
    __syncthreads();

    // Thread 0 flips the variable
    if (tid == 0) {
        x_proposed[flip_var] = 1 - x_current[flip_var];

        // Compute delta energy (only terms involving flip_var)
        double delta = 0.0;
        int row_start = row_ptr[flip_var];
        int row_end = row_ptr[flip_var + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int col = col_idx[idx];
            double q_val = values[idx];

            if (x_current[flip_var] == 1) {
                // Flipping from 1 to 0: remove this term
                if (x_current[col] == 1) {
                    delta -= q_val;
                }
            } else {
                // Flipping from 0 to 1: add this term
                if (x_proposed[col] == 1) {
                    delta += q_val;
                }
            }
        }

        *delta_energy = delta;
    }
}

// Kernel: Accept or reject flip based on Metropolis criterion
__global__ void qubo_metropolis_accept_kernel(
    unsigned char* x_current,
    const unsigned char* x_proposed,
    double delta_energy,
    double temperature,
    double random_value,
    int num_vars
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Metropolis acceptance
    bool accept = false;
    if (delta_energy < 0.0) {
        accept = true;  // Always accept improvements
    } else if (temperature > 1e-10) {
        double prob = exp(-delta_energy / temperature);
        accept = (random_value < prob);
    }

    // Update state if accepted
    if (accept && tid < num_vars) {
        x_current[tid] = x_proposed[tid];
    }
}
```

**Estimated lines**: ~150 new CUDA lines

---

### **4. Update `lib.rs`**

**File**: `foundation/prct-core/src/lib.rs`
**Lines to add**: After other GPU modules

```rust
#[cfg(feature = "cuda")]
pub mod gpu_quantum_annealing;
```

---

## Alternative: Simpler GPU Acceleration

### **Option C: GPU Energy Evaluation Only** (Simplest, ~4-6 hours)

Keep the entire algorithm on CPU but accelerate just the energy evaluations:

**Files to modify**: Only `quantum_coloring.rs`

**Implementation**:
```rust
fn find_coloring_gpu(...) -> Result<ColoringSolution> {
    // Keep all CPU logic
    // But when evaluating QUBO energy, use GPU kernel

    // Replace CPU energy evaluation loops with:
    let energy = evaluate_qubo_energy_gpu(cuda_device, &qubo, &state)?;

    // Everything else stays CPU
}
```

**Pros**:
- Minimal changes
- Low risk
- Still gets GPU speedup on bottleneck

**Cons**:
- Not "full" GPU (hybrid approach)
- Smaller speedup (2-3x vs 5-10x for full GPU)

---

## Recommendation

### **What I Propose for prism-gpu-orchestrator**:

**Implement Option B: Hybrid GPU-Accelerated QUBO Solver**

**Rationale**:
- Practical for graph coloring (doesn't require exponential state space)
- Leverages existing sparse QUBO infrastructure
- GPU accelerates the actual bottleneck (simulated annealing)
- Can achieve 3-10x speedup
- Constitutional compliance maintained

**Files to Create/Modify**:
1. **CREATE**: `foundation/prct-core/src/gpu_quantum_annealing.rs` (~400 lines)
2. **MODIFY**: `foundation/prct-core/src/quantum_coloring.rs` lines 88-113 (~80 lines)
3. **MODIFY**: `foundation/kernels/quantum_evolution.cu` add QUBO kernels (~150 lines)
4. **MODIFY**: `foundation/prct-core/src/lib.rs` add module export (1 line)

**Total effort**: ~8-12 hours for proper implementation

**Expected speedup**: 3-10x vs current CPU quantum solver

---

## Specific Task for prism-gpu-orchestrator

```
Implement full GPU acceleration for Phase 3 (Quantum Coloring) using sparse QUBO simulated annealing on GPU.

FILES TO CREATE:
- foundation/prct-core/src/gpu_quantum_annealing.rs

FILES TO MODIFY:
- foundation/prct-core/src/quantum_coloring.rs (lines 88-113: replace stub)
- foundation/kernels/quantum_evolution.cu (append QUBO kernels)
- foundation/prct-core/src/lib.rs (add module)

IMPLEMENTATION:
1. In quantum_evolution.cu, add 3 kernels:
   - qubo_energy_kernel (sparse CSR energy evaluation)
   - qubo_propose_flip_kernel (single-variable flip delta energy)
   - qubo_metropolis_accept_kernel (Metropolis acceptance)

2. In gpu_quantum_annealing.rs, implement:
   - gpu_qubo_simulated_annealing() - Main GPU SA loop
   - qubo_solution_to_coloring() - Decode binary to colors
   - Proper error handling, no stubs

3. In quantum_coloring.rs, replace find_coloring_gpu() to:
   - Keep TDA bounds computation (CPU)
   - Keep initial solution (CPU)
   - REPLACE annealing loop with gpu_qubo_simulated_annealing()
   - Decode GPU result to ColoringSolution

STANDARDS:
- Single Arc<CudaDevice> (passed as parameter)
- Sparse CSR format for QUBO matrix
- Proper error handling (PRCTError::GpuError)
- CPU fallback if GPU fails
- Accurate logging (only claim GPU when actually used)
- No stubs/unwraps in production code

TEST:
Verify GPU usage with nvidia-smi when Phase 3 executes.
```

---

## Summary

**Current Phase 3**: ❌ Stub that calls CPU

**Proper Implementation Needs**:
- New GPU module: `gpu_quantum_annealing.rs` (~400 lines)
- Modified quantum solver: Replace lines 88-113 in `quantum_coloring.rs`
- New CUDA kernels: 3 QUBO-specific kernels in `quantum_evolution.cu`
- Module export: 1 line in `lib.rs`

**Effort**: 8-12 hours
**Speedup**: 3-10x

**Should the agent do this?** Your call - it's substantial work but achievable.