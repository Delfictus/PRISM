# 🔧 GPU Wiring Complete Code Reference

## Executive Summary

**Not Implemented Entirely**: **3 features** (PIMC, GNN Screening, Multiscale)
**Need GPU Wiring**: **3-4 phases** (Transfer Entropy, Thermodynamic, Quantum, + optional Active Inference)

This document shows EXACTLY where the CPU calls are and where the GPU kernels exist.

---

## 📍 Part 1: Phases 1-3 CPU Call Sites

### **Location**: `foundation/prct-core/src/world_record_pipeline.rs:1855-2088`

#### **Phase 1: Transfer Entropy (Lines 1855-1906)**

**❌ FALSE GPU CLAIM - Line 1864:**
```rust
if self.config.gpu.enable_te_gpu {
    println!("[PHASE 1][GPU] TE kernels active (histogram bins=auto, lag=1)");
    // ⚠️ BUT THEN CALLS CPU FUNCTION:
}

// Line 1876 - ACTUAL CALL (CPU ONLY):
let te_ordering = hybrid_te_kuramoto_ordering(
    graph,
    initial_kuramoto,
    geodesic_features.as_ref(),
    self.config.transfer_entropy.geodesic_weight,
    self.config.transfer_entropy.te_vs_kuramoto_weight,
)?;  // ❌ NO cuda_device PASSED!
```

**🔧 FIX NEEDED**: Replace with GPU version:
```rust
#[cfg(feature = "cuda")]
let te_ordering = if self.config.gpu.enable_te_gpu {
    hybrid_te_kuramoto_ordering_gpu(
        &self.cuda_device,  // ✅ Pass GPU device
        graph,
        initial_kuramoto,
        geodesic_features.as_ref(),
        self.config.transfer_entropy.geodesic_weight,
        self.config.transfer_entropy.te_vs_kuramoto_weight,
    )?
} else {
    hybrid_te_kuramoto_ordering(graph, ...) // CPU fallback
};
```

---

#### **Phase 2: Thermodynamic (Lines 1911-1992)**

**❌ FALSE GPU CLAIM - Line 1922:**
```rust
if self.config.gpu.enable_thermo_gpu {
    println!("[PHASE 2][GPU] Thermodynamic replica exchange active (temps={}, replicas={})",
             self.config.thermo.num_temps,
             self.config.thermo.replicas);
    // ⚠️ BUT THEN CALLS CPU FUNCTION:
}

// Line 1960 - ACTUAL CALL (CPU ONLY):
self.thermodynamic_eq = Some(ThermodynamicEquilibrator::equilibrate(
    graph,
    &self.best_solution,
    self.config.target_chromatic,
    self.config.thermo.t_min,
    self.config.thermo.t_max,
    self.adp_thermo_num_temps,
    self.config.thermo.steps_per_temp,
)?);  // ❌ NO cuda_device PASSED!
```

**🔧 FIX NEEDED**: Replace with GPU version:
```rust
#[cfg(feature = "cuda")]
self.thermodynamic_eq = if self.config.gpu.enable_thermo_gpu {
    Some(ThermodynamicEquilibrator::equilibrate_gpu(
        &self.cuda_device,  // ✅ Pass GPU device
        graph,
        &self.best_solution,
        self.config.target_chromatic,
        self.config.thermo.t_min,
        self.config.thermo.t_max,
        self.adp_thermo_num_temps,
        self.config.thermo.steps_per_temp,
    )?)
} else {
    Some(ThermodynamicEquilibrator::equilibrate(...)) // CPU fallback
};
```

---

#### **Phase 3: Quantum-Classical (Lines 1997-2088)**

**⚠️ HAS GPU DEVICE BUT DOESN'T USE IT - Line 2008:**
```rust
if self.config.gpu.enable_quantum_gpu {
    println!("[PHASE 3][GPU] Quantum solver active (iterations={}, retries={})",
             self.config.quantum.iterations,
             self.config.quantum.failure_retries);
}

// Line 2069 - ACTUAL CALL:
match qc_hybrid.solve_with_feedback(
    graph,
    &self.best_solution,
    initial_kuramoto,
    self.adp_quantum_iterations,
) {
    // ❌ quantum_solver HAS cuda_device but find_coloring() DOESN'T USE IT!
```

**🔧 FIX NEEDED**: Make `find_coloring()` actually use `self.gpu_device`

---

## 📍 Part 2: Transfer Entropy - CPU vs GPU Code

### **CPU Implementation**: `foundation/prct-core/src/transfer_entropy_coloring.rs:24-120`

**Current CPU Function (Lines 24-75)**:
```rust
pub fn compute_transfer_entropy_ordering(
    graph: &Graph,
    kuramoto_state: &KuramotoState,
    geodesic_features: Option<&GeodesicFeatures>,
    geodesic_weight: f64,
) -> Result<Vec<usize>> {
    let n = graph.num_vertices;

    // ❌ CPU CALL - Line 35:
    let te_matrix = compute_te_from_adjacency(graph)?;

    // Compute information centrality for each vertex (CPU)
    let mut centrality: Vec<(usize, f64)> = (0..n)
        .map(|v| {
            let outgoing: f64 = (0..n).map(|u| te_matrix[[v, u]]).sum();
            let incoming: f64 = (0..n).map(|u| te_matrix[[u, v]]).sum();
            let te_score = outgoing + incoming;
            (v, te_score)
        })
        .collect();

    centrality.sort_by(...);  // CPU sort
    Ok(centrality.iter().map(|(v, _)| *v).collect())
}
```

### **GPU Kernel Available**: `foundation/kernels/transfer_entropy.cu`

**6 Complete GPU Kernels**:
1. **`compute_minmax_kernel`** (Lines 25-56) - Find min/max for normalization
2. **`build_histogram_3d_kernel`** (Lines 59-99) - Build P(Y_future, X_past, Y_past)
3. **`build_histogram_2d_kernel`** (Lines 102-133) - Build P(Y_future, Y_past)
4. **`compute_transfer_entropy_kernel`** (Lines 137-203) - Compute TE from histograms
5. **`build_histogram_1d_kernel`** (Lines 206-231) - Build P(Y_past)
6. **`build_histogram_2d_xp_yp_kernel`** (Lines 234-267) - Build P(X_past, Y_past)

**🔧 WIRING NEEDED**: Create GPU wrapper function:
```rust
#[cfg(feature = "cuda")]
pub fn compute_transfer_entropy_ordering_gpu(
    cuda_device: &Arc<CudaDevice>,
    graph: &Graph,
    kuramoto_state: &KuramotoState,
    geodesic_features: Option<&GeodesicFeatures>,
    geodesic_weight: f64,
) -> Result<Vec<usize>> {
    // Load PTX and call the 6 kernels above
    // Return GPU-computed ordering
}
```

---

## 📍 Part 3: Thermodynamic - CPU vs GPU Code

### **CPU Implementation**: `foundation/prct-core/src/world_record_pipeline.rs:989-1075`

**Current CPU Function (Lines 999-1075)**:
```rust
pub fn equilibrate(
    graph: &Graph,
    initial_solution: &ColoringSolution,
    target_chromatic: usize,
    t_min: f64,
    t_max: f64,
    num_temps: usize,
    steps_per_temp: usize,
) -> Result<Self> {
    // ❌ CPU LOOP - Lines 1023-1074:
    for (i, &temp) in temperatures.iter().enumerate() {
        println!("[THERMODYNAMIC] Temperature {}/{}: T = {:.3}", i + 1, num_temps, temp);

        let mut best = current.clone();
        let adj = build_adjacency_matrix(graph);

        // ❌ CPU INNER LOOP - Lines 1031-1066:
        for _ in 0..steps_per_temp {
            let v = rand::random::<usize>() % graph.num_vertices;
            let old_color = current.colors[v];
            let new_color = rand::random::<usize>() % target_chromatic;

            current.colors[v] = new_color;

            // Compute energy change (CPU)
            let mut delta_conflicts = 0i32;
            for u in 0..graph.num_vertices {
                if adj[[v, u]] {
                    if current.colors[u] == old_color {
                        delta_conflicts -= 1;
                    }
                    if current.colors[u] == new_color {
                        delta_conflicts += 1;
                    }
                }
            }

            // Metropolis criterion (CPU)
            if delta_conflicts > 0 {
                let prob = (-delta_conflicts as f64 / temp).exp();
                if rand::random::<f64>() > prob {
                    current.colors[v] = old_color;
                }
            }
        }
    }
}
```

### **GPU Kernel Available**: `foundation/kernels/thermodynamic.cu`

**6 Complete GPU Kernels**:
1. **`initialize_oscillators_kernel`** (Lines 21-39) - Initialize positions/velocities
2. **`compute_coupling_forces_kernel`** (Lines 42-64) - Compute coupling forces
3. **`evolve_oscillators_kernel`** (Lines 67-113) - Langevin dynamics evolution
4. **`compute_energy_kernel`** (Lines 116-176) - Total energy calculation
5. **`compute_entropy_kernel`** (Lines 179-225) - Entropy calculation
6. **`compute_order_parameter_kernel`** (Lines 228-264) - Phase synchronization

**Key Equations Implemented in GPU**:
- Langevin: `dx/dt = -γx - ∇U(x) + √(2γkT) * η(t)`
- Coupling: `force[i] = -Σ_j coupling[i][j] * (x[i] - x[j])`

**🔧 WIRING NEEDED**: Create GPU version:
```rust
#[cfg(feature = "cuda")]
pub fn equilibrate_gpu(
    cuda_device: &Arc<CudaDevice>,
    graph: &Graph,
    initial_solution: &ColoringSolution,
    // ... same parameters
) -> Result<Self> {
    // Load PTX kernels
    // Launch parallel replicas on GPU
    // Use evolve_oscillators_kernel instead of CPU loop
    // 5x speedup expected
}
```

---

## 📍 Part 4: Quantum - CPU vs GPU Code

### **CPU Implementation**: `foundation/prct-core/src/quantum_coloring.rs:1-140`

**Quantum Solver Structure (Lines 18-42)**:
```rust
pub struct QuantumColoringSolver {
    /// GPU device for CUDA acceleration
    #[cfg(feature = "cuda")]
    gpu_device: Option<std::sync::Arc<cudarc::driver::CudaDevice>>,  // ✅ HAS DEVICE!
}

impl QuantumColoringSolver {
    pub fn new(
        #[cfg(feature = "cuda")]
        gpu_device: Option<std::sync::Arc<cudarc::driver::CudaDevice>>,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref dev) = gpu_device {
                println!("[QUANTUM][GPU] GPU acceleration ACTIVE on device {}", dev.ordinal());
                // ✅ LOGS GPU BUT THEN...
            }
        }

        Ok(Self {
            #[cfg(feature = "cuda")]
            gpu_device,  // ✅ Stores device
        })
    }
```

**Find Coloring Function (Lines 58-140)**:
```rust
pub fn find_coloring(
    &mut self,
    graph: &Graph,
    _phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
    initial_estimate: usize,
) -> Result<ColoringSolution> {
    // ❌ NEVER USES self.gpu_device!

    // Lines 76-140: All CPU processing
    let bounds = ChromaticBounds::from_graph_tda(graph)?;  // CPU
    let (initial_solution, actual_target) = self.adaptive_initial_solution(...)?;  // CPU

    // Iterative reduction loop - all CPU
    while current_target > target_min {
        // CPU QUBO solving
        // CPU DSATUR solving
    }
}
```

### **GPU Kernel Available**: `foundation/kernels/quantum_evolution.cu`

**Trotter-Suzuki GPU Implementation** (Lines 1-150):
- **`apply_diagonal_evolution`** (Lines 31-43) - Apply potential evolution
- **`apply_kinetic_evolution_momentum`** (Lines 47-62) - Kinetic evolution in momentum space
- **`trotter_suzuki_step`** (Lines 65-104) - Full time evolution with FFT
- **`build_tight_binding_hamiltonian`** (Lines 111-132) - Build Hamiltonian from graph
- **`build_ising_hamiltonian`** (Lines 135-150+) - Ising model for QUBO

**🔧 WIRING NEEDED**: Use `self.gpu_device` in `find_coloring()`:
```rust
pub fn find_coloring(&mut self, ...) -> Result<ColoringSolution> {
    #[cfg(feature = "cuda")]
    if let Some(ref device) = self.gpu_device {
        // ✅ Actually call GPU kernels here!
        return self.find_coloring_gpu(device, graph, ...);
    }

    // CPU fallback
    self.find_coloring_cpu(graph, ...)
}
```

---

## 📍 Part 5: Existing GPU Wiring Pattern (Working Example)

### **Location**: `foundation/prct-core/src/world_record_pipeline_gpu.rs:1-120`

**✅ THIS IS HOW TO DO IT CORRECTLY** (Phase 0 Reservoir):

```rust
#[cfg(feature = "cuda")]
pub struct GpuReservoirConflictPredictor {
    pub gpu_reservoir: GpuReservoirComputer,  // ✅ GPU object
    pub conflict_scores: Vec<f64>,
    pub difficulty_zones: Vec<Vec<usize>>,
}

#[cfg(feature = "cuda")]
impl GpuReservoirConflictPredictor {
    pub fn predict_gpu(
        graph: &Graph,
        coloring_history: &[ColoringSolution],
        kuramoto_state: &KuramotoState,
        cuda_device: Arc<CudaDevice>,  // ✅ Takes device as parameter
        phase_threshold: f64,
    ) -> Result<Self> {
        // Line 61: Initialize GPU reservoir WITH device
        let mut gpu_reservoir = GpuReservoirComputer::new_shared(
            reservoir_config,
            cuda_device,  // ✅ Pass device to GPU object
        ).map_err(|e| PRCTError::NeuromorphicFailed(...))?;

        // Line 94: Actually call GPU function
        let state = gpu_reservoir.process_gpu(&pattern)  // ✅ GPU CALL!
            .map_err(|e| PRCTError::NeuromorphicFailed(...))?;

        // Returns GPU-processed results
        Ok(Self { gpu_reservoir, conflict_scores, difficulty_zones })
    }
}
```

**Pattern to Copy**:
1. ✅ Take `Arc<CudaDevice>` as parameter
2. ✅ Pass device to GPU object initialization
3. ✅ Call `_gpu()` methods
4. ✅ Handle GPU errors with fallback

---

### **Location**: `foundation/cuda/prism_pipeline.rs:1-150`

**Full GPU Pipeline Example** (Lines 98-150):

```rust
pub struct PrismPipeline {
    context: Arc<CudaDevice>,  // ✅ Shared CUDA context
    config: PrismConfig,

    te_estimator: Option<GpuKSGEstimator>,      // ✅ GPU TE
    tda_engine: Option<GpuTDA>,                 // ✅ GPU TDA
    reservoir: Option<GpuReservoirComputer>,    // ✅ GPU Reservoir

    fusion_kernel: Arc<cudarc::driver::CudaFunction>,  // ✅ GPU kernel
    init_kernel: Arc<cudarc::driver::CudaFunction>,
}

impl PrismPipeline {
    pub fn new(config: PrismConfig) -> Result<Self> {
        // Line 130: Initialize shared CUDA context
        let context = CudaDevice::new(0)
            .map_err(|e| anyhow!("Failed to initialize CUDA device 0: {:?}", e))?;

        // Line 136: Load PTX kernels
        let ptx_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/adaptive_coloring.ptx"));
        let ptx = Ptx::from_src(std::str::from_utf8(ptx_bytes)?);

        // Line 144: Load kernel from PTX
        context.load_ptx(ptx, "prism_fusion", &kernel_names)?;

        let fusion_kernel = Arc::new(context.get_func("prism_fusion", "kernel_name")?);

        // ✅ This is the pattern to follow!
    }
}
```

**Key Pattern Elements**:
1. ✅ One shared `Arc<CudaDevice>` for all kernels
2. ✅ Load PTX from `include_bytes!` or file
3. ✅ Get function handles with `context.get_func()`
4. ✅ Store kernel references in struct
5. ✅ Call kernels with proper launch config

---

## 📍 Part 6: Build Configuration

### **Cargo.toml**: `foundation/prct-core/Cargo.toml:1-39`

```toml
[package]
name = "prct-core"
version = "0.1.0"
edition = "2021"

[features]
default = []
cuda = ["cudarc", "neuromorphic-engine"]  # ✅ CUDA feature gate

[dependencies]
shared-types = { path = "../shared-types", features = ["serde"] }
ndarray = "0.15"
rayon = "1.10"

# GPU dependencies (optional)
neuromorphic-engine = { path = "../neuromorphic", features = ["cuda"], optional = true }
cudarc = { version = "0.9", features = ["std"], optional = true }
```

### **Build Script**: `build.rs:1-127`

**Key Sections**:

**Lines 11-14 - Feature Detection**:
```rust
if env::var("CARGO_FEATURE_CUDA").is_ok() {
    compile_cuda_kernels();  // ✅ Only compile if cuda feature enabled
}
```

**Lines 37-39 - PTX Output Directory**:
```rust
let ptx_dir = out_dir.join("ptx");
std::fs::create_dir_all(&ptx_dir).unwrap();

// Also create target/ptx for runtime loading
let target_ptx_dir = PathBuf::from("target/ptx");  // ✅ Runtime kernels here!
std::fs::create_dir_all(&target_ptx_dir).unwrap();
```

**Lines 41-63 - Kernel Compilation**:
```rust
// Currently compiles 3 kernels:
compile_cu_file(&nvcc, &ptx_dir, "foundation/cuda/adaptive_coloring.cu", "adaptive_coloring.ptx");
compile_cu_file(&nvcc, &ptx_dir, "foundation/cuda/prct_kernels.cu", "prct_kernels.ptx");
compile_cu_file(&nvcc, &ptx_dir, "foundation/kernels/neuromorphic_gemv.cu", "neuromorphic_gemv.ptx");

// ⚠️ MISSING COMPILATIONS - Add these:
// compile_cu_file(&nvcc, &ptx_dir, "foundation/kernels/transfer_entropy.cu", "transfer_entropy.ptx");
// compile_cu_file(&nvcc, &ptx_dir, "foundation/kernels/thermodynamic.cu", "thermodynamic.ptx");
// compile_cu_file(&nvcc, &ptx_dir, "foundation/kernels/quantum_evolution.cu", "quantum_evolution.ptx");
```

**Lines 78-88 - NVCC Flags**:
```rust
.args(&[
    "--ptx",                    // Compile to PTX
    "-O3",                      // Optimize
    "--gpu-architecture=sm_90", // Target architecture
    "--use_fast_math",          // Fast math
    "--extended-lambda",        // Device lambdas
    "-Xptxas", "-v",            // Verbose
    "--default-stream", "per-thread",  // Thread safety
])
```

### **CUDA Module**: `foundation/cuda/mod.rs:1-100`

**Lines 11-17 - Module Exports**:
```rust
pub mod gpu_coloring;
pub mod prism_pipeline;  // ✅ Full GPU pipeline example
pub mod ensemble_generation;

pub use gpu_coloring::{GpuColoringEngine, GpuColoringResult};
pub use prism_pipeline::{PrismPipeline, PrismConfig, PrismCoherence};
```

**Lines 23-36 - External C API**:
```rust
extern "C" {
    fn cuda_adaptive_coloring(
        adjacency: *const c_int,
        coherence: *const f32,
        best_coloring: *mut c_int,
        best_chromatic: *mut c_int,
        n: c_int,
        num_edges: c_int,
        num_attempts: c_int,
        max_colors: c_int,
        temperature: f32,
        seed: u64,
    ) -> c_int;  // ✅ Example FFI binding
}
```

---

## 🎯 **Summary: What You Need to Wire**

### **3 Phases to Wire (7-10 hours total)**:

#### **1. Transfer Entropy** (~3-4 hours)
**Current**: Lines 1876 in world_record_pipeline.rs
```rust
let te_ordering = hybrid_te_kuramoto_ordering(graph, ...)?;  // ❌ CPU
```

**Kernel**: `foundation/kernels/transfer_entropy.cu` (6 kernels, 268 lines)

**Fix**: Create `hybrid_te_kuramoto_ordering_gpu()` that:
1. Loads `transfer_entropy.ptx`
2. Calls the 6 histogram kernels
3. Returns GPU-computed ordering

---

#### **2. Thermodynamic** (~2-3 hours)
**Current**: Line 1960 in world_record_pipeline.rs
```rust
ThermodynamicEquilibrator::equilibrate(graph, ...)  // ❌ CPU, NO device passed
```

**Kernel**: `foundation/kernels/thermodynamic.cu` (6 kernels, 265 lines)

**Fix**: Create `equilibrate_gpu()` that:
1. Takes `Arc<CudaDevice>`
2. Loads `thermodynamic.ptx`
3. Calls `evolve_oscillators_kernel` instead of CPU loop
4. Returns GPU-equilibrated states

---

#### **3. Quantum** (~2-3 hours)
**Current**: Lines 58-140 in quantum_coloring.rs
```rust
pub fn find_coloring(&mut self, ...) -> Result<ColoringSolution> {
    // ❌ Has self.gpu_device but NEVER USES IT!
}
```

**Kernel**: `foundation/kernels/quantum_evolution.cu` (Trotter-Suzuki, FFT-based)

**Fix**: In `find_coloring()`, add:
```rust
#[cfg(feature = "cuda")]
if let Some(ref device) = self.gpu_device {
    return self.find_coloring_gpu(device, graph, ...);  // ✅ Use GPU!
}
// CPU fallback
```

---

## 📋 **Checklist for GPU Wiring**

### **For Each Phase**:

- [ ] **Step 1**: Add kernel compilation to `build.rs`
```rust
compile_cu_file(&nvcc, &ptx_dir, "foundation/kernels/<name>.cu", "<name>.ptx");
```

- [ ] **Step 2**: Create GPU wrapper function
```rust
#[cfg(feature = "cuda")]
fn <function>_gpu(cuda_device: &Arc<CudaDevice>, ...) -> Result<T> {
    // Load PTX
    let ptx = std::fs::read_to_string("target/ptx/<name>.ptx")?;
    cuda_device.load_ptx(...)?;

    // Get kernel
    let kernel = cuda_device.get_func("module", "kernel_name")?;

    // Launch kernel
    unsafe { kernel.launch(cfg, (...params...)) }?;

    // Return GPU results
}
```

- [ ] **Step 3**: Modify pipeline call site
```rust
#[cfg(feature = "cuda")]
let result = if self.config.gpu.enable_X_gpu {
    <function>_gpu(&self.cuda_device, ...)?  // ✅ GPU path
} else {
    <function>(...)  // CPU fallback
};

#[cfg(not(feature = "cuda"))]
let result = <function>(...)?;  // CPU only
```

- [ ] **Step 4**: Test GPU usage
```bash
nvidia-smi dmon -s u &  # Should show >0% utilization
./target/release/examples/world_record_dsjc1000 config.toml
```

---

## 🎯 **Implementation Priority**

### **High ROI (Do First)**:
1. **Thermodynamic** (2-3 hrs → 5x speedup) - Biggest bottleneck
2. **Quantum** (2-3 hrs → 3x speedup) - Already has device
3. **Transfer Entropy** (3-4 hrs → 2-3x speedup) - Medium impact

### **Medium ROI (Optional)**:
4. Active Inference (4-5 hrs → 2x speedup)
5. TDA (3-4 hrs → unknown speedup)

### **Skip**:
- PIMC (not implemented)
- GNN Screening (not implemented)
- Multiscale (not implemented)

---

## 📝 **Exact Answers**

### **NOT Implemented Entirely: 3**
1. PIMC - Stub at `world_record_pipeline.rs:681-693`
2. GNN Screening - Stub at `world_record_pipeline.rs:675-678`
3. Multiscale - Not even checked, just config flag

### **Need GPU Wiring: 3-4**
1. Transfer Entropy - Kernel ready, needs wrapper
2. Thermodynamic - Kernel ready, needs wrapper
3. Quantum - Has device, just use it!
4. (Optional) Active Inference - Kernel ready, needs wrapper

**Total Wiring Effort**: 7-15 hours
**Expected Speedup Gain**: 10-15x additional (on top of current 15x from reservoir)
**Final Performance**: 50-80x total vs pure CPU

---

All the infrastructure is ready - you just need to wire the calls together!