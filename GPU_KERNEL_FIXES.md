# GPU Kernel Fix Guide - Complete Instructions

## 🔴 CRITICAL: 5 Unused PTX Files Must Be Wired

### 1. FIX NEUROMORPHIC GEMV
**File:** `foundation/neuromorphic/src/gpu_reservoir.rs`
**Add at line ~50:**
```rust
// Load neuromorphic GEMV kernels
let ptx_path = "foundation/kernels/ptx/neuromorphic_gemv.ptx";
let ptx = std::fs::read_to_string(ptx_path)?;
let ptx = cudarc::nvrtc::Ptx::from_src(&ptx);
let module = device.load_ptx(ptx, "neuromorphic_gemv", &[
    "spmv_kernel",
    "gemv_kernel",
    "spike_propagation_kernel"
])?;
```

### 2. FIX PARALLEL COLORING PATH
**Files:**
- `foundation/quantum/src/gpu_coloring.rs` (line ~50)
- `foundation/gpu_coloring.rs` (line ~30)

**Change:**
```rust
// OLD - WRONG PATH
let ptx_path = "target/ptx/parallel_coloring.ptx";

// NEW - CORRECT PATH
let ptx_path = "foundation/kernels/ptx/parallel_coloring.ptx";
```

### 3. FIX POLICY EVALUATION PATH
**File:** `foundation/active_inference/gpu_policy_eval.rs` (line 152)
**Change:**
```rust
// OLD
let ptx_path = "target/ptx/policy_evaluation.ptx";

// NEW
let ptx_path = "foundation/kernels/ptx/policy_evaluation.ptx";
```

### 4. WIRE QUANTUM EVOLUTION
**File:** `foundation/quantum_mlir/mod.rs`
**Add at initialization:**
```rust
// Load quantum evolution PTX
let ptx_path = "foundation/kernels/ptx/quantum_evolution.ptx";
let ptx = std::fs::read_to_string(ptx_path)?;
let ptx = cudarc::nvrtc::Ptx::from_src(&ptx);
let module = device.load_ptx(ptx, "quantum_evolution", &[
    "quantum_evolution_kernel",
    "hamiltonian_apply_kernel"
])?;
```

### 5. WIRE DOUBLE DOUBLE (Optional - not currently needed)
**File:** Create new `foundation/precision/double_double_gpu.rs`
```rust
pub struct DoubleDoubleGpu {
    device: Arc<CudaDevice>,
    module: Arc<CudaModule>,
}

impl DoubleDoubleGpu {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;
        let ptx_path = "foundation/kernels/ptx/double_double.ptx";
        let ptx = std::fs::read_to_string(ptx_path)?;
        let ptx = cudarc::nvrtc::Ptx::from_src(&ptx);
        let module = device.load_ptx(ptx, "double_double", &[
            "dd_add", "dd_mul", "dd_div"
        ])?;

        Ok(Self { device, module })
    }
}
```

## ⚠️ Fix Migration Issues (launch_builder patterns)

### ACTIVE INFERENCE
**File:** `foundation/active_inference/gpu.rs`
**Lines:** 145, 157, 177, 188, 198, 253, 275, 289

**Pattern to fix:**
```rust
// OLD
let mut launch_kl = self.device.launch_builder(&self.kl_divergence_kernel);
launch_kl.arg(&beliefs);
launch_kl.arg(&target);
unsafe { launch_kl.launch(config)?; }

// NEW
unsafe {
    self.kl_divergence_kernel.launch(
        config,
        (&beliefs, &target)
    )?;
}
```

### THERMODYNAMIC
**File:** `foundation/statistical_mechanics/gpu.rs`
**Fix all launch_builder patterns like above**

### GPU TDA - Convert to PTX
**File:** `foundation/phase6/gpu_tda.rs`

**Step 1:** Create `foundation/kernels/gpu_tda.cu`:
```cuda
extern "C" __global__ void compute_persistence_features(
    const bool* adjacency,
    float* vertex_features,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int degree = 0;
        int triangle_count = 0;

        // Count degree
        for (int i = 0; i < n; i++) {
            if (adjacency[v * n + i]) degree++;
        }

        // Count triangles
        for (int i = 0; i < n; i++) {
            if (adjacency[v * n + i]) {
                for (int j = i + 1; j < n; j++) {
                    if (adjacency[v * n + j] && adjacency[i * n + j]) {
                        triangle_count++;
                    }
                }
            }
        }

        float clustering = (degree > 1) ?
            (float)(2 * triangle_count) / (float)(degree * (degree - 1)) : 0.0f;

        vertex_features[v * 3 + 0] = (float)degree;
        vertex_features[v * 3 + 1] = (float)triangle_count;
        vertex_features[v * 3 + 2] = clustering;
    }
}
```

**Step 2:** Compile to PTX:
```bash
nvcc -ptx foundation/kernels/gpu_tda.cu -o foundation/kernels/ptx/gpu_tda.ptx
```

**Step 3:** Update `foundation/phase6/gpu_tda.rs`:
```rust
// Remove inline kernel string (lines 88-151)
// Replace with:
pub fn new() -> Result<Self> {
    let device = CudaDevice::new(0)?;

    // Load pre-compiled PTX
    let ptx_path = "foundation/kernels/ptx/gpu_tda.ptx";
    let ptx = std::fs::read_to_string(ptx_path)?;
    let ptx = cudarc::nvrtc::Ptx::from_src(&ptx);
    let module = device.load_ptx(ptx, "gpu_tda", &[
        "compute_persistence_features"
    ])?;

    Ok(Self { device, module })
}
```

## 🎯 VERIFICATION

After fixes, run:
```bash
# Check all PTX files are loaded
for ptx in foundation/kernels/ptx/*.ptx; do
    name=$(basename $ptx)
    echo -n "$name: "
    grep -r "$name" --include="*.rs" . 2>/dev/null | wc -l
done

# All should show > 0

# Test compilation
cargo check --features cuda

# Monitor GPU usage
nvidia-smi dmon -i 0 -s u
```

## PRIORITY ORDER
1. Fix path issues (Parallel Coloring, Policy Evaluation) - 5 minutes
2. Wire Neuromorphic GEMV - 10 minutes
3. Wire Quantum Evolution - 10 minutes
4. Fix remaining launch_builder patterns - 30 minutes
5. Convert GPU TDA to PTX - 20 minutes

**Total time: ~75 minutes to have ALL GPU kernels working!**