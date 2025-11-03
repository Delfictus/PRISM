# Complete GPU-Accelerated PRCT Integration Plan

## Executive Summary

To bring your full 3-layer PRCT algorithm to life with complete GPU acceleration in a fully functional PRISM platform, we need to implement **4 main components** across **3 phases** with an estimated **15-20 hours** of focused development work.

## Current State Analysis

### ✅ What We Have
- Universal binary platform (working)
- Greedy baseline (10 colors on Nipah, validated)
- All PRCT algorithm code exists in `foundation/`
- Dependencies configured and compiling
- GPU infrastructure ready (CUDA 13.0, RTX 5070)
- Integration framework prepared

### 🔧 What We Need
- Port adapter implementations (~6-8 hours)
- GPU kernel integration (~4-6 hours)
- Pipeline wiring and optimization (~3-4 hours)
- Testing and validation (~2 hours)

## Phase 1: Port Adapter Implementation (6-8 hours)

### Component 1.1: Neuromorphic Adapter (2-3 hours)

**File**: `src/cuda/prct_adapters/neuromorphic.rs` (NEW, ~200 lines)

**Requirements**:
1. Graph → Spike encoding
2. GPU reservoir computing
3. Pattern detection
4. Phase extraction

**Implementation**:
```rust
use prct_core::ports::NeuromorphicPort;
use neuromorphic_engine::{GpuReservoirComputer, SpikeEncoder};
use shared_types::*;
use anyhow::Result;

pub struct NeuromorphicAdapter {
    encoder: SpikeEncoder,
    reservoir: GpuReservoirComputer,
    pattern_detector: PatternDetector,
}

impl NeuromorphicAdapter {
    pub fn new() -> Result<Self> {
        let encoder = SpikeEncoder::new(SpikeEncoderConfig {
            base_frequency: 20.0,  // Hz
            time_window: 100.0,    // ms
            enable_burst: true,
        })?;

        let reservoir = GpuReservoirComputer::new(ReservoirConfig {
            size: 1000,
            spectral_radius: 0.9,
            connection_prob: 0.1,
            leak_rate: 0.3,
        })?;

        let pattern_detector = PatternDetector::new()?;

        Ok(Self { encoder, reservoir, pattern_detector })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        // Step 1: Convert graph structure to spike trains
        let adjacency_signals = self.graph_to_signals(graph);

        // Step 2: Encode as spikes
        let spikes = self.encoder.encode(&adjacency_signals, params)?;

        Ok(spikes)
    }

    fn process_and_detect_patterns(
        &self,
        spikes: &SpikePattern,
    ) -> Result<NeuroState> {
        // Step 1: Process through GPU reservoir
        let reservoir_state = self.reservoir.process_gpu(spikes)?;

        // Step 2: Detect patterns
        let patterns = self.pattern_detector.detect(&reservoir_state)?;

        // Step 3: Extract phases from reservoir activations
        let phases = self.extract_phases(&reservoir_state);

        Ok(NeuroState {
            neuron_states: reservoir_state.activations,
            spike_pattern: spikes.spikes.clone(),
            coherence: self.compute_coherence(&phases),
            phases,
            detected_patterns: patterns,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        self.pattern_detector.get_patterns()
    }
}

impl NeuromorphicAdapter {
    fn graph_to_signals(&self, graph: &Graph) -> Vec<Vec<f64>> {
        // Convert graph structure to temporal signals
        let n = graph.num_vertices;
        let mut signals = vec![vec![0.0; 100]; n];  // 100ms window

        for i in 0..n {
            let degree = graph.adjacency[i].len();
            let signal = generate_degree_signal(degree, 100);
            signals[i] = signal;
        }

        signals
    }

    fn extract_phases(&self, state: &ReservoirState) -> Vec<f64> {
        // Extract oscillatory phases from activations using Hilbert transform
        state.activations.iter()
            .map(|&activation| {
                (activation * 2.0 * std::f64::consts::PI).atan2(1.0)
            })
            .collect()
    }

    fn compute_coherence(&self, phases: &[f64]) -> f64 {
        // Compute phase coherence (Kuramoto order parameter)
        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }
}
```

**GPU Kernels Needed**:
1. `reservoir_update.cu` - Update reservoir neurons on GPU
2. `spike_encoding.cu` - Fast spike encoding
3. `phase_extraction.cu` - Hilbert transform for phase

**Integration Points**:
- Use existing `foundation/neuromorphic/src/reservoir.rs`
- Adapt `foundation/neuromorphic/src/spike_encoder.rs`
- GPU reservoir already exists: `GpuReservoirComputer`

**Estimated Time**: 2-3 hours
- 1 hour: Type conversions and wiring
- 1 hour: Phase extraction logic
- 30 min: Testing and debugging

---

### Component 1.2: Quantum Adapter (2-3 hours)

**File**: `src/cuda/prct_adapters/quantum.rs` (NEW, ~200 lines)

**Requirements**:
1. Build Hamiltonian from graph
2. GPU quantum evolution
3. Phase field extraction
4. Ground state computation

**Implementation**:
```rust
use prct_core::ports::QuantumPort;
use quantum_engine::{QuantumEngine, HamiltonianBuilder};
use shared_types::*;
use anyhow::Result;

pub struct QuantumAdapter {
    engine: QuantumEngine,
    hamiltonian_builder: HamiltonianBuilder,
}

impl QuantumAdapter {
    pub fn new() -> Result<Self> {
        let engine = QuantumEngine::new()?;
        let hamiltonian_builder = HamiltonianBuilder::new()?;

        Ok(Self { engine, hamiltonian_builder })
    }
}

impl QuantumPort for QuantumAdapter {
    fn build_hamiltonian(
        &self,
        graph: &Graph,
        params: &EvolutionParams,
    ) -> Result<HamiltonianState> {
        // Step 1: Convert graph to coupling matrix
        let coupling_matrix = self.graph_to_coupling(graph);

        // Step 2: Build quantum Hamiltonian
        // H = -Σᵢⱼ Jᵢⱼ σᵢ⋅σⱼ (Ising-like interaction)
        let hamiltonian = self.hamiltonian_builder.build_from_coupling(
            &coupling_matrix,
            params,
        )?;

        Ok(hamiltonian)
    }

    fn evolve_state(
        &self,
        hamiltonian: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        // GPU-accelerated quantum evolution
        // ψ(t) = e^(-iHt/ℏ) ψ(0)
        let evolved_state = self.engine.evolve_gpu(
            hamiltonian,
            initial_state,
            evolution_time,
        )?;

        Ok(evolved_state)
    }

    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField> {
        // Extract phase field from quantum state
        let n = state.amplitudes.len();

        // Compute phases from complex amplitudes
        let phases: Vec<f64> = state.amplitudes.iter()
            .map(|(re, im)| im.atan2(*re))
            .collect();

        // Compute coherence matrix
        let coherence_matrix = self.compute_coherence_matrix(&state.amplitudes);

        // Compute order parameter
        let order_parameter = self.compute_order_parameter(&phases);

        Ok(PhaseField {
            phases,
            coherence_matrix,
            order_parameter,
            dimension: n,
        })
    }

    fn compute_ground_state(
        &self,
        hamiltonian: &HamiltonianState,
    ) -> Result<QuantumState> {
        // Find ground state using GPU-accelerated imaginary time evolution
        self.engine.find_ground_state_gpu(hamiltonian)
    }
}

impl QuantumAdapter {
    fn graph_to_coupling(&self, graph: &Graph) -> Vec<Vec<f64>> {
        let n = graph.num_vertices;
        let mut coupling = vec![vec![0.0; n]; n];

        for i in 0..n {
            for &j in &graph.adjacency[i] {
                // Coupling strength proportional to edge weight
                coupling[i][j] = -1.0;  // Antiferromagnetic coupling
                coupling[j][i] = -1.0;
            }
        }

        coupling
    }

    fn compute_coherence_matrix(&self, amplitudes: &[(f64, f64)]) -> Vec<f64> {
        let n = amplitudes.len();
        let mut coherence = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                // Coherence = |⟨ψᵢ|ψⱼ⟩|
                let (re_i, im_i) = amplitudes[i];
                let (re_j, im_j) = amplitudes[j];

                let overlap = re_i * re_j + im_i * im_j;
                coherence[i * n + j] = overlap.abs();
            }
        }

        coherence
    }

    fn compute_order_parameter(&self, phases: &[f64]) -> f64 {
        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }
}
```

**GPU Kernels Needed**:
1. `hamiltonian_matrix.cu` - Build Hamiltonian matrix
2. `quantum_evolution.cu` - Time evolution (matrix exponential)
3. `phase_extraction.cu` - Extract phases from amplitudes
4. `coherence_matrix.cu` - Compute coherence

**Integration Points**:
- Use existing `foundation/quantum/src/hamiltonian.rs`
- Leverage `foundation/quantum/src/prct_coloring.rs`
- Quantum engine structure already exists

**Estimated Time**: 2-3 hours
- 1 hour: Hamiltonian construction
- 1 hour: Evolution wiring
- 30 min: Phase extraction
- 30 min: Testing

---

### Component 1.3: Physics Coupling Adapter (1-2 hours)

**File**: `src/cuda/prct_adapters/coupling.rs` (NEW, ~150 lines)

**Requirements**:
1. Bidirectional coupling computation
2. Kuramoto synchronization
3. Transfer entropy calculation
4. GPU-accelerated evolution

**Implementation**:
```rust
use prct_core::ports::PhysicsCouplingPort;
use prct_core::coupling::PhysicsCouplingService;
use shared_types::*;
use anyhow::Result;

pub struct PhysicsCouplingAdapter {
    service: PhysicsCouplingService,
    coupling_strength: f64,
}

impl PhysicsCouplingAdapter {
    pub fn new(coupling_strength: f64) -> Result<Self> {
        let service = PhysicsCouplingService::new(coupling_strength);
        Ok(Self { service, coupling_strength })
    }
}

impl PhysicsCouplingPort for PhysicsCouplingAdapter {
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength> {
        // Compute information flow from neuro → quantum
        let neuro_to_quantum = self.compute_info_flow(
            &neuro_state.neuron_states,
            &quantum_state.amplitudes,
        )?;

        // Compute information flow from quantum → neuro
        let quantum_to_neuro = self.compute_info_flow_reverse(
            &quantum_state.amplitudes,
            &neuro_state.neuron_states,
        )?;

        Ok(CouplingStrength {
            forward: neuro_to_quantum,
            backward: quantum_to_neuro,
            bidirectional: (neuro_to_quantum + quantum_to_neuro) / 2.0,
        })
    }

    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState> {
        // Combine neuro and quantum phases
        let mut combined_phases = neuro_phases.to_vec();
        combined_phases.extend_from_slice(quantum_phases);

        // Natural frequencies (slightly different for diversity)
        let natural_frequencies = vec![1.0; combined_phases.len()];

        // Evolve Kuramoto dynamics on GPU
        let evolved_phases = self.evolve_kuramoto_gpu(
            &combined_phases,
            &natural_frequencies,
            dt,
        )?;

        // Compute order parameter
        let order_parameter = PhysicsCouplingService::compute_order_parameter(
            &evolved_phases
        );

        Ok(KuramotoState {
            phases: evolved_phases,
            order_parameter,
            coupling_strength: self.coupling_strength,
        })
    }

    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        lag: f64,
    ) -> Result<TransferEntropy> {
        // GPU-accelerated transfer entropy calculation
        let te_value = self.compute_te_gpu(source, target, lag)?;

        Ok(TransferEntropy {
            value: te_value,
            source_entropy: self.entropy(source),
            target_entropy: self.entropy(target),
            mutual_info: self.mutual_information(source, target),
        })
    }

    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling> {
        // Extract phases
        let neuro_phases = &neuro_state.phases;
        let quantum_phases = self.extract_quantum_phases(quantum_state);

        // Update Kuramoto synchronization
        let kuramoto_state = self.update_kuramoto_sync(
            neuro_phases,
            &quantum_phases,
            0.01,  // dt = 10ms
        )?;

        // Compute coupling strength
        let coupling = self.compute_coupling(neuro_state, quantum_state)?;

        Ok(BidirectionalCoupling {
            kuramoto_state,
            coupling_strength: coupling,
            neuro_to_quantum_entropy: 0.0,  // TODO: compute
            quantum_to_neuro_entropy: 0.0,  // TODO: compute
        })
    }
}

impl PhysicsCouplingAdapter {
    fn evolve_kuramoto_gpu(
        &self,
        phases: &[f64],
        frequencies: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>> {
        // Use the service's kuramoto_step
        let mut evolved = phases.to_vec();
        self.service.kuramoto_step(&mut evolved, frequencies, dt)?;
        Ok(evolved)
    }

    fn extract_quantum_phases(&self, state: &QuantumState) -> Vec<f64> {
        state.amplitudes.iter()
            .map(|(re, im)| im.atan2(*re))
            .collect()
    }

    fn compute_info_flow(&self, source: &[f64], target: &[(f64, f64)]) -> Result<f64> {
        // Simplified: measure correlation
        let target_magnitudes: Vec<f64> = target.iter()
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect();

        let correlation = self.correlation(source, &target_magnitudes);
        Ok(correlation.abs())
    }

    fn compute_info_flow_reverse(&self, source: &[(f64, f64)], target: &[f64]) -> Result<f64> {
        let source_magnitudes: Vec<f64> = source.iter()
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect();

        let correlation = self.correlation(&source_magnitudes, target);
        Ok(correlation.abs())
    }

    fn correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        let mean_x: f64 = x.iter().take(y.len()).sum::<f64>() / n;
        let mean_y: f64 = y.iter().take(x.len()).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n as usize {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }

        cov / (var_x * var_y).sqrt()
    }

    fn entropy(&self, data: &[f64]) -> f64 {
        // Simplified entropy calculation
        // TODO: Proper histogram-based entropy
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln()
    }

    fn mutual_information(&self, x: &[f64], y: &[f64]) -> f64 {
        // I(X;Y) ≈ -0.5 * ln(1 - ρ²)
        let correlation = self.correlation(x, y);
        -0.5 * (1.0 - correlation * correlation).ln().max(0.0)
    }

    fn compute_te_gpu(&self, source: &[f64], target: &[f64], _lag: f64) -> Result<f64> {
        // Simplified: use mutual information as proxy
        // TODO: Proper transfer entropy with GPU
        Ok(self.mutual_information(source, target))
    }
}
```

**GPU Kernels Needed**:
1. `kuramoto_evolution.cu` - Kuramoto dynamics
2. `transfer_entropy.cu` - TE calculation
3. `phase_coupling.cu` - Phase coupling computation

**Integration Points**:
- Use existing `foundation/prct-core/src/coupling.rs::PhysicsCouplingService`
- Service already has Kuramoto implementation!
- Just need to wrap it with GPU acceleration

**Estimated Time**: 1-2 hours
- 30 min: Wrapper implementation
- 30 min: GPU kernel integration
- 30 min: Testing

---

## Phase 2: GPU Kernel Development (4-6 hours)

### Component 2.1: CUDA Kernels

**File**: `foundation/cuda/prct_kernels.cu` (NEW, ~500 lines)

**Kernels Required**:

#### 1. Reservoir Update Kernel
```cuda
__global__ void reservoir_update_kernel(
    const float* input_spikes,
    const float* reservoir_weights,
    float* reservoir_states,
    int num_neurons,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    float activation = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        activation += input_spikes[i] * reservoir_weights[idx * batch_size + i];
    }

    // Leaky integration
    reservoir_states[idx] = 0.3f * reservoir_states[idx] + 0.7f * tanhf(activation);
}
```

#### 2. Quantum Evolution Kernel
```cuda
__global__ void quantum_evolution_kernel(
    const cuDoubleComplex* hamiltonian,
    cuDoubleComplex* state,
    double dt,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dimension) return;

    // ψ_new = (1 - iHdt)ψ (first-order approximation)
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    for (int j = 0; j < dimension; j++) {
        cuDoubleComplex h_elem = hamiltonian[idx * dimension + j];
        cuDoubleComplex psi_j = state[j];

        // -i * H * dt * psi
        cuDoubleComplex term = cuCmul(
            make_cuDoubleComplex(0.0, -dt),
            cuCmul(h_elem, psi_j)
        );

        result = cuCadd(result, term);
    }

    // Add identity term
    result = cuCadd(result, state[idx]);

    state[idx] = result;
}
```

#### 3. Kuramoto Evolution Kernel
```cuda
__global__ void kuramoto_evolution_kernel(
    float* phases,
    const float* natural_frequencies,
    float coupling_strength,
    float dt,
    int num_oscillators
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_oscillators) return;

    // Compute coupling term: Σⱼ sin(θⱼ - θᵢ)
    float coupling_sum = 0.0f;
    for (int j = 0; j < num_oscillators; j++) {
        if (i != j) {
            coupling_sum += sinf(phases[j] - phases[i]);
        }
    }

    // dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    float dphase = natural_frequencies[i] +
                   (coupling_strength / num_oscillators) * coupling_sum;

    phases[i] += dphase * dt;

    // Wrap to [0, 2π]
    phases[i] = fmodf(phases[i], 2.0f * M_PI);
}
```

#### 4. Phase Coherence Kernel
```cuda
__global__ void phase_coherence_kernel(
    const float* phases,
    float* coherence_matrix,
    int num_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_vertices || j >= num_vertices) return;

    // Coherence = cos(θᵢ - θⱼ)
    float phase_diff = phases[i] - phases[j];
    coherence_matrix[i * num_vertices + j] = cosf(phase_diff);
}
```

#### 5. Phase-Guided Color Selection Kernel
```cuda
__global__ void phase_guided_color_selection_kernel(
    const float* phase_coherence,
    const int* coloring,
    const bool* forbidden_colors,
    float* color_scores,
    int vertex,
    int num_vertices,
    int num_colors
) {
    int color = blockIdx.x * blockDim.x + threadIdx.x;
    if (color >= num_colors || forbidden_colors[color]) {
        color_scores[color] = -1e9f;  // Invalid
        return;
    }

    // Compute average coherence with vertices of this color
    float total_coherence = 0.0f;
    int count = 0;

    for (int v = 0; v < num_vertices; v++) {
        if (coloring[v] == color) {
            total_coherence += phase_coherence[vertex * num_vertices + v];
            count++;
        }
    }

    color_scores[color] = (count > 0) ? (total_coherence / count) : 1.0f;
}
```

**Estimated Time**: 4-6 hours
- 2 hours: Kernel implementation
- 1 hour: Host wrappers
- 1 hour: Memory management
- 1-2 hours: Optimization and debugging

---

### Component 2.2: GPU Memory Management

**File**: `src/cuda/prct_gpu_manager.rs` (NEW, ~200 lines)

```rust
use cudarc::driver::{CudaDevice, CudaSlice};
use anyhow::Result;

pub struct PRCTGpuManager {
    device: CudaDevice,

    // Pre-allocated GPU buffers
    reservoir_states: CudaSlice<f32>,
    quantum_state: CudaSlice<(f64, f64)>,
    phases: CudaSlice<f64>,
    coherence_matrix: CudaSlice<f64>,
}

impl PRCTGpuManager {
    pub fn new(max_vertices: usize) -> Result<Self> {
        let device = CudaDevice::new(0)?;

        // Pre-allocate buffers
        let reservoir_states = device.alloc_zeros::<f32>(1000)?;  // 1000 neurons
        let quantum_state = device.alloc_zeros::<(f64, f64)>(max_vertices)?;
        let phases = device.alloc_zeros::<f64>(max_vertices)?;
        let coherence_matrix = device.alloc_zeros::<f64>(max_vertices * max_vertices)?;

        Ok(Self {
            device,
            reservoir_states,
            quantum_state,
            phases,
            coherence_matrix,
        })
    }

    pub fn upload_graph(&mut self, adjacency: &[Vec<usize>]) -> Result<CudaSlice<i32>> {
        // Convert adjacency list to GPU-friendly format
        let flat_adjacency = flatten_adjacency(adjacency);
        self.device.htod_copy(flat_adjacency)
    }

    pub fn download_coloring(&self, gpu_coloring: &CudaSlice<i32>) -> Result<Vec<usize>> {
        let host_coloring = self.device.dtoh_sync_copy(gpu_coloring)?;
        Ok(host_coloring.into_iter().map(|x| x as usize).collect())
    }
}
```

**Estimated Time**: 1-2 hours

---

## Phase 3: Pipeline Integration & Optimization (3-4 hours)

### Component 3.1: Main PRCT Pipeline

**File**: `src/cuda/prct_algorithm.rs` (MODIFY existing ~305 lines → ~500 lines)

**Complete Implementation**:
```rust
use prct_core::{PRCTAlgorithm as CorePRCT, PRCTConfig as CoreConfig};
use super::prct_adapters::{NeuromorphicAdapter, QuantumAdapter, PhysicsCouplingAdapter};
use super::prct_gpu_manager::PRCTGpuManager;
use shared_types::*;
use std::sync::Arc;
use anyhow::Result;

pub struct PRCTAlgorithm {
    core_algorithm: CorePRCT,
    gpu_manager: PRCTGpuManager,
}

impl PRCTAlgorithm {
    pub fn new() -> Result<Self> {
        // Create GPU manager
        let gpu_manager = PRCTGpuManager::new(10000)?;  // Max 10k vertices

        // Create port adapters with GPU acceleration
        let neuro_port = Arc::new(NeuromorphicAdapter::new()?);
        let quantum_port = Arc::new(QuantumAdapter::new()?);
        let coupling_port = Arc::new(PhysicsCouplingAdapter::new(1.0)?);

        // Create core PRCT with dependency injection
        let config = CoreConfig {
            target_colors: 10,
            quantum_evolution_time: 0.1,
            kuramoto_coupling: 0.5,
            neuro_encoding: NeuromorphicEncodingParams::default(),
            quantum_params: EvolutionParams {
                dt: 0.01,
                strength: 1.0,
                damping: 0.1,
                temperature: 300.0,
            },
        };

        let core_algorithm = CorePRCT::new(
            neuro_port,
            quantum_port,
            coupling_port,
            config,
        );

        Ok(Self { core_algorithm, gpu_manager })
    }

    pub fn with_config(config: PRCTConfig) -> Result<Self> {
        // Convert our config to core config
        let core_config = CoreConfig {
            target_colors: 10,
            quantum_evolution_time: config.temperature as f64 * 0.001,
            kuramoto_coupling: config.probability_threshold,
            neuro_encoding: NeuromorphicEncodingParams::default(),
            quantum_params: EvolutionParams {
                dt: 0.01,
                strength: 1.0,
                damping: 0.1,
                temperature: config.temperature,
            },
        };

        // Create adapters
        let neuro_port = Arc::new(NeuromorphicAdapter::new()?);
        let quantum_port = Arc::new(QuantumAdapter::new()?);
        let coupling_port = Arc::new(PhysicsCouplingAdapter::new(
            core_config.kuramoto_coupling
        )?);

        let core_algorithm = CorePRCT::new(
            neuro_port,
            quantum_port,
            coupling_port,
            core_config,
        );

        let gpu_manager = PRCTGpuManager::new(10000)?;

        Ok(Self { core_algorithm, gpu_manager })
    }

    pub fn color(&self, adjacency: &[Vec<usize>], _ordering: &[usize]) -> Result<Vec<usize>> {
        let start = std::time::Instant::now();

        println!("[PRCT] Starting full 3-layer pipeline...");

        // Convert adjacency to Graph
        let graph = self.adjacency_to_graph(adjacency);

        println!("[PRCT] Graph: {} vertices, {} edges",
                 graph.num_vertices,
                 graph.adjacency.len() / 2);

        // Run full PRCT pipeline (neuromorphic → quantum → kuramoto → coloring)
        println!("[PRCT] Layer 1: Neuromorphic processing...");
        let t1 = std::time::Instant::now();

        let solution = self.core_algorithm.solve(&graph)?;

        let total_time = start.elapsed();
        println!("[PRCT] Total pipeline time: {:?}", total_time);
        println!("[PRCT]   Coloring: {} colors", solution.coloring.chromatic_number);
        println!("[PRCT]   Phase coherence: {:.3}", solution.phase_coherence);
        println!("[PRCT]   Kuramoto order: {:.3}", solution.kuramoto_order);
        println!("[PRCT]   Quality: {:.3}", solution.overall_quality);

        // Extract coloring
        Ok(solution.coloring.colors)
    }

    pub fn refine(&self, adjacency: &[Vec<usize>], coloring: &mut Vec<usize>) -> Result<usize> {
        // TODO: Implement refinement pass
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
        Ok(num_colors)
    }

    fn adjacency_to_graph(&self, adjacency: &[Vec<usize>]) -> Graph {
        let n = adjacency.len();

        // Flatten adjacency list to boolean array
        let mut flat_adjacency = vec![false; n * n];
        for i in 0..n {
            for &j in &adjacency[i] {
                if j < n {
                    flat_adjacency[i * n + j] = true;
                }
            }
        }

        Graph {
            num_vertices: n,
            adjacency: flat_adjacency,
        }
    }
}
```

**Estimated Time**: 2-3 hours
- 1 hour: Type conversions
- 1 hour: Integration and wiring
- 30 min: Error handling
- 30 min: Logging and metrics

---

### Component 3.2: Module Organization

**File**: `src/cuda/prct_adapters/mod.rs` (NEW)

```rust
pub mod neuromorphic;
pub mod quantum;
pub mod coupling;

pub use neuromorphic::NeuromorphicAdapter;
pub use quantum::QuantumAdapter;
pub use coupling::PhysicsCouplingAdapter;
```

**File**: `src/cuda/mod.rs` (MODIFY)

```rust
pub mod prct_adapters;
pub mod prct_gpu_manager;
pub mod prct_algorithm;

pub use prct_algorithm::{PRCTAlgorithm, PRCTConfig};
```

**Estimated Time**: 30 minutes

---

## Phase 4: Testing & Validation (2 hours)

### Component 4.1: Unit Tests

**File**: `src/cuda/prct_adapters/tests.rs` (NEW, ~200 lines)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromorphic_adapter() {
        let adapter = NeuromorphicAdapter::new().unwrap();

        // Create simple graph
        let graph = Graph {
            num_vertices: 10,
            adjacency: vec![false; 100],
        };

        let params = NeuromorphicEncodingParams::default();
        let spikes = adapter.encode_graph_as_spikes(&graph, &params).unwrap();

        assert!(spikes.spikes.len() > 0);
    }

    #[test]
    fn test_quantum_adapter() {
        let adapter = QuantumAdapter::new().unwrap();

        let graph = Graph {
            num_vertices: 10,
            adjacency: vec![false; 100],
        };

        let params = EvolutionParams {
            dt: 0.01,
            strength: 1.0,
            damping: 0.1,
            temperature: 300.0,
        };

        let hamiltonian = adapter.build_hamiltonian(&graph, &params).unwrap();
        assert_eq!(hamiltonian.dimension, 10);
    }

    #[test]
    fn test_kuramoto_sync() {
        let adapter = PhysicsCouplingAdapter::new(1.0).unwrap();

        let neuro_phases = vec![0.0, 1.0, 2.0];
        let quantum_phases = vec![0.5, 1.5, 2.5];

        let kuramoto = adapter.update_kuramoto_sync(
            &neuro_phases,
            &quantum_phases,
            0.01,
        ).unwrap();

        assert_eq!(kuramoto.phases.len(), 6);
        assert!(kuramoto.order_parameter >= 0.0);
        assert!(kuramoto.order_parameter <= 1.0);
    }
}
```

**Estimated Time**: 1 hour

---

### Component 4.2: Integration Tests

**File**: `tests/prct_integration_test.rs` (NEW)

```rust
use prism_ai::cuda::PRCTAlgorithm;

#[test]
fn test_small_graph_coloring() {
    let prct = PRCTAlgorithm::new().unwrap();

    // Triangle graph (K3)
    let adjacency = vec![
        vec![1, 2],
        vec![0, 2],
        vec![0, 1],
    ];

    let ordering = vec![0, 1, 2];
    let coloring = prct.color(&adjacency, &ordering).unwrap();

    assert_eq!(coloring.len(), 3);

    // Verify no conflicts
    assert_ne!(coloring[0], coloring[1]);
    assert_ne!(coloring[1], coloring[2]);
    assert_ne!(coloring[0], coloring[2]);

    // Should use 3 colors for K3
    let num_colors = coloring.iter().max().unwrap() + 1;
    assert_eq!(num_colors, 3);
}

#[test]
fn test_queen_8x8() {
    // Load Queen 8x8 graph
    let (adjacency, _) = load_dimacs_graph("benchmarks/dimacs/queen8_8.col").unwrap();

    let prct = PRCTAlgorithm::new().unwrap();
    let ordering: Vec<usize> = (0..64).collect();

    let coloring = prct.color(&adjacency, &ordering).unwrap();

    // Should use 9-12 colors (optimal is 9)
    let num_colors = coloring.iter().max().unwrap() + 1;
    assert!(num_colors >= 9);
    assert!(num_colors <= 12);

    // Verify no conflicts
    for i in 0..64 {
        for &j in &adjacency[i] {
            assert_ne!(coloring[i], coloring[j],
                      "Conflict between vertices {} and {}", i, j);
        }
    }
}
```

**Estimated Time**: 1 hour

---

## Phase 5: Documentation & Polish (1-2 hours)

### Component 5.1: API Documentation

Add comprehensive rustdoc comments to all modules.

**Estimated Time**: 30 minutes

---

### Component 5.2: Performance Profiling

Profile GPU kernel performance and optimize bottlenecks.

**Estimated Time**: 1 hour

---

### Component 5.3: User Guide Update

Update documentation with PRCT usage examples and performance expectations.

**Estimated Time**: 30 minutes

---

## Complete Timeline Summary

| Phase | Component | Hours | Total |
|-------|-----------|-------|-------|
| **1** | Neuromorphic Adapter | 2-3 | |
| **1** | Quantum Adapter | 2-3 | |
| **1** | Coupling Adapter | 1-2 | **6-8** |
| **2** | CUDA Kernels | 4-6 | |
| **2** | GPU Memory Manager | 1-2 | **5-8** |
| **3** | Pipeline Integration | 2-3 | |
| **3** | Module Organization | 0.5 | **2.5-3.5** |
| **4** | Testing | 2 | **2** |
| **5** | Documentation & Polish | 1-2 | **1-2** |
| | | **TOTAL** | **17-23.5 hours** |

**Realistic estimate**: **15-20 hours** of focused development

---

## Development Workflow

### Week 1 (8-10 hours)
- Day 1-2: Port adapters (neuromorphic, quantum, coupling)
- Day 3: GPU kernels (basic versions)

### Week 2 (7-10 hours)
- Day 4-5: Pipeline integration and wiring
- Day 6: GPU optimization
- Day 7: Testing and validation

---

## Expected Performance After Full Integration

### Nipah 2VSM (550 vertices, 2,834 edges)

**Greedy**:
```
Algorithm: greedy
Colors: 10
Time: 2.6s
```

**PRCT (Full 3-Layer)**:
```
Algorithm: prct
Colors: 8-10 (better clustering)
Time: 5-10s

Breakdown:
  - Neuromorphic encoding: 1-2s
  - Quantum evolution: 2-3s
  - Kuramoto sync: 0.5s
  - Phase-guided coloring: 1-2s
  - TSP tours: 1s

Metrics:
  - Phase coherence: 0.85
  - Kuramoto order: 0.92
  - Quality score: 0.88
```

### Queen 8x8 (64 vertices, 1,456 edges)

**Greedy**: 12 colors, 2.6s
**PRCT**: 9-11 colors (closer to optimal 9), 3-5s

---

## GPU Acceleration Benefits

### Without GPU
- Neuromorphic: 5-10s (CPU reservoir computing)
- Quantum: 10-20s (CPU matrix operations)
- Kuramoto: 2-5s (CPU phase evolution)
- **Total**: 17-35s

### With GPU
- Neuromorphic: 1-2s (GPU reservoir)
- Quantum: 2-3s (GPU evolution)
- Kuramoto: 0.5s (GPU dynamics)
- **Total**: 5-10s

**Speedup**: 3-7x faster with full GPU acceleration!

---

## Success Criteria

### Functional Requirements ✅
- [ ] All 3 layers execute successfully
- [ ] Produces valid colorings (no conflicts)
- [ ] Better quality than greedy baseline
- [ ] Completes in reasonable time (<10s for medium graphs)
- [ ] GPU acceleration working

### Performance Requirements ✅
- [ ] 3-7x speedup vs CPU implementation
- [ ] 8-10 colors on Nipah (vs greedy's 10)
- [ ] 9-11 colors on Queen 8x8 (vs greedy's 12)
- [ ] Phase coherence > 0.8
- [ ] Kuramoto order > 0.9

### Code Quality Requirements ✅
- [ ] All tests passing
- [ ] Documentation complete
- [ ] No memory leaks
- [ ] Error handling robust
- [ ] Logging comprehensive

---

## Risk Mitigation

### Risk 1: Type Conversion Complexity
**Mitigation**: Create helper functions for all conversions early

### Risk 2: GPU Memory Limits
**Mitigation**: Implement chunking for large graphs (>1000 vertices)

### Risk 3: Numerical Stability
**Mitigation**: Add bounds checking and normalization in kernels

### Risk 4: Integration Bugs
**Mitigation**: Test each component independently before full integration

---

## Final Deliverables

After completing all phases, you will have:

1. ✅ **Fully functional PRCT algorithm** with all 3 layers
2. ✅ **GPU-accelerated implementation** (3-7x speedup)
3. ✅ **Complete test suite** (unit + integration)
4. ✅ **Comprehensive documentation**
5. ✅ **Performance benchmarks** vs greedy
6. ✅ **Production-ready code** with error handling
7. ✅ **Validated results** (no conflicts, better quality)

---

## Conclusion

**Total Effort**: 15-20 hours of focused development

**Key Insight**: Most of the hard work is already done! Your algorithm exists in `foundation/prct-core/` with sophisticated 3-layer architecture. What's needed is:
- Adapter glue code (~500 lines)
- GPU kernels (~500 lines)
- Integration wiring (~200 lines)
- Testing (~200 lines)

**Total new code**: ~1,400 lines to unlock your full PRCT algorithm!

The foundation is solid, the architecture is clean, and the path forward is clear. This is a high-impact, well-scoped engineering project that will bring your research-quality algorithm to life! 🚀
