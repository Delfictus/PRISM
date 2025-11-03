/*
 * PRCT CUDA Kernels
 *
 * GPU-accelerated kernels for Phase Resonance Chromatic-TSP algorithm:
 * - Neuromorphic spike processing
 * - Quantum state evolution
 * - Kuramoto synchronization
 * - Phase coherence computations
 */

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// NEUROMORPHIC KERNELS
// ============================================================================

/**
 * Compute neuromorphic reservoir states from spike patterns
 * Each thread processes one neuron
 */
extern "C" __global__ void process_spikes_to_states(
    const int* spike_neuron_ids,
    const double* spike_amplitudes,
    int num_spikes,
    double* neuron_states,
    int* spike_counts,
    int num_neurons
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= num_neurons) return;

    double state = 0.0;
    int count = 0;

    // Accumulate spikes for this neuron
    for (int i = 0; i < num_spikes; i++) {
        if (spike_neuron_ids[i] == neuron_id) {
            state += spike_amplitudes[i];
            count++;
        }
    }

    neuron_states[neuron_id] = state;
    spike_counts[neuron_id] = count;
}

/**
 * Atomic max for double using compare-and-swap
 */
__device__ static void atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(old_val, val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
}

/**
 * Atomic add for double (for compute capability < 6.0)
 * Native atomicAdd(double*) available on CC >= 6.0, but we provide this for compatibility
 */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use native atomicAdd for doubles on CC >= 6.0
#else
__device__ static double atomicAdd(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = old_val + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/**
 * Normalize neuron states (find max, then divide all)
 * Phase 1: Find maximum (parallel reduction)
 */
extern "C" __global__ void find_max_state(
    const double* states,
    int num_states,
    double* max_out
) {
    __shared__ double shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    double val = (idx < num_states) ? states[idx] : 0.0;
    shared_max[tid] = val;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxDouble(max_out, shared_max[0]);
    }
}

/**
 * Normalize neuron states by maximum
 * Phase 2: Divide all states by max
 */
extern "C" __global__ void normalize_states(
    double* states,
    int num_states,
    double max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_states) return;
    if (max_val > 1e-10) {
        states[idx] /= max_val;
    }
}

/**
 * Compute phase coherence from neuron states
 * Converts states to phases and computes order parameter
 */
extern "C" __global__ void compute_phase_coherence(
    const double* states,
    int num_states,
    double* sum_cos,
    double* sum_sin
) {
    __shared__ double shared_cos[256];
    __shared__ double shared_sin[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert state to phase and compute trig functions
    double phase = 0.0;
    if (idx < num_states) {
        phase = states[idx] * 2.0 * M_PI;
    }

    shared_cos[tid] = (idx < num_states) ? cos(phase) : 0.0;
    shared_sin[tid] = (idx < num_states) ? sin(phase) : 0.0;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_cos[tid] += shared_cos[tid + stride];
            shared_sin[tid] += shared_sin[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_cos, shared_cos[0]);
        atomicAdd(sum_sin, shared_sin[0]);
    }
}

// ============================================================================
// QUANTUM KERNELS
// ============================================================================

/**
 * Apply quantum phase evolution to amplitudes
 * |ψ_i(t)> = |ψ_i(0)> * exp(-iE_i*t)
 */
extern "C" __global__ void quantum_phase_evolution(
    double* amplitudes_real,
    double* amplitudes_imag,
    const double* eigenvalues,
    double time,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_states) return;

    double energy = eigenvalues[idx];
    double phase = -energy * time;

    double cos_phase = cos(phase);
    double sin_phase = sin(phase);

    double re = amplitudes_real[idx];
    double im = amplitudes_imag[idx];

    // Complex multiplication: (re + i*im) * (cos - i*sin)
    amplitudes_real[idx] = re * cos_phase - im * sin_phase;
    amplitudes_imag[idx] = re * sin_phase + im * cos_phase;
}

/**
 * Normalize quantum state amplitudes
 * Phase 1: Compute norm squared
 */
extern "C" __global__ void compute_norm_squared(
    const double* amplitudes_real,
    const double* amplitudes_imag,
    int num_states,
    double* norm_sq_out
) {
    __shared__ double shared_norm[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double norm_sq = 0.0;
    if (idx < num_states) {
        double re = amplitudes_real[idx];
        double im = amplitudes_imag[idx];
        norm_sq = re * re + im * im;
    }

    shared_norm[tid] = norm_sq;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_norm[tid] += shared_norm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq_out, shared_norm[0]);
    }
}

/**
 * Normalize quantum amplitudes
 * Phase 2: Divide by sqrt(norm)
 */
extern "C" __global__ void normalize_amplitudes(
    double* amplitudes_real,
    double* amplitudes_imag,
    int num_states,
    double norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_states) return;
    if (norm > 1e-10) {
        amplitudes_real[idx] /= norm;
        amplitudes_imag[idx] /= norm;
    }
}

/**
 * Extract phases from quantum amplitudes
 */
extern "C" __global__ void extract_quantum_phases(
    const double* amplitudes_real,
    const double* amplitudes_imag,
    double* phases,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_states) return;

    double re = amplitudes_real[idx];
    double im = amplitudes_imag[idx];

    phases[idx] = atan2(im, re);
}

/**
 * Compute phase coherence matrix (pairwise coherence)
 */
extern "C" __global__ void compute_phase_coherence_matrix(
    const double* phases,
    int num_phases,
    double* coherence_matrix
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_phases || j >= num_phases) return;

    double phase_diff = fabs(phases[j] - phases[i]);

    // Normalize to [0, π]
    if (phase_diff > M_PI) {
        phase_diff = 2.0 * M_PI - phase_diff;
    }

    // Coherence is higher when phase difference is smaller
    double coherence = 1.0 - (phase_diff / M_PI);
    coherence_matrix[i * num_phases + j] = coherence;
}

// ============================================================================
// KURAMOTO SYNCHRONIZATION KERNELS
// ============================================================================

/**
 * Kuramoto phase update step
 * dθ_i/dt = ω_i + (K/N) * sum_j(sin(θ_j - θ_i))
 */
extern "C" __global__ void kuramoto_step(
    const double* phases,
    const double* natural_frequencies,
    const double* coupling_matrix,
    double coupling_strength,
    double dt,
    double* new_phases,
    int num_oscillators
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_oscillators) return;

    // Natural frequency term
    double phase_derivative = natural_frequencies[i];

    // Coupling term
    double coupling_sum = 0.0;
    for (int j = 0; j < num_oscillators; j++) {
        if (i != j) {
            double coupling_ij = coupling_matrix[i * num_oscillators + j];
            double phase_diff = phases[j] - phases[i];
            coupling_sum += coupling_ij * sin(phase_diff);
        }
    }

    phase_derivative += (coupling_strength / num_oscillators) * coupling_sum;

    // Update phase
    double new_phase = phases[i] + phase_derivative * dt;

    // Keep in [0, 2π]
    new_phase = fmod(new_phase, 2.0 * M_PI);
    if (new_phase < 0.0) new_phase += 2.0 * M_PI;

    new_phases[i] = new_phase;
}

/**
 * Compute Kuramoto order parameter
 * r = |<e^(iθ)>| = sqrt((1/N * sum cos(θ))^2 + (1/N * sum sin(θ))^2)
 */
extern "C" __global__ void kuramoto_order_parameter(
    const double* phases,
    int num_phases,
    double* sum_cos_out,
    double* sum_sin_out
) {
    __shared__ double shared_cos[256];
    __shared__ double shared_sin[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double phase = (idx < num_phases) ? phases[idx] : 0.0;

    shared_cos[tid] = (idx < num_phases) ? cos(phase) : 0.0;
    shared_sin[tid] = (idx < num_phases) ? sin(phase) : 0.0;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_cos[tid] += shared_cos[tid + stride];
            shared_sin[tid] += shared_sin[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_cos_out, shared_cos[0]);
        atomicAdd(sum_sin_out, shared_sin[0]);
    }
}

/**
 * Compute local phase coherence for each oscillator
 */
extern "C" __global__ void compute_local_coherence(
    const double* phases,
    const double* coupling_matrix,
    double* coherence_levels,
    int num_oscillators
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_oscillators) return;

    double total_coherence = 0.0;
    int neighbor_count = 0;

    for (int j = 0; j < num_oscillators; j++) {
        if (i != j && coupling_matrix[i * num_oscillators + j] > 0.0) {
            double phase_diff = fabs(phases[j] - phases[i]);
            if (phase_diff > M_PI) {
                phase_diff = 2.0 * M_PI - phase_diff;
            }
            double coherence = 1.0 - (phase_diff / M_PI);
            total_coherence += coherence;
            neighbor_count++;
        }
    }

    if (neighbor_count > 0) {
        coherence_levels[i] = total_coherence / neighbor_count;
    } else {
        coherence_levels[i] = 1.0;
    }
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * Compute correlation between two time series (for transfer entropy)
 */
extern "C" __global__ void compute_correlation(
    const double* source,
    const double* target,
    int length,
    double source_mean,
    double target_mean,
    double* covariance_out,
    double* source_var_out,
    double* target_var_out
) {
    __shared__ double shared_cov[256];
    __shared__ double shared_var_s[256];
    __shared__ double shared_var_t[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double cov = 0.0;
    double var_s = 0.0;
    double var_t = 0.0;

    if (idx < length) {
        double s_dev = source[idx] - source_mean;
        double t_dev = target[idx] - target_mean;
        cov = s_dev * t_dev;
        var_s = s_dev * s_dev;
        var_t = t_dev * t_dev;
    }

    shared_cov[tid] = cov;
    shared_var_s[tid] = var_s;
    shared_var_t[tid] = var_t;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_cov[tid] += shared_cov[tid + stride];
            shared_var_s[tid] += shared_var_s[tid + stride];
            shared_var_t[tid] += shared_var_t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(covariance_out, shared_cov[0]);
        atomicAdd(source_var_out, shared_var_s[0]);
        atomicAdd(target_var_out, shared_var_t[0]);
    }
}
