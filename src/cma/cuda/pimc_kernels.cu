// Path Integral Monte Carlo CUDA Kernels
// Constitution: Phase 6 Implementation Constitution - Sprint 1.3
//
// Implements GPU-accelerated quantum annealing via PIMC

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

extern "C" {

// Update all beads in parallel
__global__ void update_beads_kernel(
    float* path,              // [n_beads * n_dim] - worldline coordinates
    const float* hamiltonian, // [n_dim * n_dim] - problem Hamiltonian matrix
    const float* manifold_edges, // [n_edges * 4] - (src, tgt, TE, penalty)
    int n_edges,
    int n_beads,
    int n_dim,
    float beta,
    float tau,
    float mass,
    float tunneling_strength,
    curandState* rand_states,
    int* accepted_moves      // Output: count of accepted moves
) {
    int bead_id = blockIdx.x;
    int dim_id = threadIdx.x;

    if (bead_id >= n_beads || dim_id >= n_dim) return;

    int idx = bead_id * n_dim + dim_id;

    // Get random state for this thread
    curandState local_state = rand_states[idx];

    float old_value = path[idx];

    // Propose new value
    float step_size = 0.1f * tunneling_strength;
    float uniform = curand_uniform(&local_state);
    float new_value = old_value + (uniform - 0.5f) * 2.0f * step_size;

    // Compute kinetic action change
    int prev_bead = (bead_id == 0) ? (n_beads - 1) : (bead_id - 1);
    int next_bead = (bead_id + 1) % n_beads;

    int prev_idx = prev_bead * n_dim + dim_id;
    int next_idx = next_bead * n_dim + dim_id;

    float x_prev = path[prev_idx];
    float x_next = path[next_idx];

    // Kinetic action: m/(2τ²) [(x - x_prev)² + (x_next - x)²]
    float kinetic_factor = mass / (2.0f * tau * tau);

    float old_kinetic = kinetic_factor * (
        (old_value - x_prev) * (old_value - x_prev) +
        (x_next - old_value) * (x_next - old_value)
    );

    float new_kinetic = kinetic_factor * (
        (new_value - x_prev) * (new_value - x_prev) +
        (x_next - new_value) * (x_next - new_value)
    );

    // Compute potential energy change
    // V(x) = Σ_ij H_ij x_i x_j + manifold penalties
    float old_potential = 0.0f;
    float new_potential = 0.0f;

    // Quadratic term from Hamiltonian
    for (int j = 0; j < n_dim; j++) {
        int other_idx = bead_id * n_dim + j;
        float x_j = path[other_idx];

        int h_idx = dim_id * n_dim + j;
        float h_ij = hamiltonian[h_idx];

        old_potential += h_ij * old_value * x_j;
        new_potential += h_ij * new_value * x_j;
    }

    // Manifold constraint penalties
    for (int e = 0; e < n_edges; e++) {
        int src = (int)manifold_edges[e * 4 + 0];
        int tgt = (int)manifold_edges[e * 4 + 1];
        float te = manifold_edges[e * 4 + 2];
        float coupling = manifold_edges[e * 4 + 3];

        if (src == dim_id || tgt == dim_id) {
            int other_dim = (src == dim_id) ? tgt : src;
            if (other_dim < n_dim) {
                int other_idx = bead_id * n_dim + other_dim;
                float x_other = path[other_idx];

                old_potential += coupling * te * fabsf(old_value - x_other);
                new_potential += coupling * te * fabsf(new_value - x_other);
            }
        }
    }

    // Total action change
    float delta_action = (new_kinetic - old_kinetic) + tau * (new_potential - old_potential);

    // Metropolis acceptance
    float accept_prob = expf(-delta_action);
    float rand_val = curand_uniform(&local_state);

    if (rand_val < accept_prob) {
        path[idx] = new_value;
        atomicAdd(accepted_moves, 1);
    }

    // Save random state
    rand_states[idx] = local_state;
}

// Initialize random states for each thread
__global__ void init_rand_states_kernel(
    curandState* states,
    unsigned long long seed,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curand_init(seed, idx, 0, &states[idx]);
}

// Compute energy of entire path
__global__ void compute_path_energy_kernel(
    const float* path,
    const float* hamiltonian,
    float* energies,    // [n_beads] output
    int n_beads,
    int n_dim
) {
    int bead = blockIdx.x * blockDim.x + threadIdx.x;
    if (bead >= n_beads) return;

    float energy = 0.0f;

    // Extract this bead's configuration
    const float* x = path + bead * n_dim;

    // Compute H(x) = x^T H x
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            energy += x[i] * hamiltonian[i * n_dim + j] * x[j];
        }
    }

    energies[bead] = energy;
}

// Reduce energies to average
__global__ void reduce_average_kernel(
    const float* values,
    float* result,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? values[i] : 0.0f;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0] / (float)n);
    }
}

// Compute spectral gap for adaptive scheduling
__global__ void compute_spectral_gap_kernel(
    const float* hamiltonian,
    float* eigenvalues,  // [2] - two lowest eigenvalues
    int n_dim
) {
    // Simplified: use power iteration for smallest eigenvalue
    // Full implementation would use more sophisticated methods

    __shared__ float v[256];  // Eigenvector approximation

    int tid = threadIdx.x;

    if (tid < n_dim && tid < 256) {
        v[tid] = 1.0f / sqrtf((float)n_dim);
    }
    __syncthreads();

    // Power iteration (simplified)
    for (int iter = 0; iter < 100; iter++) {
        __shared__ float v_new[256];

        if (tid < n_dim && tid < 256) {
            float sum = 0.0f;
            for (int j = 0; j < n_dim && j < 256; j++) {
                sum += hamiltonian[tid * n_dim + j] * v[j];
            }
            v_new[tid] = sum;
        }
        __syncthreads();

        // Normalize
        if (tid == 0) {
            float norm = 0.0f;
            for (int i = 0; i < n_dim && i < 256; i++) {
                norm += v_new[i] * v_new[i];
            }
            norm = sqrtf(norm);

            for (int i = 0; i < n_dim && i < 256; i++) {
                v[i] = v_new[i] / norm;
            }
        }
        __syncthreads();
    }

    // Rayleigh quotient for eigenvalue
    if (tid == 0) {
        float numerator = 0.0f;
        for (int i = 0; i < n_dim && i < 256; i++) {
            float Hv_i = 0.0f;
            for (int j = 0; j < n_dim && j < 256; j++) {
                Hv_i += hamiltonian[i * n_dim + j] * v[j];
            }
            numerator += v[i] * Hv_i;
        }
        eigenvalues[0] = numerator;
    }
}

// Apply manifold projection to path
__global__ void project_onto_manifold_kernel(
    float* path,
    const float* metric_tensor,
    const float* causal_edges,
    int n_edges,
    int n_beads,
    int n_dim
) {
    int bead = blockIdx.x;
    int dim = threadIdx.x;

    if (bead >= n_beads || dim >= n_dim) return;

    int idx = bead * n_dim + dim;
    float x = path[idx];

    // Project using metric tensor
    float projected = x;

    for (int e = 0; e < n_edges; e++) {
        int src = (int)causal_edges[e * 4 + 0];
        int tgt = (int)causal_edges[e * 4 + 1];
        float strength = causal_edges[e * 4 + 2];

        if (src == dim && tgt < n_dim) {
            int tgt_idx = bead * n_dim + tgt;
            float x_tgt = path[tgt_idx];

            // Soft projection towards causal partner
            projected += 0.1f * strength * (x_tgt - x);
        } else if (tgt == dim && src < n_dim) {
            int src_idx = bead * n_dim + src;
            float x_src = path[src_idx];

            projected += 0.1f * strength * (x_src - x);
        }
    }

    path[idx] = projected;
}

} // extern "C"