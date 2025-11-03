// GPU Policy Evaluation Kernels
// Constitution: Article VII - PTX Runtime Loading
//
// Implements GPU-accelerated policy evaluation for Active Inference:
// - Hierarchical physics simulation (satellite, atmosphere, windows)
// - Trajectory prediction for multiple policies in parallel
// - Expected free energy (EFE) computation
//
// Target: 231ms CPU → <10ms GPU (23x speedup)

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Constants
#define MAX_POLICIES 5
#define HORIZON 3
#define SATELLITE_DIM 6
#define ATMOSPHERE_MODES 50
#define N_WINDOWS 900
#define OBS_DIM 100

// Gravitational parameter (Earth)
#define MU_EARTH 3.986004418e14  // m³/s²

//==============================================================================
// KERNEL 1: Satellite Orbital Evolution (Verlet Integration)
//==============================================================================

/// Evolve satellite orbital state using Verlet integration
///
/// State: [r_x, r_y, r_z, v_x, v_y, v_z] (6 dimensions)
/// Dynamics: d²r/dt² = -μ·r/|r|³ (Keplerian)
///
/// Grid: (n_policies, 1, 1)
/// Block: (6, 1, 1) - One thread per state dimension
__global__ void evolve_satellite_kernel(
    const double* __restrict__ current_state,  // [n_policies × 6]
    double* __restrict__ next_state,           // [n_policies × 6]
    double dt,
    int n_policies
) {
    int policy_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (policy_idx >= n_policies || dim_idx >= SATELLITE_DIM) return;

    // Shared memory for position and velocity
    __shared__ double position[3];
    __shared__ double velocity[3];
    __shared__ double acceleration[3];
    __shared__ double new_position[3];
    __shared__ double new_acceleration[3];

    // Load current state
    int state_offset = policy_idx * SATELLITE_DIM;
    if (dim_idx < 3) {
        position[dim_idx] = current_state[state_offset + dim_idx];
        velocity[dim_idx] = current_state[state_offset + 3 + dim_idx];
    }
    __syncthreads();

    // Compute current acceleration: a = -μ·r/|r|³
    if (dim_idx == 0) {
        double r_mag = sqrt(position[0]*position[0] +
                           position[1]*position[1] +
                           position[2]*position[2]);
        double r_mag_cubed = r_mag * r_mag * r_mag;
        double factor = -MU_EARTH / r_mag_cubed;

        acceleration[0] = factor * position[0];
        acceleration[1] = factor * position[1];
        acceleration[2] = factor * position[2];
    }
    __syncthreads();

    // Verlet position update: r_{n+1} = r_n + v_n·dt + 0.5·a_n·dt²
    if (dim_idx < 3) {
        new_position[dim_idx] = position[dim_idx] +
                                velocity[dim_idx] * dt +
                                0.5 * acceleration[dim_idx] * dt * dt;
    }
    __syncthreads();

    // Compute new acceleration at predicted position
    if (dim_idx == 0) {
        double r_mag = sqrt(new_position[0]*new_position[0] +
                           new_position[1]*new_position[1] +
                           new_position[2]*new_position[2]);
        double r_mag_cubed = r_mag * r_mag * r_mag;
        double factor = -MU_EARTH / r_mag_cubed;

        new_acceleration[0] = factor * new_position[0];
        new_acceleration[1] = factor * new_position[1];
        new_acceleration[2] = factor * new_position[2];
    }
    __syncthreads();

    // Verlet velocity update: v_{n+1} = v_n + 0.5·(a_n + a_{n+1})·dt
    double new_velocity = 0.0;
    if (dim_idx < 3) {
        new_velocity = velocity[dim_idx] +
                      0.5 * (acceleration[dim_idx] + new_acceleration[dim_idx]) * dt;
    }

    // Write output
    int out_offset = policy_idx * SATELLITE_DIM;
    if (dim_idx < 3) {
        next_state[out_offset + dim_idx] = new_position[dim_idx];
        next_state[out_offset + 3 + dim_idx] = new_velocity;
    }

    // Variance update (small increase due to perturbations)
    // Not implemented here - handle in separate variance kernel if needed
}

//==============================================================================
// KERNEL 2: Atmospheric Turbulence Evolution
//==============================================================================

/// Evolve atmospheric turbulence modes
///
/// Dynamics: dφ/dt = -λ·φ + √(2λ·C_n²)·η(t)
/// (Ornstein-Uhlenbeck process reaching stationary variance C_n²)
///
/// Grid: (n_policies, 1, 1)
/// Block: (n_modes, 1, 1) - One thread per mode
__global__ void evolve_atmosphere_kernel(
    const double* __restrict__ current_modes,   // [n_policies × n_modes]
    const double* __restrict__ current_variances, // [n_policies × n_modes]
    double* __restrict__ next_modes,            // [n_policies × n_modes]
    double* __restrict__ next_variances,        // [n_policies × n_modes]
    curandState* __restrict__ rng_states,       // [n_policies × n_modes]
    double dt,
    double decorrelation_rate,
    double c_n_squared,
    int n_policies,
    int n_modes
) {
    int policy_idx = blockIdx.x;
    int mode_idx = threadIdx.x;

    if (policy_idx >= n_policies || mode_idx >= n_modes) return;

    int idx = policy_idx * n_modes + mode_idx;

    // Load current state
    double current_mode = current_modes[idx];
    double current_var = current_variances[idx];

    // Decorrelation: φ_{t+1} = φ_t · exp(-λ·dt)
    double decay_factor = exp(-decorrelation_rate * dt);
    double decorrelated = current_mode * decay_factor;

    // Noise injection: √(2λ·C_n²·dt) · N(0,1)
    double noise_scale = sqrt(2.0 * decorrelation_rate * c_n_squared * dt);

    // Generate random noise
    curandState local_rng = rng_states[idx];
    double noise = curand_normal_double(&local_rng);
    rng_states[idx] = local_rng; // Save RNG state

    // Update mode
    next_modes[idx] = decorrelated + noise_scale * noise;

    // Variance evolution: approaches stationary value C_n²
    double target_variance = c_n_squared;
    next_variances[idx] = current_var + (target_variance - current_var) * dt;
}

//==============================================================================
// KERNEL 3: Window Phase Evolution (Langevin Dynamics)
//==============================================================================

/// Evolve window phases using Langevin dynamics
///
/// Dynamics: dφ/dt = -γ·φ + coupling·sin(φ_atm) + control + √(2D)·η(t)
///
/// Grid: (n_policies × horizon, 1, 1)
/// Block: (256, 1, 1) - Multiple threads handle 900 windows in chunks
__global__ void evolve_windows_kernel(
    const double* __restrict__ current_windows,    // [n_policies × horizon × n_windows]
    const double* __restrict__ current_variances,  // [n_policies × horizon × n_windows]
    const double* __restrict__ atmosphere_modes,   // [n_policies × horizon × n_modes]
    const double* __restrict__ actions,            // [n_policies × horizon × n_windows]
    double* __restrict__ next_windows,             // [n_policies × horizon × n_windows]
    double* __restrict__ next_variances,           // [n_policies × horizon × n_windows]
    curandState* __restrict__ rng_states,          // [n_policies × n_windows]
    double dt,
    double damping,
    double diffusion,
    int n_policies,
    int horizon,
    int n_windows,
    int n_modes,
    int substeps
) {
    int ph_idx = blockIdx.x;  // Policy × horizon combined
    int policy_idx = ph_idx / horizon;
    int step_idx = ph_idx % horizon;

    if (policy_idx >= n_policies || step_idx >= horizon) return;

    // Each thread handles multiple windows
    for (int win_idx = threadIdx.x; win_idx < n_windows; win_idx += blockDim.x) {
        int state_idx = ph_idx * n_windows + win_idx;
        int rng_idx = policy_idx * n_windows + win_idx;

        double window_phase = current_windows[state_idx];
        double window_var = current_variances[state_idx];

        // Project atmospheric turbulence onto this window
        // Simplified: weighted sum of nearby modes
        int mode_idx = (win_idx * n_modes) / n_windows;
        int atm_offset = ph_idx * n_modes;

        double atm_drive = atmosphere_modes[atm_offset + mode_idx];
        if (mode_idx > 0) {
            atm_drive += 0.3 * atmosphere_modes[atm_offset + mode_idx - 1];
        }
        if (mode_idx + 1 < n_modes) {
            atm_drive += 0.3 * atmosphere_modes[atm_offset + mode_idx + 1];
        }

        // Load control action
        double control = actions[state_idx];

        // Multiple substeps for numerical stability
        curandState local_rng = rng_states[rng_idx];

        for (int sub = 0; sub < substeps; sub++) {
            double dt_sub = dt / substeps;

            // Drift: f(φ) = -γ·φ + coupling·sin(φ_atm)
            double drift = -damping * window_phase + 0.5 * sin(atm_drive);

            // Diffusion noise: √(2D·dt)·η
            double noise_scale = sqrt(2.0 * diffusion * dt_sub);
            double noise = curand_normal_double(&local_rng);

            // Euler-Maruyama update
            window_phase += drift * dt_sub + noise_scale * noise;

            // Apply control (negative to cancel aberrations)
            if (sub == 0) {
                window_phase -= control;
            }
        }

        rng_states[rng_idx] = local_rng; // Save RNG state

        // Variance evolution: dΣ/dt = -2γ·Σ + 2D
        double steady_state_var = diffusion / damping;
        window_var += (-2.0 * damping * window_var + 2.0 * diffusion) * dt;

        // Clamp variance
        window_var = fmax(1e-6, fmin(window_var, steady_state_var * 10.0));

        // Write output
        next_windows[state_idx] = window_phase;
        next_variances[state_idx] = window_var;
    }
}

//==============================================================================
// KERNEL 4: Observation Prediction
//==============================================================================

/// Predict observations from state
///
/// o = C * x (observation matrix × state vector)
///
/// NOTE: This should use cuBLAS cublasDgemv for efficiency
/// This kernel is backup if cuBLAS not available
///
/// Grid: (n_policies × horizon, 1, 1)
/// Block: (obs_dim, 1, 1)
__global__ void predict_observations_kernel(
    const double* __restrict__ state_windows,      // [n_policies × horizon × n_windows]
    const double* __restrict__ state_variances,    // [n_policies × horizon × n_windows]
    const double* __restrict__ observation_matrix, // [obs_dim × n_windows]
    const double* __restrict__ observation_noise,  // [obs_dim]
    double* __restrict__ predicted_obs,            // [n_policies × horizon × obs_dim]
    double* __restrict__ obs_variances,            // [n_policies × horizon × obs_dim]
    int n_policies,
    int horizon,
    int n_windows,
    int obs_dim
) {
    int ph_idx = blockIdx.x;
    int obs_idx = threadIdx.x;

    if (obs_idx >= obs_dim) return;

    int state_offset = ph_idx * n_windows;
    int obs_offset = ph_idx * obs_dim + obs_idx;

    // Matrix-vector multiply: o_i = Σ_j C_ij * x_j
    double pred_mean = 0.0;
    double pred_var = observation_noise[obs_idx];

    for (int j = 0; j < n_windows; j++) {
        double c_ij = observation_matrix[obs_idx * n_windows + j];
        double x_j = state_windows[state_offset + j];
        double var_j = state_variances[state_offset + j];

        // Mean: o = C * x
        pred_mean += c_ij * x_j;

        // Variance: σ²_o = Σ_j (C_ij)² * σ²_x,j + R
        pred_var += c_ij * c_ij * var_j;
    }

    predicted_obs[obs_offset] = pred_mean;
    obs_variances[obs_offset] = pred_var;
}

//==============================================================================
// KERNEL 5: Expected Free Energy Computation
//==============================================================================

/// Compute EFE components: Risk, Ambiguity, Novelty
///
/// Risk: E[(o_pred - o_pref)²] - deviation from goal
/// Ambiguity: E[σ²_o] - observation uncertainty
/// Novelty: H(prior) - H(posterior) - information gain
///
/// Grid: (n_policies, 1, 1)
/// Block: (256, 1, 1) - Parallel reduction
__global__ void compute_efe_kernel(
    const double* __restrict__ predicted_obs,      // [n_policies × horizon × obs_dim]
    const double* __restrict__ obs_variances,      // [n_policies × horizon × obs_dim]
    const double* __restrict__ preferred_obs,      // [obs_dim]
    const double* __restrict__ future_variances,   // [n_policies × horizon × n_windows]
    const double* __restrict__ prior_variance,     // [n_windows]
    double* __restrict__ risk_out,                 // [n_policies]
    double* __restrict__ ambiguity_out,            // [n_policies]
    double* __restrict__ novelty_out,              // [n_policies]
    int n_policies,
    int horizon,
    int obs_dim,
    int n_windows
) {
    int policy_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (policy_idx >= n_policies) return;

    // Shared memory for reduction
    __shared__ double shared_risk;
    __shared__ double shared_ambiguity;
    __shared__ double shared_novelty;

    // Initialize shared memory
    if (thread_idx == 0) {
        shared_risk = 0.0;
        shared_ambiguity = 0.0;
        shared_novelty = 0.0;
    }
    __syncthreads();

    // Accumulate over trajectory
    for (int step = 0; step < horizon; step++) {
        int obs_offset = (policy_idx * horizon + step) * obs_dim;
        int state_offset = (policy_idx * horizon + step) * n_windows;

        // Risk: Σ (o_pred - o_pref)²
        for (int i = thread_idx; i < obs_dim; i += blockDim.x) {
            double error = predicted_obs[obs_offset + i] - preferred_obs[i];
            atomicAdd(&shared_risk, error * error);
        }

        // Ambiguity: Σ σ²_o
        for (int i = thread_idx; i < obs_dim; i += blockDim.x) {
            atomicAdd(&shared_ambiguity, obs_variances[obs_offset + i]);
        }

        // Novelty: H(prior) - H(posterior)
        // H(Gaussian) = 0.5 * ln((2πe)^n * |Σ|)
        // For diagonal: ln|Σ| = Σ ln(σ²)
        for (int i = thread_idx; i < n_windows; i += blockDim.x) {
            double prior_var = prior_variance[i];
            double post_var = future_variances[state_offset + i];

            // Guard against log(0) or log(negative)
            if (prior_var > 1e-10 && post_var > 1e-10) {
                // Contribution to entropy difference
                double entropy_diff = 0.5 * (log(prior_var) - log(post_var));
                atomicAdd(&shared_novelty, entropy_diff);
            }
        }
    }
    __syncthreads();

    // Normalize by horizon and write results
    if (thread_idx == 0) {
        double h = (double)horizon;
        risk_out[policy_idx] = shared_risk / h;
        ambiguity_out[policy_idx] = shared_ambiguity / h;
        novelty_out[policy_idx] = shared_novelty / h;
    }
}

//==============================================================================
// KERNEL 6: RNG Initialization
//==============================================================================

/// Initialize cuRAND states for all policies and dimensions
///
/// Grid: (n_policies, 1, 1)
/// Block: (n_windows, 1, 1) or (n_modes, 1, 1)
__global__ void init_rng_states_kernel(
    curandState* __restrict__ states,
    unsigned long long seed,
    int n_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_states) {
        // Initialize with unique sequence for each state
        curand_init(seed, idx, 0, &states[idx]);
    }
}

//==============================================================================
// KERNEL 7: Full Trajectory Prediction (Orchestrator)
//==============================================================================

/// Predict full multi-step trajectory for all policies in parallel
///
/// For each policy:
///   For each time step in horizon:
///     1. Evolve satellite
///     2. Evolve atmosphere
///     3. Evolve windows
///     4. Store future state
///
/// This kernel orchestrates the hierarchical evolution
/// Can be split into multiple kernel launches for flexibility
extern "C" __global__ void predict_trajectories_kernel(
    // Initial states
    const double* __restrict__ initial_satellite,   // [n_policies × 6]
    const double* __restrict__ initial_atmosphere,  // [n_policies × n_modes]
    const double* __restrict__ initial_windows,     // [n_policies × n_windows]
    const double* __restrict__ initial_variances,   // [n_policies × (6 + n_modes + n_windows)]

    // Policy actions
    const double* __restrict__ actions,             // [n_policies × horizon × n_windows]

    // Outputs (trajectories)
    double* __restrict__ future_satellite,          // [n_policies × horizon × 6]
    double* __restrict__ future_atmosphere,         // [n_policies × horizon × n_modes]
    double* __restrict__ future_windows,            // [n_policies × horizon × n_windows]
    double* __restrict__ future_variances,          // [n_policies × horizon × n_windows]

    // RNG states
    curandState* __restrict__ rng_states_atm,       // [n_policies × n_modes]
    curandState* __restrict__ rng_states_win,       // [n_policies × n_windows]

    // Parameters
    double dt_satellite,
    double dt_atmosphere,
    double dt_windows,
    double damping,
    double diffusion,
    double decorrelation_rate,
    double c_n_squared,
    int n_policies,
    int horizon,
    int substeps
) {
    // This is a high-level orchestrator
    // In practice, we'll launch separate kernels for each evolution step
    // This kernel documents the overall structure

    int policy_idx = blockIdx.x;
    if (policy_idx >= n_policies) return;

    // NOTE: This kernel is conceptual
    // Actual implementation will use separate kernel launches:
    // 1. evolve_satellite_kernel
    // 2. evolve_atmosphere_kernel
    // 3. evolve_windows_kernel (with substeps)
    // in sequence for each timestep
}

//==============================================================================
// KERNEL 8: Matrix-Vector Multiply (Backup - Prefer cuBLAS)
//==============================================================================

/// Matrix-vector multiply: y = A * x
///
/// NOTE: Use cuBLAS cublasDgemv instead for production
/// This is backup implementation
///
/// Grid: (1, 1, 1)
/// Block: (m, 1, 1) - One thread per output element
__global__ void matvec_kernel(
    const double* __restrict__ matrix,  // [m × n]
    const double* __restrict__ vector,  // [n]
    double* __restrict__ result,        // [m]
    int m,
    int n
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < m) {
        double sum = 0.0;
        for (int col = 0; col < n; col++) {
            sum += matrix[row * n + col] * vector[col];
        }
        result[row] = sum;
    }
}

//==============================================================================
// KERNEL 9: Reduction (Sum)
//==============================================================================

/// Parallel reduction to sum array elements
///
/// Grid: (1, 1, 1)
/// Block: (256, 1, 1)
__global__ void sum_reduction_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n
) {
    __shared__ double shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// End of policy_evaluation.cu
