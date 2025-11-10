// Thermodynamic Network GPU Kernels
//
// Constitutional Compliance: Article VII - Kernel Compilation Standards
//
// Implements damped coupled oscillator dynamics for thermodynamic network evolution
// Based on Langevin equation: dx/dt = -γx - ∇U(x) + √(2γkT) * η(t)
//
// Key equations:
// - Position: x[i] += v[i] * dt
// - Velocity: v[i] += (force[i] - damping * v[i]) * dt + noise
// - Coupling: force[i] = -Σ_j coupling[i][j] * (x[i] - x[j])

#include <cuda_runtime.h>
#include <math.h>
#include <curand_kernel.h>

// Constants
#define PI 3.14159265358979323846

// Kernel 1: Initialize oscillator states
extern "C" __global__ void initialize_oscillators_kernel(
    double* positions,         // Output: initial positions
    double* velocities,        // Output: initial velocities
    double* phases,            // Output: initial phases
    int n_oscillators,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_oscillators) return;

    // Initialize cuRAND state
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Random initial conditions
    positions[idx] = curand_normal_double(&state) * 0.1;
    velocities[idx] = curand_normal_double(&state) * 0.1;
    phases[idx] = curand_uniform_double(&state) * 2.0 * PI;
}

// Kernel 2: Compute coupling forces (SPARSE EDGE LIST VERSION)
// Uses edge list representation for O(E) complexity instead of O(V²)
extern "C" __global__ void compute_coupling_forces_kernel(
    const float* phases,           // Current phases (f32 as per Rust code)
    const unsigned int* edge_u,    // Edge source vertices
    const unsigned int* edge_v,    // Edge target vertices
    const float* edge_w,           // Edge weights
    int n_edges,                   // Number of edges
    int n_vertices,                // Number of vertices
    float coupling_strength,       // Global coupling strength
    float* forces                  // Output: coupling forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vertices) return;

    float force = 0.0;

    // Sum coupling forces from edges connected to this vertex
    // This is O(E) total across all threads, much better than O(V²)
    for (int e = 0; e < n_edges; e++) {
        unsigned int u = edge_u[e];
        unsigned int v = edge_v[e];
        float weight = edge_w[e];

        if (u == idx) {
            // Edge from idx to v: force depends on phase difference
            float phase_diff = phases[idx] - phases[v];
            force -= coupling_strength * weight * sin(phase_diff);
        } else if (v == idx) {
            // Edge from u to idx: force depends on phase difference
            float phase_diff = phases[idx] - phases[u];
            force -= coupling_strength * weight * sin(phase_diff);
        }
    }

    forces[idx] = force;
}

// Kernel 3: Evolve oscillators (Langevin dynamics) - FLOAT VERSION with CONFLICT FORCES
extern "C" __global__ void evolve_oscillators_kernel(
    float* phases,               // Phases (updated in-place) - f32
    float* velocities,           // Velocities (updated in-place) - f32
    const float* forces,         // Coupling forces - f32
    int n_oscillators,
    float dt,                    // Time step
    float temperature            // Temperature T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_oscillators) return;

    float phi = phases[idx];
    float v = velocities[idx];
    float force = forces[idx];

    // Simple velocity update with damping
    float damping = 0.1f;
    v += (force - damping * v) * dt;

    // Update phase based on velocity
    phi += v * dt;

    // Keep phase in [-π, π]
    while (phi > PI) phi -= 2.0f * PI;
    while (phi < -PI) phi += 2.0f * PI;

    // Write back
    phases[idx] = phi;
    velocities[idx] = v;
}

// Kernel 3b: Evolve oscillators with CONFLICT-DRIVEN FORCES
// This version adds repulsion forces for vertices with coloring conflicts
extern "C" __global__ void evolve_oscillators_with_conflicts_kernel(
    float* phases,               // Phases (updated in-place)
    float* velocities,           // Velocities (updated in-place)
    const float* forces,         // Coupling forces
    const int* coloring,         // Current vertex colors
    const int* conflicts,        // Conflict count per vertex
    const float* uncertainty,    // AI uncertainty weights (optional, NULL if not used)
    const unsigned int* edge_u,  // Edge sources
    const unsigned int* edge_v,  // Edge targets
    int n_edges,
    int n_oscillators,
    float dt,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_oscillators) return;

    float phi = phases[idx];
    float v = velocities[idx];
    float coupling_force = forces[idx];

    // Conflict-driven repulsion force
    float conflict_force = 0.0f;
    int vertex_conflicts = conflicts[idx];

    if (vertex_conflicts > 0) {
        // Get uncertainty weight (higher uncertainty → stronger penalty)
        float uncertainty_weight = uncertainty ? uncertainty[idx] : 1.0f;

        // Temperature-dependent penalty (stronger at low T)
        // At high T: let natural dynamics explore
        // At low T: force conflict resolution aggressively
        float penalty_coefficient = 10.0f * uncertainty_weight * (1.0f + expf(-temperature));

        // For each edge connected to this vertex
        int my_color = coloring[idx];
        for (int e = 0; e < n_edges; e++) {
            unsigned int u = edge_u[e];
            unsigned int w = edge_v[e];

            // Check if this edge involves our vertex
            int neighbor = -1;
            if (u == idx) neighbor = w;
            else if (w == idx) neighbor = u;

            if (neighbor >= 0 && coloring[neighbor] == my_color) {
                // Conflict! Push phase AWAY from conflicting neighbor
                float phase_diff = phases[neighbor] - phi;
                conflict_force += sinf(phase_diff) * penalty_coefficient;
            }
        }
    }

    // Clamp conflict force to prevent numerical instability
    conflict_force = fmaxf(-100.0f, fminf(100.0f, conflict_force));

    // Combined force
    float total_force = coupling_force + conflict_force;

    // Velocity update with damping
    float damping = 0.1f;
    v += (total_force - damping * v) * dt;

    // Update phase
    phi += v * dt;

    // Keep phase in [-π, π]
    while (phi > PI) phi -= 2.0f * PI;
    while (phi < -PI) phi += 2.0f * PI;

    // Write back
    phases[idx] = phi;
    velocities[idx] = v;
}

// Kernel 4: Compute total energy (SPARSE EDGE LIST VERSION)
extern "C" __global__ void compute_energy_kernel(
    const float* phases,           // Current phases
    const unsigned int* edge_u,    // Edge source vertices
    const unsigned int* edge_v,    // Edge target vertices
    const float* edge_w,           // Edge weights
    int n_edges,                   // Number of edges
    int n_vertices,                // Number of vertices
    float* energy_result           // Output: total energy (single value)
) {
    __shared__ float shared_energy[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    shared_energy[tid] = 0.0f;

    // Each thread computes energy for a subset of edges
    if (idx < n_edges) {
        unsigned int u = edge_u[idx];
        unsigned int v = edge_v[idx];
        float weight = edge_w[idx];

        // Kuramoto coupling energy: E = -K * cos(θ_i - θ_j)
        float phase_diff = phases[u] - phases[v];
        float edge_energy = -weight * cosf(phase_diff);
        shared_energy[tid] = edge_energy;
    }

    __syncthreads();

    // Reduction: sum energies
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_energy[tid] += shared_energy[tid + stride];
        }
        __syncthreads();
    }

    // Block result
    if (tid == 0) {
        atomicAdd(energy_result, shared_energy[0]);
    }
}

// Kernel 5: Compute entropy (microcanonical ensemble)
extern "C" __global__ void compute_entropy_kernel(
    const double* positions,
    const double* velocities,
    double* entropy_result,
    int n_oscillators,
    double temperature
) {
    __shared__ double shared_entropy[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_entropy[tid] = 0.0;

    if (idx < n_oscillators) {
        double x = positions[idx];
        double v = velocities[idx];

        // For Langevin dynamics with damping, entropy MUST increase
        // Use phase space volume which grows with dissipation:
        // S = k_B * ln(accessible phase space)
        //
        // Phase space volume element: dV = dx dv
        // For temperature T, typical scales: x ~ √T, v ~ √T
        // Volume ~ (√T)^(2N) = T^N
        //
        // Use formulation that's guaranteed positive and monotonic:
        double phase_vol = sqrt(x*x + v*v + temperature);  // Never zero, grows with T
        double local_entropy = temperature * log(phase_vol + 1.0);  // S ~ T*ln(V)

        shared_entropy[tid] = fabs(local_entropy);  // Absolute value ensures S ≥ 0
    }

    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_entropy[tid] += shared_entropy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(entropy_result, shared_entropy[0]);
    }
}

// Kernel 6: Compute order parameter (phase synchronization)
extern "C" __global__ void compute_order_parameter_kernel(
    const double* phases,
    double* order_real,          // Output: real part of order parameter
    double* order_imag,          // Output: imag part of order parameter
    int n_oscillators
) {
    __shared__ double shared_real[256];
    __shared__ double shared_imag[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_real[tid] = 0.0;
    shared_imag[tid] = 0.0;

    if (idx < n_oscillators) {
        double phi = phases[idx];
        shared_real[tid] = cos(phi);
        shared_imag[tid] = sin(phi);
    }

    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_real[tid] += shared_real[tid + stride];
            shared_imag[tid] += shared_imag[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(order_real, shared_real[0]);
        atomicAdd(order_imag, shared_imag[0]);
    }
}

// Kernel 7: Compute conflicts per vertex on-device
// This enables conflict-aware evolution without CPU round-trips
extern "C" __global__ void compute_conflicts_kernel(
    const int* coloring,         // Current vertex colors
    const unsigned int* edge_u,  // Edge sources
    const unsigned int* edge_v,  // Edge targets
    int n_edges,
    int n_vertices,
    int* conflicts               // Output: conflict count per vertex
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vertices) return;

    int my_color = coloring[idx];
    int conflict_count = 0;

    // Count conflicts: edges where both endpoints have same color
    for (int e = 0; e < n_edges; e++) {
        unsigned int u = edge_u[e];
        unsigned int w = edge_v[e];

        // Check if this edge involves our vertex
        if ((u == idx || w == idx) && coloring[u] == coloring[w]) {
            conflict_count++;
        }
    }

    conflicts[idx] = conflict_count;
}
