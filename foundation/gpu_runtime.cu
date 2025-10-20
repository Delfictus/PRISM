// GPU Runtime Wrapper - Actually launches kernels
// Compile with: nvcc -shared -o libgpu_runtime.so gpu_runtime.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

extern "C" {

// Simple Transfer Entropy kernel that actually runs
__global__ void compute_te_gpu(double* source, double* target, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Simplified TE calculation for demonstration
    double local_te = 0.0;

    // Compute local transfer entropy
    if (idx > 0 && idx < n-1) {
        double s_prev = source[idx-1];
        double s_curr = source[idx];
        double t_curr = target[idx];
        double t_next = target[idx+1];

        // Simplified TE formula
        double joint_prob = fabs(s_curr - t_next) + 0.001;
        double conditional_prob = fabs(t_curr - t_next) + 0.001;

        local_te = log2(conditional_prob / joint_prob);
    }

    // Sum reduction (simplified)
    atomicAdd(result, local_te / n);
}

// Thermodynamic evolution kernel that actually runs
__global__ void evolve_thermo_gpu(double* phases, double* velocities, int n_osc, int n_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_osc) return;

    double dt = 0.001;
    double damping = 0.1;
    double coupling = 0.5;

    for (int step = 0; step < n_steps; step++) {
        // Load current state
        double phase = phases[idx];
        double vel = velocities[idx];

        // Compute forces (simplified Kuramoto)
        double force = 1.0; // Natural frequency

        // Add coupling forces from neighbors
        if (idx > 0) {
            double phase_diff = phases[idx-1] - phase;
            force += coupling * sin(phase_diff);
        }
        if (idx < n_osc-1) {
            double phase_diff = phases[idx+1] - phase;
            force += coupling * sin(phase_diff);
        }

        // Apply damping
        force -= damping * vel;

        // Update velocity and phase
        vel += force * dt;
        phase += vel * dt;

        // Keep phase in [0, 2π]
        while (phase > 2*M_PI) phase -= 2*M_PI;
        while (phase < 0) phase += 2*M_PI;

        // Store back
        phases[idx] = phase;
        velocities[idx] = vel;
    }
}

// C interface functions that CPU can call
float launch_transfer_entropy(double* source, double* target, int n) {
    double *d_source, *d_target, *d_result;
    double h_result = 0.0;

    // Allocate GPU memory
    cudaMalloc(&d_source, n * sizeof(double));
    cudaMalloc(&d_target, n * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    // Copy to GPU
    cudaMemcpy(d_source, source, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (n + 255) / 256;
    compute_te_gpu<<<blocks, 256>>>(d_source, d_target, d_result, n);

    // Get result
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_source);
    cudaFree(d_target);
    cudaFree(d_result);

    return (float)h_result;
}

void launch_thermodynamic(double* phases, double* velocities, int n_osc, int n_steps) {
    double *d_phases, *d_velocities;

    // Allocate GPU memory
    cudaMalloc(&d_phases, n_osc * sizeof(double));
    cudaMalloc(&d_velocities, n_osc * sizeof(double));

    // Copy to GPU
    cudaMemcpy(d_phases, phases, n_osc * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, n_osc * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (n_osc + 255) / 256;
    evolve_thermo_gpu<<<blocks, 256>>>(d_phases, d_velocities, n_osc, n_steps);

    // Copy back
    cudaMemcpy(phases, d_phases, n_osc * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_velocities, n_osc * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_phases);
    cudaFree(d_velocities);
}

// Test if GPU is available
int gpu_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // Reset any prior errors
    cudaGetLastError();

    // Check if we got a valid device count
    if (error != cudaSuccess || deviceCount <= 0 || deviceCount > 100) {
        // Try to reset and check again
        cudaDeviceReset();
        error = cudaGetDeviceCount(&deviceCount);
    }

    // Return 1 if we have at least one GPU
    return (error == cudaSuccess && deviceCount > 0 && deviceCount < 100) ? 1 : 0;
}

} // extern "C"