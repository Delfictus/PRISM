// Active Inference GPU Kernels
//
// Constitutional Compliance: Article VII - Kernel Compilation Standards
//
// Implements variational inference operations for Active Inference:
// - Free Energy minimization: F = Complexity - Accuracy
// - Belief updates: dμ/dt = κ·(ε_sensory + ε_dynamical)
// - Prediction errors: ε = Π·(o - g(μ))
//
// Key operations:
// - Matrix-vector multiplication (GEMV)
// - Element-wise operations (precision weighting)
// - Belief propagation (hierarchical updates)

#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: General Matrix-Vector multiplication (GEMV)
// y = alpha * A * x + beta * y  (or A^T * x if transpose)
extern "C" __global__ void gemv_kernel(
    double* y,              // Output vector [m]
    const double* A,        // Matrix [m x n] (row-major)
    const double* x,        // Input vector [n]
    int m,                  // Rows
    int n,                  // Columns
    double alpha,           // Scalar multiplier
    double beta,            // Output scaling
    int transpose           // 0 = A*x, 1 = A^T*x
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (transpose) {
        // Compute (A^T * x)[i]
        if (i >= n) return;

        double sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += A[j * n + i] * x[j];  // A[j,i] in row-major
        }

        y[i] = alpha * sum + beta * y[i];
    } else {
        // Compute (A * x)[i]
        if (i >= m) return;

        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];  // A[i,j] in row-major
        }

        y[i] = alpha * sum + beta * y[i];
    }
}

// Kernel 2: Prediction error (element-wise)
// error = precision * (observation - prediction)
extern "C" __global__ void prediction_error_kernel(
    double* error,           // Output: prediction error
    const double* observation,
    const double* prediction,
    const double* precision, // Precision (inverse variance)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double diff = observation[idx] - prediction[idx];
    error[idx] = precision[idx] * diff;
}

// Kernel 3: Belief update (natural gradient ascent)
// mean = mean + learning_rate * gradient
extern "C" __global__ void belief_update_kernel(
    double* mean,            // Belief mean (updated in-place)
    const double* gradient,  // Gradient direction
    double learning_rate,    // Step size κ
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    mean[idx] += learning_rate * gradient[idx];
}

// Kernel 4: Element-wise precision weighting
// output = precision * input
extern "C" __global__ void precision_weight_kernel(
    double* output,
    const double* input,
    const double* precision,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = precision[idx] * input[idx];
}

// Kernel 5: Compute KL divergence (complexity term)
// KL[q||p] = 0.5 * [tr(Σ_p^-1 Σ_q) + (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q) - k + ln(det(Σ_p)/det(Σ_q))]
// Simplified for diagonal covariance: KL = 0.5 * Σ_i [σ_q²/σ_p² + (μ_p - μ_q)²/σ_p² - 1 - ln(σ_q²/σ_p²)]
extern "C" __global__ void kl_divergence_kernel(
    const double* mean_q,      // Posterior mean
    const double* mean_p,      // Prior mean
    const double* var_q,       // Posterior variance
    const double* var_p,       // Prior variance
    double* kl_components,     // Output: KL contribution per dimension
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double mu_diff = mean_q[idx] - mean_p[idx];
    double sigma_q2 = var_q[idx];
    double sigma_p2 = var_p[idx];

    // Avoid division by zero
    if (sigma_p2 < 1e-10) sigma_p2 = 1e-10;
    if (sigma_q2 < 1e-10) sigma_q2 = 1e-10;

    // KL contribution for dimension i
    double ratio = sigma_q2 / sigma_p2;
    double mahalanobis = (mu_diff * mu_diff) / sigma_p2;

    kl_components[idx] = 0.5 * (ratio + mahalanobis - 1.0 - log(ratio));
}

// Kernel 6: Compute accuracy (log-likelihood)
// accuracy = -0.5 * [ε^T Π ε + ln(det(2π/Π))]
// For diagonal precision: -0.5 * [Σ_i (π_i * ε_i² + ln(2π/π_i))]
extern "C" __global__ void accuracy_kernel(
    const double* error,       // Prediction error
    const double* precision,   // Observation precision
    double* accuracy_components, // Output: accuracy per dimension
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double eps = error[idx];
    double pi = precision[idx];

    // Avoid log(0)
    if (pi < 1e-10) pi = 1e-10;

    // Accuracy contribution: -0.5 * (π*ε² + ln(2π/π))
    accuracy_components[idx] = -0.5 * (pi * eps * eps + log(2.0 * M_PI / pi));
}

// Kernel 7: Sum reduction (for aggregating KL/accuracy components)
extern "C" __global__ void sum_reduction_kernel(
    const double* input,
    double* output,
    int n
) {
    __shared__ double shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

    // Write block result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// Kernel 8: Element-wise add (for combining gradients)
// output = alpha * a + beta * b
extern "C" __global__ void axpby_kernel(
    double* output,
    const double* a,
    const double* b,
    double alpha,
    double beta,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = alpha * a[idx] + beta * b[idx];
}

// Kernel 9: Generalized coordinates update
// velocity = (mean - mean_prev) / dt
extern "C" __global__ void velocity_update_kernel(
    double* velocity,
    const double* mean_current,
    const double* mean_prev,
    double dt,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    velocity[idx] = (mean_current[idx] - mean_prev[idx]) / dt;
}

// Kernel 10: Hierarchical projection (atmosphere to windows)
// windows[i] = atmosphere[i % n_atmosphere]  (broadcast/tile)
extern "C" __global__ void hierarchical_project_kernel(
    double* windows,          // Output: [n_windows]
    const double* atmosphere, // Input: [n_atmosphere]
    int n_windows,
    int n_atmosphere
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_windows) return;

    // Tile atmospheric state across windows
    windows[idx] = atmosphere[idx % n_atmosphere];
}
