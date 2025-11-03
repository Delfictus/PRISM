// Transfer Entropy GPU Kernels
//
// Constitutional Compliance: Article VII - Kernel Compilation Standards
//
// Transfer Entropy: TE(X→Y) = I(Y_future; X_past | Y_past)
// Measures information flow from source X to target Y
//
// Implementation: Histogram-based mutual information estimation
// for GPU-accelerated time series analysis

#include <cuda_runtime.h>
#include <math.h>

// Device helper: 3D histogram binning
__device__ int compute_bin(double value, double min_val, double max_val, int n_bins) {
    if (value <= min_val) return 0;
    if (value >= max_val) return n_bins - 1;

    double normalized = (value - min_val) / (max_val - min_val);
    int bin = (int)(normalized * n_bins);
    return min(bin, n_bins - 1);
}

// Kernel 1: Compute min/max for normalization
extern "C" __global__ void compute_minmax_kernel(
    const double* data,
    int length,
    double* min_val,
    double* max_val
) {
    __shared__ double shared_min[256];
    __shared__ double shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    shared_min[tid] = (idx < length) ? data[idx] : 1e308;
    shared_max[tid] = (idx < length) ? data[idx] : -1e308;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fmin(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicMin((unsigned long long*)min_val, __double_as_longlong(shared_min[0]));
        atomicMax((unsigned long long*)max_val, __double_as_longlong(shared_max[0]));
    }
}

// Kernel 2: Build 3D histogram for joint probability P(Y_future, X_past, Y_past)
extern "C" __global__ void build_histogram_3d_kernel(
    const double* source,      // X time series
    const double* target,      // Y time series
    int length,                // Time series length
    int embedding_dim,         // Embedding dimension (k)
    int tau,                   // Time delay
    int n_bins,                // Number of histogram bins
    double source_min,
    double source_max,
    double target_min,
    double target_max,
    int* histogram             // Output: 3D histogram [n_bins x n_bins x n_bins]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute valid range for time series reconstruction
    int min_time = embedding_dim * tau;
    int max_time = length - 1;
    int valid_length = max_time - min_time;

    if (idx >= valid_length) return;

    int t = min_time + idx;

    // Y_future: target at time t
    double y_future = target[t];
    int bin_y_future = compute_bin(y_future, target_min, target_max, n_bins);

    // X_past: source embedded state at time t-1
    // Simplified: use single lag for performance
    double x_past = (t >= tau) ? source[t - tau] : source[0];
    int bin_x_past = compute_bin(x_past, source_min, source_max, n_bins);

    // Y_past: target embedded state at time t-1
    double y_past = (t >= tau) ? target[t - tau] : target[0];
    int bin_y_past = compute_bin(y_past, target_min, target_max, n_bins);

    // Update 3D histogram atomically
    int hist_idx = bin_y_future * n_bins * n_bins + bin_x_past * n_bins + bin_y_past;
    atomicAdd(&histogram[hist_idx], 1);
}

// Kernel 3: Build 2D histogram for marginal probability P(Y_future, Y_past)
extern "C" __global__ void build_histogram_2d_kernel(
    const double* target,      // Y time series
    int length,
    int embedding_dim,
    int tau,
    int n_bins,
    double target_min,
    double target_max,
    int* histogram             // Output: 2D histogram [n_bins x n_bins]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int min_time = embedding_dim * tau;
    int max_time = length - 1;
    int valid_length = max_time - min_time;

    if (idx >= valid_length) return;

    int t = min_time + idx;

    // Y_future
    double y_future = target[t];
    int bin_y_future = compute_bin(y_future, target_min, target_max, n_bins);

    // Y_past
    double y_past = (t >= tau) ? target[t - tau] : target[0];
    int bin_y_past = compute_bin(y_past, target_min, target_max, n_bins);

    // Update 2D histogram
    int hist_idx = bin_y_future * n_bins + bin_y_past;
    atomicAdd(&histogram[hist_idx], 1);
}

// Kernel 4: Compute transfer entropy from histograms
// TE = sum P(y_f, x_p, y_p) * log[ P(y_f, x_p, y_p) * P(y_p) / (P(y_f, y_p) * P(x_p, y_p)) ]
extern "C" __global__ void compute_transfer_entropy_kernel(
    const int* hist_3d,        // P(Y_future, X_past, Y_past)
    const int* hist_2d_yf_yp,  // P(Y_future, Y_past)
    const int* hist_2d_xp_yp,  // P(X_past, Y_past)
    const int* hist_1d_yp,     // P(Y_past)
    int n_bins,
    int total_samples,
    double* te_result          // Output: transfer entropy value
) {
    __shared__ double shared_te[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bins = n_bins * n_bins * n_bins;

    shared_te[tid] = 0.0;

    // Each thread processes multiple histogram bins
    for (int i = idx; i < total_bins; i += blockDim.x * gridDim.x) {
        int bin_y_future = i / (n_bins * n_bins);
        int bin_x_past = (i / n_bins) % n_bins;
        int bin_y_past = i % n_bins;

        int count_3d = hist_3d[i];
        if (count_3d == 0) continue;

        // Get marginal counts
        int idx_yf_yp = bin_y_future * n_bins + bin_y_past;
        int idx_xp_yp = bin_x_past * n_bins + bin_y_past;

        int count_yf_yp = hist_2d_yf_yp[idx_yf_yp];
        int count_xp_yp = hist_2d_xp_yp[idx_xp_yp];
        int count_yp = hist_1d_yp[bin_y_past];

        if (count_yf_yp == 0 || count_xp_yp == 0 || count_yp == 0) continue;

        // Compute probabilities
        double p_joint = (double)count_3d / total_samples;
        double p_yf_yp = (double)count_yf_yp / total_samples;
        double p_xp_yp = (double)count_xp_yp / total_samples;
        double p_yp = (double)count_yp / total_samples;

        // TE contribution: p(y_f, x_p, y_p) * log[ p(y_f, x_p, y_p) * p(y_p) / (p(y_f, y_p) * p(x_p, y_p)) ]
        double numerator = p_joint * p_yp;
        double denominator = p_yf_yp * p_xp_yp;

        if (denominator > 1e-10) {
            double log_ratio = log(numerator / denominator);
            shared_te[tid] += p_joint * log_ratio;
        }
    }

    __syncthreads();

    // Reduction to sum transfer entropy contributions
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_te[tid] += shared_te[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(te_result, shared_te[0]);
    }
}

// Kernel 5: Build 1D histogram for P(Y_past)
extern "C" __global__ void build_histogram_1d_kernel(
    const double* target,
    int length,
    int embedding_dim,
    int tau,
    int n_bins,
    double target_min,
    double target_max,
    int* histogram
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int min_time = embedding_dim * tau;
    int max_time = length - 1;
    int valid_length = max_time - min_time;

    if (idx >= valid_length) return;

    int t = min_time + idx;

    // Y_past
    double y_past = (t >= tau) ? target[t - tau] : target[0];
    int bin_y_past = compute_bin(y_past, target_min, target_max, n_bins);

    atomicAdd(&histogram[bin_y_past], 1);
}

// Kernel 6: Build 2D histogram for P(X_past, Y_past)
extern "C" __global__ void build_histogram_2d_xp_yp_kernel(
    const double* source,
    const double* target,
    int length,
    int embedding_dim,
    int tau,
    int n_bins,
    double source_min,
    double source_max,
    double target_min,
    double target_max,
    int* histogram
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int min_time = embedding_dim * tau;
    int max_time = length - 1;
    int valid_length = max_time - min_time;

    if (idx >= valid_length) return;

    int t = min_time + idx;

    // X_past
    double x_past = (t >= tau) ? source[t - tau] : source[0];
    int bin_x_past = compute_bin(x_past, source_min, source_max, n_bins);

    // Y_past
    double y_past = (t >= tau) ? target[t - tau] : target[0];
    int bin_y_past = compute_bin(y_past, target_min, target_max, n_bins);

    int hist_idx = bin_x_past * n_bins + bin_y_past;
    atomicAdd(&histogram[hist_idx], 1);
}
