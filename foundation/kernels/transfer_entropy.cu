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

//==============================================================================
// BATCHED TRANSFER ENTROPY KERNELS
// Process ALL vertex pairs in parallel with minimal kernel launches
//==============================================================================

// Kernel: Batched global min/max computation across ALL time series
// Computes min/max for all n vertices in parallel
extern "C" __global__ void compute_global_minmax_batched_kernel(
    const double* all_time_series,  // [n_vertices x time_steps]
    int n_vertices,
    int time_steps,
    double* min_vals,               // Output: [n_vertices]
    double* max_vals                // Output: [n_vertices]
) {
    int vertex_id = blockIdx.x;
    if (vertex_id >= n_vertices) return;

    int tid = threadIdx.x;
    int ts_offset = vertex_id * time_steps;

    __shared__ double shared_min[256];
    __shared__ double shared_max[256];

    // Initialize with first values
    shared_min[tid] = 1e308;
    shared_max[tid] = -1e308;

    // Each thread processes a subset of time steps
    for (int t = tid; t < time_steps; t += blockDim.x) {
        double val = all_time_series[ts_offset + t];
        shared_min[tid] = fmin(shared_min[tid], val);
        shared_max[tid] = fmax(shared_max[tid], val);
    }
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fmin(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_vals[vertex_id] = shared_min[0];
        max_vals[vertex_id] = shared_max[0];
    }
}

// Kernel: Batched TE computation - each thread block processes ONE vertex pair
// Grid: (n_vertices, n_vertices) - one block per pair
// Much faster than sequential O(n²) processing!
extern "C" __global__ void compute_te_matrix_batched_kernel(
    const double* all_time_series,  // [n_vertices x time_steps]
    const double* min_vals,         // [n_vertices]
    const double* max_vals,         // [n_vertices]
    int n_vertices,
    int time_steps,
    int embedding_dim,
    int tau,
    int n_bins,
    double* te_matrix               // Output: [n_vertices x n_vertices]
) {
    // CRITICAL FIX: Correct mapping to match row-major output
    // Grid launch: grid_dim = (n, n, 1) where x=target, y=source
    int target_id = blockIdx.x;     // Target vertex (Y) - column index
    int source_id = blockIdx.y;     // Source vertex (X) - row index

    // Bounds checking
    if (target_id >= n_vertices || source_id >= n_vertices) return;

    // Compute output index: row-major layout (source * n + target)
    int output_idx = source_id * n_vertices + target_id;

    // Safety check: ensure output index is within bounds
    if (output_idx >= n_vertices * n_vertices) return;

    // Self-loops have zero TE
    if (source_id == target_id) {
        te_matrix[output_idx] = 0.0;
        return;
    }

    int tid = threadIdx.x;

    // Allocate shared histograms (use small n_bins=8 for memory efficiency)
    // 8^3 = 512 bins for 3D, 8^2 = 64 for 2D, 8 for 1D
    __shared__ int hist_3d[512];      // P(Y_future, X_past, Y_past)
    __shared__ int hist_2d_yf_yp[64]; // P(Y_future, Y_past)
    __shared__ int hist_2d_xp_yp[64]; // P(X_past, Y_past)
    __shared__ int hist_1d_yp[8];     // P(Y_past)

    // Zero-initialize histograms
    for (int i = tid; i < 512; i += blockDim.x) hist_3d[i] = 0;
    for (int i = tid; i < 64; i += blockDim.x) hist_2d_yf_yp[i] = 0;
    for (int i = tid; i < 64; i += blockDim.x) hist_2d_xp_yp[i] = 0;
    for (int i = tid; i < 8; i += blockDim.x) hist_1d_yp[i] = 0;
    __syncthreads();

    // Get time series pointers
    const double* source_ts = all_time_series + source_id * time_steps;
    const double* target_ts = all_time_series + target_id * time_steps;

    double source_min = min_vals[source_id];
    double source_max = max_vals[source_id];
    double target_min = min_vals[target_id];
    double target_max = max_vals[target_id];

    // Compute valid time range
    int min_time = embedding_dim * tau;
    int max_time = time_steps - 1;
    int valid_length = max_time - min_time;

    // Build histograms in parallel across threads
    for (int idx = tid; idx < valid_length; idx += blockDim.x) {
        int t = min_time + idx;

        // Y_future
        double y_future = target_ts[t];
        int bin_yf = compute_bin(y_future, target_min, target_max, n_bins);

        // X_past
        double x_past = (t >= tau) ? source_ts[t - tau] : source_ts[0];
        int bin_xp = compute_bin(x_past, source_min, source_max, n_bins);

        // Y_past
        double y_past = (t >= tau) ? target_ts[t - tau] : target_ts[0];
        int bin_yp = compute_bin(y_past, target_min, target_max, n_bins);

        // Update histograms atomically
        atomicAdd(&hist_3d[bin_yf * n_bins * n_bins + bin_xp * n_bins + bin_yp], 1);
        atomicAdd(&hist_2d_yf_yp[bin_yf * n_bins + bin_yp], 1);
        atomicAdd(&hist_2d_xp_yp[bin_xp * n_bins + bin_yp], 1);
        atomicAdd(&hist_1d_yp[bin_yp], 1);
    }
    __syncthreads();

    // Compute transfer entropy from histograms (parallel reduction)
    __shared__ double shared_te[256];
    shared_te[tid] = 0.0;

    int total_bins_3d = n_bins * n_bins * n_bins;
    for (int idx = tid; idx < total_bins_3d; idx += blockDim.x) {
        int bin_yf = idx / (n_bins * n_bins);
        int bin_xp = (idx / n_bins) % n_bins;
        int bin_yp = idx % n_bins;

        int count_3d = hist_3d[idx];
        if (count_3d == 0) continue;

        int count_2d_yf_yp = hist_2d_yf_yp[bin_yf * n_bins + bin_yp];
        int count_2d_xp_yp = hist_2d_xp_yp[bin_xp * n_bins + bin_yp];
        int count_1d_yp = hist_1d_yp[bin_yp];

        if (count_2d_yf_yp == 0 || count_2d_xp_yp == 0 || count_1d_yp == 0) continue;

        double p_3d = (double)count_3d / valid_length;
        double p_2d_yf_yp = (double)count_2d_yf_yp / valid_length;
        double p_2d_xp_yp = (double)count_2d_xp_yp / valid_length;
        double p_1d_yp = (double)count_1d_yp / valid_length;

        // TE formula: sum p(yf,xp,yp) * log[ p(yf,xp,yp)*p(yp) / (p(yf,yp)*p(xp,yp)) ]
        double numerator = p_3d * p_1d_yp;
        double denominator = p_2d_yf_yp * p_2d_xp_yp;

        if (denominator > 0.0) {
            shared_te[tid] += p_3d * log(numerator / denominator);
        }
    }
    __syncthreads();

    // Final reduction to get total TE
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_te[tid] += shared_te[tid + stride];
        }
        __syncthreads();
    }

    // Write result (using pre-computed output_idx for safety)
    if (tid == 0) {
        te_matrix[output_idx] = shared_te[0];
    }
}
