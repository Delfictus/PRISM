// GPU kernels for KSG Transfer Entropy estimation
// Constitution: Phase 6 Implementation Constitution - Sprint 1.2

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

// Compute pairwise distances in joint space
// Each thread handles one query point
__global__ void compute_distances_kernel(
    const float* y_current,     // [n_points]
    const float* y_past,        // [n_points * embed_dim]
    const float* x_past,        // [n_points * embed_dim]
    float* distances,           // [n_points * n_points] output
    int n_points,
    int embed_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    // Compute distance from point i to all other points
    for (int j = 0; j < n_points; j++) {
        if (i == j) {
            distances[i * n_points + j] = INFINITY;
            continue;
        }

        // Max norm distance (L-infinity)
        float max_dist = fabsf(y_current[i] - y_current[j]);

        // Distance in y_past
        for (int d = 0; d < embed_dim; d++) {
            int idx_i = i * embed_dim + d;
            int idx_j = j * embed_dim + d;
            float dist_y = fabsf(y_past[idx_i] - y_past[idx_j]);
            max_dist = fmaxf(max_dist, dist_y);
        }

        // Distance in x_past
        for (int d = 0; d < embed_dim; d++) {
            int idx_i = i * embed_dim + d;
            int idx_j = j * embed_dim + d;
            float dist_x = fabsf(x_past[idx_i] - x_past[idx_j]);
            max_dist = fmaxf(max_dist, dist_x);
        }

        distances[i * n_points + j] = max_dist;
    }
}

// Find k-th smallest distance for each point
// Uses parallel selection algorithm
__global__ void find_kth_distance_kernel(
    const float* distances,     // [n_points * n_points]
    float* epsilon_values,      // [n_points] output - k-th distance for each point
    int n_points,
    int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    // Extract distances for point i
    const float* dist_row = distances + i * n_points;

    // Use partial sort to find k-th element
    // For small k, simple selection is efficient
    float kth_dist = INFINITY;

    if (k <= 0 || k >= n_points) {
        epsilon_values[i] = 0.0f;
        return;
    }

    // Find k smallest distances
    for (int iteration = 0; iteration < k; iteration++) {
        float min_val = INFINITY;

        for (int j = 0; j < n_points; j++) {
            if (dist_row[j] < min_val && dist_row[j] > kth_dist) {
                min_val = dist_row[j];
            }
        }

        kth_dist = min_val;
    }

    epsilon_values[i] = kth_dist;
}

// Count neighbors in marginal space Y (y_current, y_past)
__global__ void count_neighbors_y_kernel(
    const float* y_current,     // [n_points]
    const float* y_past,        // [n_points * embed_dim]
    const float* epsilon_values,// [n_points] - distance threshold per point
    int* neighbor_counts_y,     // [n_points] output
    int n_points,
    int embed_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    float epsilon = epsilon_values[i];
    int count = 0;

    for (int j = 0; j < n_points; j++) {
        if (i == j) continue;

        // Max norm in Y space
        float max_dist = fabsf(y_current[i] - y_current[j]);

        for (int d = 0; d < embed_dim; d++) {
            int idx_i = i * embed_dim + d;
            int idx_j = j * embed_dim + d;
            float dist = fabsf(y_past[idx_i] - y_past[idx_j]);
            max_dist = fmaxf(max_dist, dist);
        }

        if (max_dist < epsilon) {
            count++;
        }
    }

    neighbor_counts_y[i] = count;
}

// Count neighbors in joint XZ space (x_past, y_past)
__global__ void count_neighbors_xz_kernel(
    const float* x_past,        // [n_points * embed_dim]
    const float* y_past,        // [n_points * embed_dim]
    const float* epsilon_values,// [n_points]
    int* neighbor_counts_xz,    // [n_points] output
    int n_points,
    int embed_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    float epsilon = epsilon_values[i];
    int count = 0;

    for (int j = 0; j < n_points; j++) {
        if (i == j) continue;

        float max_dist = 0.0f;

        // Distance in x_past and y_past
        for (int d = 0; d < embed_dim; d++) {
            int idx_i = i * embed_dim + d;
            int idx_j = j * embed_dim + d;

            float dist_x = fabsf(x_past[idx_i] - x_past[idx_j]);
            float dist_y = fabsf(y_past[idx_i] - y_past[idx_j]);

            max_dist = fmaxf(max_dist, dist_x);
            max_dist = fmaxf(max_dist, dist_y);
        }

        if (max_dist < epsilon) {
            count++;
        }
    }

    neighbor_counts_xz[i] = count;
}

// Count neighbors in Z space (y_past only)
__global__ void count_neighbors_z_kernel(
    const float* y_past,        // [n_points * embed_dim]
    const float* epsilon_values,// [n_points]
    int* neighbor_counts_z,     // [n_points] output
    int n_points,
    int embed_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    float epsilon = epsilon_values[i];
    int count = 0;

    for (int j = 0; j < n_points; j++) {
        if (i == j) continue;

        float max_dist = 0.0f;

        for (int d = 0; d < embed_dim; d++) {
            int idx_i = i * embed_dim + d;
            int idx_j = j * embed_dim + d;
            float dist = fabsf(y_past[idx_i] - y_past[idx_j]);
            max_dist = fmaxf(max_dist, dist);
        }

        if (max_dist < epsilon) {
            count++;
        }
    }

    neighbor_counts_z[i] = count;
}

// Compute digamma function on GPU
__device__ float digamma_device(float x) {
    if (x <= 0.0f) return -INFINITY;

    // Asymptotic expansion for large x
    if (x > 10.0f) {
        return logf(x) - 0.5f / x - 1.0f / (12.0f * x * x) + 1.0f / (120.0f * x * x * x * x);
    }

    // Euler-Mascheroni constant
    const float EULER_GAMMA = 0.5772156649f;
    float result = -EULER_GAMMA;
    float val = x;

    // Recurrence relation
    while (val < 10.0f) {
        result -= 1.0f / val;
        val += 1.0f;
    }

    return result + logf(val) - 0.5f / val;
}

// Compute final TE values from neighbor counts
__global__ void compute_te_kernel(
    const int* neighbor_counts_y,   // [n_points]
    const int* neighbor_counts_xz,  // [n_points]
    const int* neighbor_counts_z,   // [n_points]
    float* te_contributions,        // [n_points] output
    int n_points,
    int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    int n_y = neighbor_counts_y[i];
    int n_xz = neighbor_counts_xz[i];
    int n_z = neighbor_counts_z[i];

    // KSG formula: ψ(k) - ψ(n_y+1) - ψ(n_xz+1) + ψ(n_z+1)
    float te = digamma_device((float)k)
             - digamma_device((float)(n_y + 1))
             - digamma_device((float)(n_xz + 1))
             + digamma_device((float)(n_z + 1));

    te_contributions[i] = te;
}

// Reduction kernel to sum TE contributions
__global__ void reduce_sum_kernel(
    const float* values,
    float* output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? values[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

} // extern "C"