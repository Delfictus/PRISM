/**
 * CUDA Kernels for Matrix Operations
 * Optimized for RTX 5070 (Ada Lovelace architecture)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define TILE_SIZE 16

/**
 * Matrix multiplication kernel: C = A @ B
 * A: [M x K], B: [K x N], C: [M x N]
 */
extern "C" __global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tx < K) {
            tileA[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile from B
        if (col < N && t * TILE_SIZE + ty < K) {
            tileB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Bias addition kernel: output += bias (broadcasted)
 */
extern "C" __global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;

    if (idx < total) {
        int feature_idx = idx % features;
        output[idx] += bias[feature_idx];
    }
}

/**
 * ReLU activation kernel
 */
extern "C" __global__ void relu_kernel(
    float* __restrict__ data,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * Softmax kernel (for classification)
 * Operates on each row independently
 */
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int num_classes)
{
    int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size) return;

    int offset = batch_idx * num_classes;

    // Cooperative thread block for reduction
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Find max for numerical stability
    if (threadIdx.x == 0) {
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[offset + i]);
        }
        shared_max = max_val;
    }
    __syncthreads();

    float max_val = shared_max;

    // Compute exp and sum
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[offset + i] - max_val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }
        shared_sum = sum;
    }
    __syncthreads();

    float sum = shared_sum;

    // Normalize
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_classes; i++) {
            output[offset + i] /= sum;
        }
    }
}

/**
 * Cross-entropy loss gradient kernel
 * Computes gradient of loss with respect to logits
 */
extern "C" __global__ void cross_entropy_grad_kernel(
    const float* __restrict__ probs,
    const int* __restrict__ labels,
    float* __restrict__ grad,
    int batch_size,
    int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * num_classes) {
        int batch_idx = idx / num_classes;
        int class_idx = idx % num_classes;

        // Gradient is prob - 1 for correct class, prob otherwise
        if (class_idx == labels[batch_idx]) {
            grad[idx] = probs[idx] - 1.0f;
        } else {
            grad[idx] = probs[idx];
        }
    }
}

/**
 * Element-wise operations
 */
extern "C" __global__ void sigmoid_kernel(
    float* __restrict__ data,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

extern "C" __global__ void tanh_kernel(
    float* __restrict__ data,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

/**
 * Vector operations
 */
extern "C" __global__ void saxpy_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] += alpha * x[idx];
    }
}