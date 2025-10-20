#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMul(float *a, float *b, float *c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║   RTX 5070 + CUDA 13 Performance Benchmark  ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");
    
    // Vector addition benchmark
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    cudaEventRecord(start);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("🚀 Vector Addition (10M elements):\n");
    printf("   Time: %.3f ms\n", milliseconds);
    printf("   Throughput: %.2f GB/s\n", (3 * N * sizeof(float)) / (milliseconds / 1000.0) / 1e9);
    
    // Matrix multiplication benchmark
    int matrixSize = 1024;
    size_t matrixBytes = matrixSize * matrixSize * sizeof(float);
    
    float *h_ma = (float*)malloc(matrixBytes);
    float *h_mb = (float*)malloc(matrixBytes);
    float *h_mc = (float*)malloc(matrixBytes);
    
    float *d_ma, *d_mb, *d_mc;
    cudaMalloc(&d_ma, matrixBytes);
    cudaMalloc(&d_mb, matrixBytes);
    cudaMalloc(&d_mc, matrixBytes);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((matrixSize + blockSize.x - 1) / blockSize.x,
                  (matrixSize + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(start);
    matrixMul<<<gridSize, blockSize>>>(d_ma, d_mb, d_mc, matrixSize);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\n🔥 Matrix Multiplication (1024x1024):\n");
    printf("   Time: %.3f ms\n", milliseconds);
    float gflops = (2.0 * matrixSize * matrixSize * matrixSize) / (milliseconds / 1000.0) / 1e9;
    printf("   Performance: %.2f GFLOPS\n", gflops);
    
    printf("\n✅ Benchmark complete! RTX 5070 ready for AI workloads.\n");
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_ma); cudaFree(d_mb); cudaFree(d_mc);
    free(h_a); free(h_b); free(h_c);
    free(h_ma); free(h_mb); free(h_mc);
    
    return 0;
}
