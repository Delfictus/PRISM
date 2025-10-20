// Quantum MLIR GPU Kernels - Native Complex Number Support
//
// This provides the actual GPU kernels for quantum state evolution
// with first-class complex number support using cuComplex

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdio.h>

// Complex number operations using CUDA's native complex support
__device__ __forceinline__ cuDoubleComplex complex_mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
}

__device__ __forceinline__ cuDoubleComplex complex_add(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
}

__device__ __forceinline__ cuDoubleComplex complex_exp(double theta) {
    return make_cuDoubleComplex(cos(theta), sin(theta));
}

// Quantum Hadamard gate kernel
extern "C" __global__ void hadamard_gate_kernel(
    cuDoubleComplex* state,
    int qubit_index,
    int num_qubits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_dim = 1 << num_qubits;

    if (idx >= state_dim / 2) return;

    // Calculate indices for the two states that differ in qubit_index
    int mask = 1 << qubit_index;
    int idx0 = ((idx >> qubit_index) << (qubit_index + 1)) | (idx & ((1 << qubit_index) - 1));
    int idx1 = idx0 | mask;

    // Apply Hadamard transformation
    cuDoubleComplex amp0 = state[idx0];
    cuDoubleComplex amp1 = state[idx1];

    double sqrt2_inv = 0.7071067811865475; // 1/sqrt(2)
    cuDoubleComplex factor = make_cuDoubleComplex(sqrt2_inv, 0.0);

    state[idx0] = cuCmul(factor, cuCadd(amp0, amp1));
    state[idx1] = cuCmul(factor, cuCsub(amp0, amp1));
}

// CNOT gate kernel
extern "C" __global__ void cnot_gate_kernel(
    cuDoubleComplex* state,
    int control_qubit,
    int target_qubit,
    int num_qubits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_dim = 1 << num_qubits;

    if (idx >= state_dim) return;

    int control_mask = 1 << control_qubit;
    int target_mask = 1 << target_qubit;

    // Only flip target if control is |1>
    if ((idx & control_mask) != 0) {
        int flipped_idx = idx ^ target_mask;
        if (idx < flipped_idx) {
            // Swap amplitudes
            cuDoubleComplex temp = state[idx];
            state[idx] = state[flipped_idx];
            state[flipped_idx] = temp;
        }
    }
}

// Time evolution kernel using Trotter-Suzuki decomposition
extern "C" __global__ void time_evolution_kernel(
    cuDoubleComplex* state,
    const cuDoubleComplex* hamiltonian,
    double time_step,
    int dimension,
    int trotter_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dimension) return;

    // Local copy of state amplitude
    cuDoubleComplex local_state = state[idx];

    // Trotter-Suzuki approximation: e^(-iHt) ≈ (e^(-iHt/n))^n
    double dt = time_step / trotter_steps;

    for (int step = 0; step < trotter_steps; step++) {
        cuDoubleComplex new_amplitude = make_cuDoubleComplex(0.0, 0.0);

        // Matrix-vector multiplication: |ψ'> = e^(-iHdt)|ψ>
        for (int j = 0; j < dimension; j++) {
            cuDoubleComplex H_ij = hamiltonian[idx * dimension + j];
            cuDoubleComplex phase = complex_exp(-dt * cuCreal(H_ij));
            cuDoubleComplex contribution = cuCmul(phase, cuCmul(H_ij, state[j]));
            new_amplitude = cuCadd(new_amplitude, contribution);
        }

        local_state = new_amplitude;
    }

    // Write back evolved state
    state[idx] = local_state;
}

// Quantum Fourier Transform kernel
extern "C" __global__ void qft_kernel(
    cuDoubleComplex* state,
    int num_qubits,
    bool inverse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = 1 << num_qubits;

    if (idx >= N) return;

    extern __shared__ cuDoubleComplex shared_state[];

    // Load state into shared memory
    shared_state[threadIdx.x] = state[idx];
    __syncthreads();

    // QFT is essentially a DFT on quantum amplitudes
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    double sign = inverse ? 1.0 : -1.0;

    for (int k = 0; k < N; k++) {
        double phase = sign * 2.0 * M_PI * idx * k / N;
        cuDoubleComplex twiddle = complex_exp(phase);
        result = cuCadd(result, cuCmul(twiddle, shared_state[k]));
    }

    // Normalize
    double norm = 1.0 / sqrt((double)N);
    state[idx] = cuCmul(make_cuDoubleComplex(norm, 0.0), result);
}

// VQE ansatz kernel for variational quantum algorithms
extern "C" __global__ void vqe_ansatz_kernel(
    cuDoubleComplex* state,
    const double* parameters,
    int num_qubits,
    int num_layers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_dim = 1 << num_qubits;

    if (idx >= state_dim) return;

    // Apply parameterized quantum circuit
    for (int layer = 0; layer < num_layers; layer++) {
        // Rotation layer
        for (int qubit = 0; qubit < num_qubits; qubit++) {
            int param_idx = layer * num_qubits + qubit;
            double theta = parameters[param_idx];

            int mask = 1 << qubit;
            if ((idx & mask) != 0) {
                // Apply Rz rotation
                cuDoubleComplex phase = complex_exp(theta / 2.0);
                state[idx] = cuCmul(state[idx], phase);
            } else {
                // Apply Rz rotation (opposite phase)
                cuDoubleComplex phase = complex_exp(-theta / 2.0);
                state[idx] = cuCmul(state[idx], phase);
            }
        }

        // Entangling layer (linear connectivity)
        __syncthreads();
        for (int qubit = 0; qubit < num_qubits - 1; qubit++) {
            // CNOT between qubit and qubit+1
            int control_mask = 1 << qubit;
            int target_mask = 1 << (qubit + 1);

            if ((idx & control_mask) != 0 && (idx & target_mask) == 0) {
                int flipped_idx = idx | target_mask;
                cuDoubleComplex temp = state[idx];
                state[idx] = state[flipped_idx];
                state[flipped_idx] = temp;
            }
        }
        __syncthreads();
    }
}

// Measurement kernel - calculates probabilities
extern "C" __global__ void measurement_kernel(
    const cuDoubleComplex* state,
    double* probabilities,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dimension) return;

    cuDoubleComplex amp = state[idx];
    probabilities[idx] = cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
}

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            return error; \
        } \
    } while(0)

// C interface for calling from Rust
extern "C" {
    // Initialize quantum state to |00...0>
    cudaError_t quantum_init_state(cuDoubleComplex* state, int dimension) {
        CHECK_CUDA_ERROR(cudaMemset(state, 0, dimension * sizeof(cuDoubleComplex)));
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        CHECK_CUDA_ERROR(cudaMemcpy(state, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        return cudaSuccess;
    }

    // Apply Hadamard gate
    cudaError_t quantum_hadamard(cuDoubleComplex* state, int qubit, int num_qubits) {
        int state_dim = 1 << num_qubits;
        int num_blocks = (state_dim / 2 + 255) / 256;
        hadamard_gate_kernel<<<num_blocks, 256>>>(state, qubit, num_qubits);
        return cudaGetLastError();
    }

    // Apply CNOT gate
    cudaError_t quantum_cnot(cuDoubleComplex* state, int control, int target, int num_qubits) {
        int state_dim = 1 << num_qubits;
        int num_blocks = (state_dim + 255) / 256;
        cnot_gate_kernel<<<num_blocks, 256>>>(state, control, target, num_qubits);
        return cudaGetLastError();
    }

    // Time evolution
    cudaError_t quantum_evolve(
        cuDoubleComplex* state,
        const cuDoubleComplex* hamiltonian,
        double time,
        int dimension,
        int trotter_steps
    ) {
        int num_blocks = (dimension + 255) / 256;
        time_evolution_kernel<<<num_blocks, 256>>>(state, hamiltonian, time, dimension, trotter_steps);
        return cudaGetLastError();
    }

    // Quantum Fourier Transform
    cudaError_t quantum_qft(cuDoubleComplex* state, int num_qubits, bool inverse) {
        int state_dim = 1 << num_qubits;
        int threads = min(256, state_dim);
        int blocks = (state_dim + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(cuDoubleComplex);
        qft_kernel<<<blocks, threads, shared_mem>>>(state, num_qubits, inverse);
        return cudaGetLastError();
    }

    // VQE ansatz
    cudaError_t quantum_vqe_ansatz(
        cuDoubleComplex* state,
        const double* parameters,
        int num_qubits,
        int num_layers
    ) {
        int state_dim = 1 << num_qubits;
        int num_blocks = (state_dim + 255) / 256;
        vqe_ansatz_kernel<<<num_blocks, 256>>>(state, parameters, num_qubits, num_layers);
        return cudaGetLastError();
    }

    // Measure quantum state
    cudaError_t quantum_measure(
        const cuDoubleComplex* state,
        double* probabilities,
        int dimension
    ) {
        int num_blocks = (dimension + 255) / 256;
        measurement_kernel<<<num_blocks, 256>>>(state, probabilities, dimension);
        return cudaGetLastError();
    }
}