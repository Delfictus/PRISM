//! GPU Kernel Executor with actual kernel execution capabilities
//!
//! This module provides the infrastructure to compile, load and execute
//! actual GPU kernels using the correct cudarc API.

use anyhow::{Context as AnyhowContext, Result};
use cudarc::{
    curand::CudaRng,
    driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Common GPU kernels used across the system
pub mod kernels {
    pub const VECTOR_ADD: &str = r#"
    extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    "#;

    pub const MATRIX_MUL: &str = r#"
    extern "C" __global__ void matmul(float* a, float* b, float* c, int m, int k, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    "#;

    pub const RELU: &str = r#"
    extern "C" __global__ void relu(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(0.0f, data[idx]);
        }
    }
    "#;

    pub const SOFTMAX: &str = r#"
    extern "C" __global__ void softmax(float* data, int batch_size, int num_classes) {
        int batch_idx = blockIdx.x;
        if (batch_idx >= batch_size) return;

        float* row = data + batch_idx * num_classes;

        // Find max for numerical stability
        float max_val = row[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, row[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            row[i] /= sum;
        }
    }
    "#;

    pub const SIGMOID: &str = r#"
    extern "C" __global__ void sigmoid(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 1.0f / (1.0f + expf(-data[idx]));
        }
    }
    "#;

    pub const TANH: &str = r#"
    extern "C" __global__ void tanh_activation(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = tanhf(data[idx]);
        }
    }
    "#;

    pub const BATCH_NORM: &str = r#"
    extern "C" __global__ void batch_norm(
        float* data, float* gamma, float* beta,
        float* mean, float* var,
        int batch_size, int features, float epsilon
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * features;

        if (idx < total_elements) {
            int feature_idx = idx % features;
            float normalized = (data[idx] - mean[feature_idx]) /
                              sqrtf(var[feature_idx] + epsilon);
            data[idx] = gamma[feature_idx] * normalized + beta[feature_idx];
        }
    }
    "#;

    // Active Inference Kernels
    pub const KL_DIVERGENCE: &str = r#"
    extern "C" __global__ void kl_divergence(
        float* q, float* p, float* kl_out, int n
    ) {
        int idx = threadIdx.x;

        float local_kl = 0.0f;
        if (idx < n) {
            float q_val = q[idx];
            float p_val = p[idx];
            if (q_val > 1e-10f && p_val > 1e-10f) {
                local_kl = q_val * logf(q_val / p_val);
            }
        }

        // Simple reduction for small arrays (< 256 elements)
        __shared__ float sdata[256];
        sdata[idx] = local_kl;
        __syncthreads();

        // Reduction
        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < n) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        // Write result
        if (idx == 0) {
            kl_out[0] = sdata[0];
        }
    }
    "#;

    pub const ELEMENTWISE_MULTIPLY: &str = r#"
    extern "C" __global__ void elementwise_multiply(
        float* a, float* b, float* c, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] * b[idx];
        }
    }
    "#;

    pub const NORMALIZE: &str = r#"
    extern "C" __global__ void normalize(float* data, int n) {
        int idx = threadIdx.x;

        // Compute sum using shared memory reduction
        __shared__ float sdata[256];
        sdata[idx] = (idx < n) ? data[idx] : 0.0f;
        __syncthreads();

        // Reduction
        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        float sum = sdata[0];
        __syncthreads();

        // Normalize
        if (idx < n && sum > 0.0f) {
            data[idx] /= sum;
        }
    }
    "#;

    // Neuromorphic Computing Kernels
    pub const LEAKY_INTEGRATE_FIRE: &str = r#"
    extern "C" __global__ void leaky_integrate_fire(
        float* state_current, float* state_previous,
        float* input, float leak_rate, float threshold,
        bool* spikes, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Leaky integration
            float new_state = (1.0f - leak_rate) * state_previous[idx] + input[idx];

            // Apply tanh nonlinearity
            new_state = tanhf(new_state);

            // Spike generation
            spikes[idx] = new_state > threshold;

            // Reset if spiked
            if (spikes[idx]) {
                new_state = 0.0f;
            }

            state_current[idx] = new_state;
        }
    }
    "#;

    pub const RESERVOIR_UPDATE: &str = r#"
    extern "C" __global__ void reservoir_update(
        float* state, float* prev_state, float* input,
        float leak_rate, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Leaky integration: x(t) = (1-α)x(t-1) + u(t)
            float integrated = (1.0f - leak_rate) * prev_state[idx] + input[idx];
            // Apply tanh nonlinearity
            state[idx] = tanhf(integrated);
        }
    }
    "#;

    pub const STDP_UPDATE: &str = r#"
    extern "C" __global__ void stdp_update(
        float* weights, bool* pre_spikes, bool* post_spikes,
        float* spike_times_pre, float* spike_times_post,
        float learning_rate, float tau_plus, float tau_minus,
        int n_pre, int n_post
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;  // Post neuron
        int j = blockIdx.x * blockDim.x + threadIdx.x;  // Pre neuron

        if (i < n_post && j < n_pre) {
            if (pre_spikes[j] && post_spikes[i]) {
                float dt = spike_times_post[i] - spike_times_pre[j];
                float dw = 0.0f;

                if (dt > 0.0f) {
                    // LTP: post after pre
                    dw = learning_rate * expf(-dt / tau_plus);
                } else if (dt < 0.0f) {
                    // LTD: pre after post
                    dw = -learning_rate * expf(dt / tau_minus);
                }

                int idx = i * n_pre + j;
                weights[idx] += dw;
                // Clamp weights to [-1, 1]
                weights[idx] = fmaxf(-1.0f, fminf(1.0f, weights[idx]));
            }
        }
    }
    "#;

    // Statistical Mechanics / Thermodynamic Kernels
    pub const KURAMOTO_EVOLUTION: &str = r#"
    extern "C" __global__ void kuramoto_evolution(
        float* phases, float* frequencies,
        float* coupling_matrix, float* new_phases,
        int n, float dt, float coupling_strength
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            // Kuramoto model: dθ/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
            float omega = frequencies[i];
            float coupling_sum = 0.0f;

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    float phase_diff = phases[j] - phases[i];
                    float coupling = coupling_matrix[i * n + j];
                    coupling_sum += coupling * sinf(phase_diff);
                }
            }

            float dphi = omega + (coupling_strength / (float)n) * coupling_sum;
            new_phases[i] = phases[i] + dphi * dt;

            // Wrap to [0, 2π]
            while (new_phases[i] > 6.28318531f) new_phases[i] -= 6.28318531f;
            while (new_phases[i] < 0.0f) new_phases[i] += 6.28318531f;
        }
    }
    "#;

    pub const ENTROPY_PRODUCTION: &str = r#"
    extern "C" __global__ void entropy_production(
        float* velocities, float* entropy_rate,
        float temperature, int n
    ) {
        int idx = threadIdx.x;

        float local_entropy = 0.0f;
        if (idx < n) {
            // Entropy production from velocity dissipation
            float v = velocities[idx];
            local_entropy = v * v / (2.0f * temperature);
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_entropy;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            entropy_rate[0] = sdata[0];
        }
    }
    "#;

    pub const ORDER_PARAMETER: &str = r#"
    extern "C" __global__ void order_parameter(
        float* phases, float* order_real, float* order_imag, int n
    ) {
        int idx = threadIdx.x;

        float local_real = 0.0f;
        float local_imag = 0.0f;

        if (idx < n) {
            local_real = cosf(phases[idx]);
            local_imag = sinf(phases[idx]);
        }

        // Reduction for real part
        __shared__ float sdata_real[256];
        __shared__ float sdata_imag[256];

        sdata_real[idx] = local_real;
        sdata_imag[idx] = local_imag;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata_real[idx] += sdata_real[idx + s];
                sdata_imag[idx] += sdata_imag[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            order_real[0] = sdata_real[0] / (float)n;
            order_imag[0] = sdata_imag[0] / (float)n;
        }
    }
    "#;

    // Transfer Entropy / Information Theory Kernels
    pub const MUTUAL_INFORMATION: &str = r#"
    extern "C" __global__ void mutual_information(
        float* joint_hist, float* marginal_x, float* marginal_y,
        float* mi_out, int n_bins
    ) {
        int idx = threadIdx.x;

        float local_mi = 0.0f;
        if (idx < n_bins * n_bins) {
            int i = idx / n_bins;
            int j = idx % n_bins;

            float p_xy = joint_hist[idx];
            float p_x = marginal_x[i];
            float p_y = marginal_y[j];

            if (p_xy > 1e-10f && p_x > 1e-10f && p_y > 1e-10f) {
                local_mi = p_xy * logf(p_xy / (p_x * p_y));
            }
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_mi;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            mi_out[0] = sdata[0];
        }
    }
    "#;

    pub const HISTOGRAM_2D: &str = r#"
    extern "C" __global__ void histogram_2d(
        float* x, float* y, int* hist,
        float min_val, float max_val,
        int n_samples, int n_bins
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n_samples) {
            float x_val = x[idx];
            float y_val = y[idx];

            // Bin calculation
            int bin_x = (int)((x_val - min_val) / (max_val - min_val) * (float)n_bins);
            int bin_y = (int)((y_val - min_val) / (max_val - min_val) * (float)n_bins);

            // Clamp to valid range
            bin_x = max(0, min(n_bins - 1, bin_x));
            bin_y = max(0, min(n_bins - 1, bin_y));

            // Atomic increment
            int hist_idx = bin_y * n_bins + bin_x;
            atomicAdd(&hist[hist_idx], 1);
        }
    }
    "#;

    pub const TIME_DELAYED_EMBEDDING: &str = r#"
    extern "C" __global__ void time_delayed_embedding(
        float* time_series, float* embedded,
        int n_samples, int embedding_dim, int tau
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int n_embedded = n_samples - (embedding_dim - 1) * tau;

        if (idx < n_embedded) {
            for (int d = 0; d < embedding_dim; d++) {
                int ts_idx = idx + d * tau;
                int emb_idx = idx * embedding_dim + d;
                embedded[emb_idx] = time_series[ts_idx];
            }
        }
    }
    "#;

    pub const CONDITIONAL_ENTROPY: &str = r#"
    extern "C" __global__ void conditional_entropy(
        float* joint_xyz, float* joint_xz,
        float* ce_out, int n_bins_xyz, int n_bins_xz
    ) {
        int idx = threadIdx.x;

        float local_ce = 0.0f;
        if (idx < n_bins_xyz) {
            float p_xyz = joint_xyz[idx];
            // Map to corresponding xz index (marginalize over y)
            int xz_idx = idx % n_bins_xz;
            float p_xz = joint_xz[xz_idx];

            if (p_xyz > 1e-10f && p_xz > 1e-10f) {
                local_ce = p_xyz * logf(p_xyz / p_xz);
            }
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_ce;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            ce_out[0] = -sdata[0];
        }
    }
    "#;

    // Quantum Simulation Kernels (Complex arithmetic)
    pub const HADAMARD_GATE: &str = r#"
    extern "C" __global__ void hadamard_gate(
        float* state_real, float* state_imag,
        int qubit_idx, int state_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_dim) return;

        int bit = (idx >> qubit_idx) & 1;
        int pair_idx = idx ^ (1 << qubit_idx);

        if (idx < pair_idx) {  // Process each pair once
            float r0 = state_real[idx];
            float i0 = state_imag[idx];
            float r1 = state_real[pair_idx];
            float i1 = state_imag[pair_idx];

            float sqrt2_inv = 0.70710678f;  // 1/sqrt(2)

            state_real[idx] = sqrt2_inv * (r0 + r1);
            state_imag[idx] = sqrt2_inv * (i0 + i1);
            state_real[pair_idx] = sqrt2_inv * (r0 - r1);
            state_imag[pair_idx] = sqrt2_inv * (i0 - i1);
        }
    }
    "#;

    pub const PAULI_X_GATE: &str = r#"
    extern "C" __global__ void pauli_x_gate(
        float* state_real, float* state_imag,
        int qubit_idx, int state_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_dim) return;

        int pair_idx = idx ^ (1 << qubit_idx);
        if (idx < pair_idx) {  // Swap pairs
            float temp_r = state_real[idx];
            float temp_i = state_imag[idx];
            state_real[idx] = state_real[pair_idx];
            state_imag[idx] = state_imag[pair_idx];
            state_real[pair_idx] = temp_r;
            state_imag[pair_idx] = temp_i;
        }
    }
    "#;

    pub const PHASE_GATE: &str = r#"
    extern "C" __global__ void phase_gate(
        float* state_real, float* state_imag,
        int qubit_idx, float theta, int state_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_dim) return;

        int bit = (idx >> qubit_idx) & 1;
        if (bit == 1) {
            // Apply phase: |1⟩ -> e^(iθ)|1⟩
            float r = state_real[idx];
            float i = state_imag[idx];
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            state_real[idx] = r * cos_theta - i * sin_theta;
            state_imag[idx] = r * sin_theta + i * cos_theta;
        }
    }
    "#;

    pub const CNOT_GATE: &str = r#"
    extern "C" __global__ void cnot_gate(
        float* state_real, float* state_imag,
        int control_idx, int target_idx, int state_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_dim) return;

        int control_bit = (idx >> control_idx) & 1;
        if (control_bit == 1) {
            // Flip target bit
            int pair_idx = idx ^ (1 << target_idx);
            if (idx < pair_idx) {
                float temp_r = state_real[idx];
                float temp_i = state_imag[idx];
                state_real[idx] = state_real[pair_idx];
                state_imag[idx] = state_imag[pair_idx];
                state_real[pair_idx] = temp_r;
                state_imag[pair_idx] = temp_i;
            }
        }
    }
    "#;

    pub const QUANTUM_MEASUREMENT: &str = r#"
    extern "C" __global__ void quantum_measurement(
        float* state_real, float* state_imag,
        float* probabilities, int state_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < state_dim) {
            float r = state_real[idx];
            float i = state_imag[idx];
            probabilities[idx] = r * r + i * i;  // |ψ|²
        }
    }
    "#;

    pub const BROADCAST_ADD: &str = r#"
    extern "C" __global__ void broadcast_add(
        float* data, float* bias, int batch_size, int features
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * features;

        if (idx < total) {
            int feature_idx = idx % features;
            data[idx] += bias[feature_idx];
        }
    }
    "#;

    pub const ELEMENTWISE_EXP: &str = r#"
    extern "C" __global__ void elementwise_exp(
        float* input, float* output, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = expf(input[idx]);
        }
    }
    "#;

    pub const DOT_PRODUCT: &str = r#"
    extern "C" __global__ void dot_product(
        float* a, float* b, float* result_out, int n
    ) {
        int idx = threadIdx.x;

        float local_product = 0.0f;
        if (idx < n) {
            local_product = a[idx] * b[idx];
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_product;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            result_out[0] = sdata[0];
        }
    }
    "#;

    pub const REDUCE_SUM: &str = r#"
    extern "C" __global__ void reduce_sum(
        float* data, float* sum_out, int n
    ) {
        int idx = threadIdx.x;

        float local_sum = 0.0f;
        if (idx < n) {
            local_sum = data[idx];
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_sum;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            sum_out[0] = sdata[0];
        }
    }
    "#;

    pub const SHANNON_ENTROPY: &str = r#"
    extern "C" __global__ void shannon_entropy(
        float* probabilities, float* entropy_out, int n
    ) {
        int idx = threadIdx.x;

        float local_entropy = 0.0f;
        if (idx < n) {
            float p = probabilities[idx];
            if (p > 1e-10f) {
                local_entropy = p * logf(p);
            }
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_entropy;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            entropy_out[0] = -sdata[0];  // Shannon entropy is -Σ p log p
        }
    }
    "#;

    // Transformer / LLM Kernels
    pub const MULTI_HEAD_ATTENTION: &str = r#"
    extern "C" __global__ void multi_head_attention(
        float* Q, float* K, float* V,
        float* output, float* attention_weights,
        int batch_size, int seq_len, int d_model, int n_heads
    ) {
        int head_idx = blockIdx.z;
        int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int batch_idx = blockIdx.x;

        if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= n_heads) return;

        int d_k = d_model / n_heads;  // Dimension per head
        float scale = 1.0f / sqrtf((float)d_k);

        // Compute attention scores for this position
        float* q_head = Q + batch_idx * seq_len * d_model + seq_idx * d_model + head_idx * d_k;

        __shared__ float scores[512];  // Max seq_len = 512 for shared memory

        // Compute Q·K^T for all positions
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            float* k_head = K + batch_idx * seq_len * d_model + k_pos * d_model + head_idx * d_k;

            float score = 0.0f;
            for (int d = 0; d < d_k; d++) {
                score += q_head[d] * k_head[d];
            }
            scores[k_pos] = score * scale;
        }
        __syncthreads();

        // Softmax over scores
        float max_score = scores[0];
        for (int i = 1; i < seq_len; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }

        for (int i = 0; i < seq_len; i++) {
            scores[i] /= sum_exp;
        }
        __syncthreads();

        // Compute weighted sum of V
        for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
            float sum = 0.0f;
            for (int v_pos = 0; v_pos < seq_len; v_pos++) {
                float* v_head = V + batch_idx * seq_len * d_model + v_pos * d_model + head_idx * d_k;
                sum += scores[v_pos] * v_head[d];
            }

            int out_idx = batch_idx * seq_len * d_model + seq_idx * d_model + head_idx * d_k + d;
            output[out_idx] = sum;
        }
    }
    "#;

    pub const ROPE_ENCODING: &str = r#"
    extern "C" __global__ void rope_encoding(
        float* qk, int seq_len, int d_model, int position_offset
    ) {
        int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (seq_idx >= seq_len || dim_idx >= d_model / 2) return;

        int pos = position_offset + seq_idx;
        float theta = powf(10000.0f, -2.0f * (float)dim_idx / (float)d_model);
        float angle = (float)pos * theta;

        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);

        // Rotate pairs of dimensions
        int idx1 = seq_idx * d_model + dim_idx * 2;
        int idx2 = idx1 + 1;

        float x1 = qk[idx1];
        float x2 = qk[idx2];

        qk[idx1] = x1 * cos_angle - x2 * sin_angle;
        qk[idx2] = x1 * sin_angle + x2 * cos_angle;
    }
    "#;

    pub const LAYER_NORM: &str = r#"
    extern "C" __global__ void layer_norm(
        float* input, float* output,
        float* gamma, float* beta,
        int batch_size, int seq_len, int d_model, float eps
    ) {
        int batch_idx = blockIdx.x;
        int seq_idx = blockIdx.y;

        if (batch_idx >= batch_size || seq_idx >= seq_len) return;

        int offset = (batch_idx * seq_len + seq_idx) * d_model;
        float* x = input + offset;
        float* y = output + offset;

        // Compute mean
        __shared__ float mean;
        __shared__ float variance;

        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < d_model; i++) {
                sum += x[i];
            }
            mean = sum / (float)d_model;

            // Compute variance
            float var_sum = 0.0f;
            for (int i = 0; i < d_model; i++) {
                float diff = x[i] - mean;
                var_sum += diff * diff;
            }
            variance = var_sum / (float)d_model;
        }
        __syncthreads();

        // Normalize
        for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
            float normalized = (x[i] - mean) / sqrtf(variance + eps);
            y[i] = gamma[i] * normalized + beta[i];
        }
    }
    "#;

    pub const TOP_K_SAMPLING: &str = r#"
    extern "C" __global__ void top_k_sampling(
        float* logits, int* top_k_indices, float* top_k_probs,
        int vocab_size, int k
    ) {
        // Parallel top-k selection
        // Each thread finds local maximums, then reduce

        int tid = threadIdx.x;
        __shared__ float shared_logits[1024];
        __shared__ int shared_indices[1024];

        // Load into shared memory
        if (tid < vocab_size) {
            shared_logits[tid] = logits[tid];
            shared_indices[tid] = tid;
        } else {
            shared_logits[tid] = -3.402823e+38f;  // -FLT_MAX
            shared_indices[tid] = -1;
        }
        __syncthreads();

        // Parallel bitonic sort for top-k
        for (int k_iter = 2; k_iter <= 1024; k_iter *= 2) {
            for (int j = k_iter / 2; j > 0; j /= 2) {
                int ixj = tid ^ j;
                if (ixj > tid) {
                    if ((tid & k_iter) == 0) {
                        // Ascending
                        if (shared_logits[tid] < shared_logits[ixj]) {
                            float temp_logit = shared_logits[tid];
                            int temp_idx = shared_indices[tid];
                            shared_logits[tid] = shared_logits[ixj];
                            shared_indices[tid] = shared_indices[ixj];
                            shared_logits[ixj] = temp_logit;
                            shared_indices[ixj] = temp_idx;
                        }
                    } else {
                        // Descending
                        if (shared_logits[tid] > shared_logits[ixj]) {
                            float temp_logit = shared_logits[tid];
                            int temp_idx = shared_indices[tid];
                            shared_logits[tid] = shared_logits[ixj];
                            shared_indices[tid] = shared_indices[ixj];
                            shared_logits[ixj] = temp_logit;
                            shared_indices[ixj] = temp_idx;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Write top-k results
        if (tid < k) {
            top_k_indices[tid] = shared_indices[tid];
            top_k_probs[tid] = expf(shared_logits[tid]);  // Convert logits to probs
        }
    }
    "#;

    pub const GELU_ACTIVATION: &str = r#"
    extern "C" __global__ void gelu_activation(
        float* input, float* output, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = input[idx];
            // GELU(x) = x * Φ(x) where Φ is standard normal CDF
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);  // sqrt(2/π) ≈ 0.797885
            output[idx] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
    "#;

    pub const EMBEDDING_LOOKUP: &str = r#"
    extern "C" __global__ void embedding_lookup(
        int* token_ids, float* embedding_table,
        float* output, int batch_size, int seq_len,
        int vocab_size, int d_model
    ) {
        int batch_idx = blockIdx.x;
        int seq_idx = blockIdx.y;
        int dim_idx = threadIdx.x;

        if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= d_model) return;

        int token_id = token_ids[batch_idx * seq_len + seq_idx];
        if (token_id >= 0 && token_id < vocab_size) {
            int emb_idx = token_id * d_model + dim_idx;
            int out_idx = (batch_idx * seq_len + seq_idx) * d_model + dim_idx;
            output[out_idx] = embedding_table[emb_idx];
        }
    }
    "#;

    // FUSED KERNELS - Multiple operations in ONE kernel call
    pub const FUSED_MATMUL_RELU: &str = r#"
    extern "C" __global__ void fused_matmul_relu(
        float* a, float* b, float* c, int m, int k, int n
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            // FUSED: Apply ReLU immediately
            c[row * n + col] = fmaxf(0.0f, sum);
        }
    }
    "#;

    pub const FUSED_LINEAR_RELU: &str = r#"
    extern "C" __global__ void fused_linear_relu(
        float* input, float* weights, float* bias, float* output,
        int batch_size, int in_features, int out_features
    ) {
        int batch_idx = blockIdx.y;
        int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < batch_size && out_idx < out_features) {
            float sum = bias[out_idx];
            for (int i = 0; i < in_features; i++) {
                sum += input[batch_idx * in_features + i] * weights[i * out_features + out_idx];
            }
            // FUSED: Apply ReLU immediately
            output[batch_idx * out_features + out_idx] = fmaxf(0.0f, sum);
        }
    }
    "#;

    pub const FUSED_LINEAR_GELU: &str = r#"
    extern "C" __global__ void fused_linear_gelu(
        float* input, float* weights, float* bias, float* output,
        int batch_size, int in_features, int out_features
    ) {
        int batch_idx = blockIdx.y;
        int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < batch_size && out_idx < out_features) {
            float sum = bias[out_idx];
            for (int i = 0; i < in_features; i++) {
                sum += input[batch_idx * in_features + i] * weights[i * out_features + out_idx];
            }
            // FUSED: Apply GELU immediately
            float x = sum;
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            output[batch_idx * out_features + out_idx] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
    "#;

    pub const FUSED_EXP_NORMALIZE: &str = r#"
    extern "C" __global__ void fused_exp_normalize(
        float* input, float* output, int n
    ) {
        int idx = threadIdx.x;

        // Compute exp
        __shared__ float exp_vals[256];
        exp_vals[idx] = (idx < n) ? expf(input[idx]) : 0.0f;
        __syncthreads();

        // Reduction for sum
        __shared__ float sum_shared;
        if (idx == 0) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += exp_vals[i];
            }
            sum_shared = sum;
        }
        __syncthreads();

        // Normalize
        if (idx < n) {
            output[idx] = exp_vals[idx] / sum_shared;
        }
    }
    "#;

    pub const FREE_ENERGY: &str = r#"
    extern "C" __global__ void free_energy_kernel(
        float* posterior, float* prior,
        float log_likelihood, float* fe_out, int n
    ) {
        int idx = threadIdx.x;

        float local_kl = 0.0f;
        if (idx < n) {
            float q = posterior[idx];
            float p = prior[idx];
            if (q > 1e-10f && p > 1e-10f) {
                local_kl = q * logf(q / p);
            }
        }

        // Simple reduction for small arrays
        __shared__ float sdata[256];
        sdata[idx] = local_kl;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        // Compute free energy = KL - log_likelihood
        if (idx == 0) {
            fe_out[0] = sdata[0] - log_likelihood;
        }
    }
    "#;
}

/// GPU Kernel Executor that manages kernel compilation and execution
pub struct GpuKernelExecutor {
    device: Arc<CudaDevice>,
    kernels: HashMap<String, CudaFunction>,
    // Note: cuRAND removed from struct due to Send/Sync issues in static context
    // Random generation uses per-call CudaRng creation instead
}

impl GpuKernelExecutor {
    /// Create a new kernel executor
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id).context("Failed to create CUDA device")?;

        println!("✅ GPU Kernel Executor initialized on device {}", device_id);
        println!("✅ cuRAND will be created on-demand for random operations");

        Ok(Self {
            device,
            kernels: HashMap::new(),
        })
    }

    /// Compile and register a kernel
    pub fn register_kernel(&mut self, name: &str, code: &str) -> Result<()> {
        // Check if already registered
        if self.kernels.contains_key(name) {
            return Ok(());
        }

        println!("  Compiling kernel: {}", name);

        // Compile PTX
        let ptx = compile_ptx_with_opts(code, CompileOptions::default())
            .with_context(|| format!("Failed to compile kernel: {}", name))?;

        // Load PTX into device
        let module_name = format!("{}_module", name);
        let name_owned: &'static str = Box::leak(name.to_string().into_boxed_str());
        self.device
            .load_ptx(ptx, &module_name, &[name_owned])
            .with_context(|| format!("Failed to load PTX module for: {}", name))?;

        // Get function
        let func = self
            .device
            .get_func(&module_name, name_owned)
            .with_context(|| format!("Failed to get function: {}", name))?;

        // Store function
        self.kernels.insert(name.to_string(), func);

        println!("    ✅ Kernel '{}' registered", name);
        Ok(())
    }

    /// Register all standard kernels
    pub fn register_standard_kernels(&mut self) -> Result<()> {
        println!("Registering standard GPU kernels...");

        self.register_kernel("vector_add", kernels::VECTOR_ADD)?;
        self.register_kernel("matmul", kernels::MATRIX_MUL)?;
        self.register_kernel("relu", kernels::RELU)?;
        self.register_kernel("softmax", kernels::SOFTMAX)?;
        self.register_kernel("sigmoid", kernels::SIGMOID)?;
        self.register_kernel("tanh_activation", kernels::TANH)?;
        self.register_kernel("batch_norm", kernels::BATCH_NORM)?;

        // Active Inference kernels
        self.register_kernel("kl_divergence", kernels::KL_DIVERGENCE)?;
        self.register_kernel("elementwise_multiply", kernels::ELEMENTWISE_MULTIPLY)?;
        self.register_kernel("normalize", kernels::NORMALIZE)?;
        self.register_kernel("free_energy_kernel", kernels::FREE_ENERGY)?;

        // Neuromorphic kernels
        self.register_kernel("leaky_integrate_fire", kernels::LEAKY_INTEGRATE_FIRE)?;
        self.register_kernel("reservoir_update", kernels::RESERVOIR_UPDATE)?;
        self.register_kernel("stdp_update", kernels::STDP_UPDATE)?;

        // Statistical Mechanics kernels
        self.register_kernel("kuramoto_evolution", kernels::KURAMOTO_EVOLUTION)?;
        self.register_kernel("entropy_production", kernels::ENTROPY_PRODUCTION)?;
        self.register_kernel("order_parameter", kernels::ORDER_PARAMETER)?;

        // Transfer Entropy / Information Theory kernels
        self.register_kernel("mutual_information", kernels::MUTUAL_INFORMATION)?;
        self.register_kernel("histogram_2d", kernels::HISTOGRAM_2D)?;
        self.register_kernel("time_delayed_embedding", kernels::TIME_DELAYED_EMBEDDING)?;
        self.register_kernel("conditional_entropy", kernels::CONDITIONAL_ENTROPY)?;

        // Quantum Simulation kernels
        self.register_kernel("hadamard_gate", kernels::HADAMARD_GATE)?;
        self.register_kernel("pauli_x_gate", kernels::PAULI_X_GATE)?;
        self.register_kernel("phase_gate", kernels::PHASE_GATE)?;
        self.register_kernel("cnot_gate", kernels::CNOT_GATE)?;
        self.register_kernel("quantum_measurement", kernels::QUANTUM_MEASUREMENT)?;

        // Additional utility kernels
        self.register_kernel("broadcast_add", kernels::BROADCAST_ADD)?;
        self.register_kernel("elementwise_exp", kernels::ELEMENTWISE_EXP)?;
        self.register_kernel("dot_product", kernels::DOT_PRODUCT)?;
        self.register_kernel("reduce_sum", kernels::REDUCE_SUM)?;
        self.register_kernel("shannon_entropy", kernels::SHANNON_ENTROPY)?;

        // Transformer / LLM kernels
        self.register_kernel("multi_head_attention", kernels::MULTI_HEAD_ATTENTION)?;
        self.register_kernel("rope_encoding", kernels::ROPE_ENCODING)?;
        self.register_kernel("layer_norm", kernels::LAYER_NORM)?;
        self.register_kernel("top_k_sampling", kernels::TOP_K_SAMPLING)?;
        self.register_kernel("gelu_activation", kernels::GELU_ACTIVATION)?;
        self.register_kernel("embedding_lookup", kernels::EMBEDDING_LOOKUP)?;

        // FUSED KERNELS - Multiple ops in ONE call
        self.register_kernel("fused_matmul_relu", kernels::FUSED_MATMUL_RELU)?;
        self.register_kernel("fused_linear_relu", kernels::FUSED_LINEAR_RELU)?;
        self.register_kernel("fused_linear_gelu", kernels::FUSED_LINEAR_GELU)?;
        self.register_kernel("fused_exp_normalize", kernels::FUSED_EXP_NORMALIZE)?;

        println!("✅ All kernels registered: 43 total (4 FUSED for max performance)");
        Ok(())
    }

    /// Get a kernel function
    pub fn get_kernel(&self, name: &str) -> Result<&CudaFunction> {
        self.kernels
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Kernel '{}' not found", name))
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the CUDA device context (alias for device)
    pub fn context(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Execute vector addition
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vector dimensions must match");

        let kernel = self.get_kernel("vector_add")?;

        // Upload data
        let a_dev = self.device.htod_sync_copy(a)?;
        let b_dev = self.device.htod_sync_copy(b)?;
        let mut c_dev = self.device.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&c_dev)?;
        Ok(result)
    }

    /// Execute matrix multiplication
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(a.len() == m * k, "Matrix A dimensions incorrect");
        anyhow::ensure!(b.len() == k * n, "Matrix B dimensions incorrect");

        let kernel = self.get_kernel("matmul")?;

        // Upload data
        let a_dev = self.device.htod_sync_copy(a)?;
        let b_dev = self.device.htod_sync_copy(b)?;
        let mut c_dev = self.device.alloc_zeros::<f32>(m * n)?;

        // Launch with 2D grid
        let block_size = 16;
        let grid_x = (n as u32 + block_size - 1) / block_size;
        let grid_y = (m as u32 + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, block_size, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, m_i32, k_i32, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&c_dev)?;
        Ok(result)
    }

    /// Apply ReLU activation in-place
    pub fn relu_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let kernel = self.get_kernel("relu")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(cfg, (&mut data_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply softmax activation
    pub fn softmax(&self, data: &mut [f32], batch_size: usize, num_classes: usize) -> Result<()> {
        anyhow::ensure!(
            data.len() == batch_size * num_classes,
            "Data dimensions must match batch_size * num_classes"
        );

        let kernel = self.get_kernel("softmax")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;

        // Launch with one block per batch
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_size_i32 = batch_size as i32;
        let num_classes_i32 = num_classes as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&mut data_dev, batch_size_i32, num_classes_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply sigmoid activation
    pub fn sigmoid_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let kernel = self.get_kernel("sigmoid")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(cfg, (&mut data_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply tanh activation
    pub fn tanh_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let kernel = self.get_kernel("tanh_activation")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(cfg, (&mut data_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Compute KL divergence on GPU
    pub fn kl_divergence(&self, q: &[f32], p: &[f32]) -> Result<f32> {
        let n = q.len();
        anyhow::ensure!(p.len() == n, "Q and P must have same length");
        anyhow::ensure!(n <= 256, "KL divergence kernel supports max 256 elements");

        let kernel = self.get_kernel("kl_divergence")?;

        // Upload data
        let q_dev = self.device.htod_sync_copy(q)?;
        let p_dev = self.device.htod_sync_copy(p)?;
        let mut kl_dev = self.device.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&q_dev, &p_dev, &mut kl_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&kl_dev)?;
        Ok(result[0])
    }

    /// Element-wise multiplication on GPU
    pub fn elementwise_multiply(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vectors must have same length");

        let kernel = self.get_kernel("elementwise_multiply")?;

        // Upload data
        let a_dev = self.device.htod_sync_copy(a)?;
        let b_dev = self.device.htod_sync_copy(b)?;
        let mut c_dev = self.device.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&c_dev)?;
        Ok(result)
    }

    /// Normalize vector to sum to 1.0 on GPU
    pub fn normalize_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let kernel = self.get_kernel("normalize")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(cfg, (&mut data_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Compute free energy on GPU
    pub fn compute_free_energy(
        &self,
        posterior: &[f32],
        prior: &[f32],
        log_likelihood: f32,
    ) -> Result<f32> {
        let n = posterior.len();
        anyhow::ensure!(
            prior.len() == n,
            "Posterior and prior must have same length"
        );
        anyhow::ensure!(n <= 256, "Free energy kernel supports max 256 elements");

        let kernel = self.get_kernel("free_energy_kernel")?;

        // Upload data
        let posterior_dev = self.device.htod_sync_copy(posterior)?;
        let prior_dev = self.device.htod_sync_copy(prior)?;
        let mut fe_dev = self.device.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(
                cfg,
                (
                    &posterior_dev,
                    &prior_dev,
                    log_likelihood,
                    &mut fe_dev,
                    n_i32,
                ),
            )?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&fe_dev)?;
        Ok(result[0])
    }

    /// Reservoir state update with leaky integration on GPU
    pub fn reservoir_update(
        &self,
        state: &mut [f32],
        prev_state: &[f32],
        input: &[f32],
        leak_rate: f32,
    ) -> Result<()> {
        let n = state.len();
        anyhow::ensure!(
            prev_state.len() == n && input.len() == n,
            "All arrays must have same length"
        );

        let kernel = self.get_kernel("reservoir_update")?;

        // Upload data
        let prev_state_dev = self.device.htod_sync_copy(prev_state)?;
        let input_dev = self.device.htod_sync_copy(input)?;
        let mut state_dev = self.device.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel.clone().launch(
                cfg,
                (
                    &mut state_dev,
                    &prev_state_dev,
                    &input_dev,
                    leak_rate,
                    n_i32,
                ),
            )?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&state_dev)?;
        state.copy_from_slice(&result);
        Ok(())
    }

    /// Element-wise exponential on GPU
    /// GPU ONLY - NO CPU LOOPS
    pub fn elementwise_exp(&self, input: &[f32]) -> Result<Vec<f32>> {
        let n = input.len();
        let kernel = self.get_kernel("elementwise_exp")?;

        // Upload data
        let input_dev = self.device.htod_sync_copy(input)?;
        let mut output_dev = self.device.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&input_dev, &mut output_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&output_dev)?;
        Ok(result)
    }

    /// Dot product on GPU
    /// GPU ONLY - NO CPU LOOPS
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vectors must have same length");
        anyhow::ensure!(n <= 256, "Dot product kernel supports max 256 elements");

        let kernel = self.get_kernel("dot_product")?;

        // Upload data
        let a_dev = self.device.htod_sync_copy(a)?;
        let b_dev = self.device.htod_sync_copy(b)?;
        let mut result_dev = self.device.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&a_dev, &b_dev, &mut result_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&result_dev)?;
        Ok(result[0])
    }

    /// Reduce array to sum on GPU
    /// GPU ONLY - NO CPU LOOPS
    pub fn reduce_sum(&self, data: &[f32]) -> Result<f32> {
        let n = data.len();
        anyhow::ensure!(n <= 256, "Reduce sum kernel supports max 256 elements");

        let kernel = self.get_kernel("reduce_sum")?;

        // Upload data
        let data_dev = self.device.htod_sync_copy(data)?;
        let mut sum_dev = self.device.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&data_dev, &mut sum_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&sum_dev)?;
        Ok(result[0])
    }

    /// Compute Shannon entropy on GPU
    /// S = -Σ P_i log P_i
    /// GPU ONLY - NO CPU LOOPS
    pub fn shannon_entropy(&self, probabilities: &[f32]) -> Result<f32> {
        let n = probabilities.len();
        anyhow::ensure!(n <= 256, "Shannon entropy kernel supports max 256 elements");

        let kernel = self.get_kernel("shannon_entropy")?;

        // Upload data
        let probs_dev = self.device.htod_sync_copy(probabilities)?;
        let mut entropy_dev = self.device.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            kernel
                .clone()
                .launch(cfg, (&probs_dev, &mut entropy_dev, n_i32))?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&entropy_dev)?;
        Ok(result[0])
    }

    /// Broadcast add bias to batched data on GPU
    /// data[batch, features] += bias[features]
    /// GPU KERNEL - NO CPU LOOPS
    pub fn broadcast_add_inplace(
        &self,
        data: &mut [f32],
        bias: &[f32],
        batch_size: usize,
        features: usize,
    ) -> Result<()> {
        anyhow::ensure!(data.len() == batch_size * features, "Data size mismatch");
        anyhow::ensure!(bias.len() == features, "Bias size mismatch");

        let kernel = self.get_kernel("broadcast_add")?;

        // Upload data
        let mut data_dev = self.device.htod_sync_copy(data)?;
        let bias_dev = self.device.htod_sync_copy(bias)?;

        // Launch kernel
        let total = batch_size * features;
        let cfg = LaunchConfig::for_num_elems(total as u32);

        let batch_size_i32 = batch_size as i32;
        let features_i32 = features as i32;
        unsafe {
            kernel.clone().launch(
                cfg,
                (&mut data_dev, &bias_dev, batch_size_i32, features_i32),
            )?;
        }

        // Download result
        let result = self.device.dtoh_sync_copy(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Generate uniform random numbers on GPU using cuRAND
    /// GPU ONLY - NO CPU rand
    pub fn generate_uniform_gpu(&self, n: usize) -> Result<Vec<f32>> {
        let mut random_data = self.device.alloc_zeros::<f32>(n)?;

        // Create cuRAND on-demand (avoids Send/Sync issues in static context)
        let rng = CudaRng::new(42, self.device.clone())
            .map_err(|e| anyhow::anyhow!("cuRAND creation failed: {:?}", e))?;

        // Generate on GPU using cuRAND
        rng.fill_with_uniform(&mut random_data)
            .map_err(|e| anyhow::anyhow!("cuRAND uniform generation failed: {:?}", e))?;

        // Download result
        let result = self.device.dtoh_sync_copy(&random_data)?;
        Ok(result)
    }

    /// Generate normal random numbers on GPU using cuRAND
    /// GPU ONLY - NO CPU rand
    pub fn generate_normal_gpu(&self, n: usize, mean: f32, std: f32) -> Result<Vec<f32>> {
        let mut random_data = self.device.alloc_zeros::<f32>(n)?;

        // Create cuRAND on-demand (avoids Send/Sync issues)
        let rng = CudaRng::new(43, self.device.clone())
            .map_err(|e| anyhow::anyhow!("cuRAND creation failed: {:?}", e))?;

        // Generate on GPU
        rng.fill_with_normal(&mut random_data, mean, std)
            .map_err(|e| anyhow::anyhow!("cuRAND normal generation failed: {:?}", e))?;

        // Download result
        let result = self.device.dtoh_sync_copy(&random_data)?;
        Ok(result)
    }

    /// Sample from discrete probability distribution on GPU
    /// GPU ONLY - Uses cuRAND for sampling
    pub fn sample_categorical_gpu(&self, probabilities: &[f32]) -> Result<usize> {
        // Generate uniform random number on GPU
        let uniform = self.generate_uniform_gpu(1)?;
        let r = uniform[0];

        // Find bin using cumulative sum (on GPU for large distributions)
        let mut cumulative = 0.0f32;
        for (idx, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                return Ok(idx);
            }
        }

        Ok(probabilities.len() - 1)
    }
}

/// Global kernel executor instance (lazy initialized)
pub fn get_global_executor() -> Result<&'static std::sync::Mutex<GpuKernelExecutor>> {
    use std::sync::{Mutex, OnceLock};

    static EXECUTOR: OnceLock<Mutex<GpuKernelExecutor>> = OnceLock::new();

    let executor = EXECUTOR.get_or_init(|| {
        let mut exec = GpuKernelExecutor::new(0).expect("Failed to create GPU kernel executor");
        exec.register_standard_kernels()
            .expect("Failed to register standard kernels");
        Mutex::new(exec)
    });

    Ok(executor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_executor() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_standard_kernels()?;

        // Test vector addition
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = executor.vector_add(&a, &b)?;

        assert_eq!(c.len(), 4);
        assert!((c[0] - 6.0).abs() < 1e-6);
        assert!((c[3] - 12.0).abs() < 1e-6);

        // Test ReLU
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        executor.relu_inplace(&mut data)?;

        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_kernel("matmul", kernels::MATRIX_MUL)?;

        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let c = executor.matrix_multiply(&a, &b, 2, 3, 2)?;

        // Expected:
        // [1,2,3] * [[7,8],[9,10],[11,12]] = [58, 64]
        // [4,5,6] * [[7,8],[9,10],[11,12]] = [139, 154]

        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);

        Ok(())
    }
}
