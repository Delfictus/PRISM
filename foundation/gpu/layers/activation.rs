//! GPU-accelerated activation functions (using simple implementation)

use crate::gpu::gpu_enabled::SimpleGpuTensor;
use anyhow::Result;

/// Apply ReLU activation in-place
pub fn relu_gpu(tensor: &mut SimpleGpuTensor) -> Result<()> {
    tensor.relu()
}

/// Apply softmax activation in-place
pub fn softmax_gpu(tensor: &mut SimpleGpuTensor, dim: usize) -> Result<()> {
    tensor.softmax(dim)
}

/// Apply sigmoid activation
pub fn sigmoid_gpu(tensor: &mut SimpleGpuTensor) -> Result<()> {
    // Get data, apply sigmoid, put back
    let mut data = tensor.to_cpu()?;
    for x in &mut data {
        *x = 1.0 / (1.0 + (-*x).exp());
    }

    // Reconstruct tensor
    let shape = tensor.shape().to_vec();
    *tensor = SimpleGpuTensor::from_cpu(data, shape)?;

    Ok(())
}

/// Apply tanh activation
pub fn tanh_gpu(tensor: &mut SimpleGpuTensor) -> Result<()> {
    // Get data, apply tanh, put back
    let mut data = tensor.to_cpu()?;
    for x in &mut data {
        *x = x.tanh();
    }

    // Reconstruct tensor
    let shape = tensor.shape().to_vec();
    *tensor = SimpleGpuTensor::from_cpu(data, shape)?;

    Ok(())
}
