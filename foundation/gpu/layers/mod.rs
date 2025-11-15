//! GPU-accelerated neural network layers

pub mod activation;
pub mod linear;

pub use activation::{relu_gpu, softmax_gpu};
pub use linear::GpuLinear;
