//! GPU-accelerated neural network layers

pub mod linear;
pub mod activation;

pub use linear::GpuLinear;
pub use activation::{relu_gpu, softmax_gpu};