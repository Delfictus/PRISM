//! GPU acceleration module for PRISM-AI
//!
//! Provides GPU memory management and operations for neural network acceleration
//! using cudarc with CUDA 13 support for RTX 5070

// Temporarily disabled until cudarc API is fixed
// pub mod memory_manager;
// pub mod tensor_ops;
// pub mod kernel_launcher;

// GPU-ONLY modules - NO CPU FALLBACK
pub mod gpu_enabled; // GPU-only implementation with kernel execution
pub mod gpu_tensor_optimized;
pub mod kernel_executor; // GPU kernel executor
pub mod layers; // FULLY OPTIMIZED: CudaSlice, persistent GPU, fused kernels

// Use GPU-enabled implementation - NO CPU FALLBACK
pub use gpu_enabled::{
    SimpleGpuBuffer as GpuBuffer, SimpleGpuContext as GpuMemoryPool, SimpleGpuLinear as GpuLinear,
    SimpleGpuTensor as GpuTensor,
};

// Export OPTIMIZED GPU tensors (4-10x faster)
pub use gpu_tensor_optimized::{FusedLinearLayerOpt, GpuTensorOpt, OptimizedGpuNetwork};

// Also export the layers
pub use layers::GpuLinear as GpuLinearLayer;

// Export kernel executor
pub use kernel_executor::{get_global_executor, GpuKernelExecutor};
