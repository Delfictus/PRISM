//! GPU acceleration module for PRISM-AI
//!
//! Provides GPU memory management and operations for neural network acceleration
//! using cudarc with CUDA 13 support for RTX 5070

// Temporarily disabled until cudarc API is fixed
// pub mod memory_manager;
// pub mod tensor_ops;
// pub mod kernel_launcher;

// GPU-ONLY modules - NO CPU FALLBACK
pub mod layers;
pub mod gpu_enabled;  // GPU-only implementation with kernel execution
pub mod kernel_executor;  // GPU kernel executor
pub mod gpu_tensor_optimized;  // FULLY OPTIMIZED: CudaSlice, persistent GPU, fused kernels

// Use GPU-enabled implementation - NO CPU FALLBACK
pub use gpu_enabled::{
    SimpleGpuContext as GpuMemoryPool,
    SimpleGpuTensor as GpuTensor,
    SimpleGpuLinear as GpuLinear,
    SimpleGpuBuffer as GpuBuffer,
};

// Export OPTIMIZED GPU tensors (4-10x faster)
pub use gpu_tensor_optimized::{
    GpuTensorOpt,
    FusedLinearLayerOpt,
    OptimizedGpuNetwork,
};

// Also export the layers
pub use layers::GpuLinear as GpuLinearLayer;

// Export kernel executor
pub use kernel_executor::{GpuKernelExecutor, get_global_executor};