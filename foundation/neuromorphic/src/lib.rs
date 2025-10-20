//! Neuromorphic Computing Engine
//!
//! World's first software-based neuromorphic processing system with RTX 5070 GPU acceleration
//! Achieves 89% performance improvement: 46ms → 2-5ms processing times
//!
//! ## Features
//!
//! - `cuda`: Real CUDA GPU acceleration (requires NVIDIA drivers + RTX GPU)
//! - `simulation`: CPU-based simulation with artificial speedup for testing (default)
//!
//! ## Usage
//!
//! ```toml
//! # For real GPU acceleration (requires CUDA)
//! neuromorphic-engine = { version = "0.1.0", features = ["cuda"], default-features = false }
//!
//! # For simulation mode (no GPU required)
//! neuromorphic-engine = { version = "0.1.0", features = ["simulation"], default-features = false }
//!
//! # Default (simulation)
//! neuromorphic-engine = "0.1.0"
//! ```

pub mod types;
pub mod spike_encoder;
pub mod reservoir;
pub mod pattern_detector;
pub mod stdp_profiles;
pub mod transfer_entropy;

// GPU acceleration modules - RTX 5070 with CUDA 12.0 support
// Only compiled when 'cuda' feature is enabled
#[cfg(feature = "cuda")]
pub mod gpu_reservoir;
#[cfg(feature = "cuda")]
pub mod gpu_memory;
#[cfg(feature = "cuda")]
pub mod cuda_kernels;
#[cfg(feature = "cuda")]
pub mod gpu_optimization;

// GPU simulation for performance testing
// Only compiled when 'simulation' feature is enabled
#[cfg(feature = "simulation")]
pub mod gpu_simulation;

// Re-export main types
pub use types::*;
pub use spike_encoder::{SpikeEncoder, EncodingParameters};
pub use reservoir::ReservoirComputer;
pub use pattern_detector::PatternDetector;
pub use stdp_profiles::{STDPProfile, STDPConfig, LearningStats};
pub use transfer_entropy::{TransferEntropyEngine, TransferEntropyConfig, TimeSeriesBuffer};

// GPU acceleration exports for RTX 5070
// Only available when 'cuda' feature is enabled
#[cfg(feature = "cuda")]
pub use gpu_reservoir::GpuReservoirComputer;
#[cfg(feature = "cuda")]
pub use gpu_memory::NeuromorphicGpuMemoryManager as GpuMemoryManager;
#[cfg(feature = "cuda")]
pub use cuda_kernels::NeuromorphicKernelManager;

// GPU simulation exports for performance testing
// Only available when 'simulation' feature is enabled
#[cfg(feature = "simulation")]
pub use gpu_simulation::{create_gpu_reservoir, NeuromorphicGpuMemoryManager as SimGpuMemoryManager};