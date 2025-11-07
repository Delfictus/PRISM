//! GPU Infrastructure for PRISM Pipeline
//!
//! Provides stream management, event synchronization, centralized
//! GPU state, and multi-GPU device pooling for the world-record pipeline.

pub mod stream_pool;
pub mod event;
pub mod state;
pub mod multi_device_pool;

pub use stream_pool::CudaStreamPool;
pub use event::{EventRegistry, event_names};
pub use state::{PipelineGpuState, StreamMode};
pub use multi_device_pool::MultiGpuDevicePool;
