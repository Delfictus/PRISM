//! GPU Infrastructure for PRISM Pipeline
//!
//! Provides stream management, event synchronization, and centralized
//! GPU state for the world-record pipeline.

pub mod stream_pool;
pub mod event;
pub mod state;

pub use stream_pool::CudaStreamPool;
pub use event::{EventRegistry, event_names};
pub use state::{PipelineGpuState, StreamMode};
