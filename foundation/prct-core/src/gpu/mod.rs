//! GPU Infrastructure for PRISM Pipeline
//!
//! Provides stream management, event synchronization, centralized
//! GPU state, and multi-GPU device pooling for the world-record pipeline.

pub mod stream_pool;
pub mod event;
pub mod state;
pub mod multi_device_pool;
pub mod device_topology;
pub mod replica_planner;
pub mod distributed_runtime;
pub mod device_profile;

pub use stream_pool::CudaStreamPool;
pub use event::{EventRegistry, event_names};
pub use state::{PipelineGpuState, StreamMode};
pub use multi_device_pool::MultiGpuDevicePool;
pub use device_topology::{DeviceInfo, DeviceSelector, discover_devices, get_device_info};
pub use replica_planner::{ReplicaAssignment, ReplicaPlanner, PhaseMask, ReplicaStreams};
pub use distributed_runtime::{
    ReplicaCommand, ReplicaSupervisor, ReplicaHandle, DistributedRuntime,
    ReplicaEvent, SnapshotMeta, ReplicaMetrics,
};
pub use device_profile::{
    DeviceProfile, DeviceProfiles, ProfileMode, ProfileStrategy, ReplicaCount,
    load_device_profile,
};
