//! Real-Time Telemetry System
//!
//! Provides fine-grained performance monitoring and metrics collection
//! for the PRISM world-record pipeline.

pub mod handle;
pub mod run_metric;
pub mod multi_gpu;

pub use handle::TelemetryHandle;
pub use run_metric::{
    GpuUsageSummary, OptimizationGuidance, PhaseExecMode, PhaseName, PhaseStats, RunMetric,
    RunSummary,
};
pub use multi_gpu::{
    ReplicaTelemetry, DeviceUtilization, CoordinationMetrics, MultiGpuSummary, ReplicaWatchdog,
};
