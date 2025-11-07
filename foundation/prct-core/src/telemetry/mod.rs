//! Real-Time Telemetry System
//!
//! Provides fine-grained performance monitoring and metrics collection
//! for the PRISM world-record pipeline.

pub mod run_metric;
pub mod handle;

pub use run_metric::{
    RunMetric,
    RunSummary,
    PhaseName,
    PhaseExecMode,
    PhaseStats,
    GpuUsageSummary,
};
pub use handle::TelemetryHandle;
