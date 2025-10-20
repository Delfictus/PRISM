//! Platform Foundation
//!
//! Unified API for the world's first software-based neuromorphic-quantum computing platform

pub mod adapters;
pub mod adaptive_coupling;
pub mod adp;
pub mod coupling_physics;
pub mod ingestion;
pub mod phase_causal_matrix;
pub mod platform;
pub mod types;

// Re-export main components
pub use adapters::{AlpacaMarketDataSource, OpticalSensorArray, SyntheticDataSource};
pub use adaptive_coupling::{
    AdaptiveCoupling, AdaptiveParameter, CouplingValues, PerformanceMetrics, PerformanceSummary,
};
pub use adp::{
    Action, AdaptiveDecisionProcessor, AdpStats, Decision, ReinforcementLearner, RlConfig, RlStats,
    State,
};
pub use coupling_physics::{
    InformationMetrics, KuramotoSync, NeuroQuantumCoupling, PhysicsCoupling, QuantumNeuroCoupling,
    StabilityAnalysis,
};
pub use ingestion::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, ComponentHealth, DataPoint,
    DataSource, EngineConfig, HealthMetrics, HealthReport, HealthStatus, IngestionConfig,
    IngestionEngine, IngestionError, IngestionStats, RetryConfig, RetryPolicy, SourceConfig,
    SourceInfo,
};
pub use phase_causal_matrix::{PcmConfig, PhaseCausalMatrixProcessor};
pub use platform::NeuromorphicQuantumPlatform;
pub use types::*;
