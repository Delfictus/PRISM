//! Platform Foundation
//!
//! Unified API for the world's first software-based neuromorphic-quantum computing platform

pub mod active_inference;
pub mod adapters;
pub mod adaptive_coupling;
pub mod adp;
pub mod coupling_physics;
pub mod gpu;
pub mod information_theory;
pub mod ingestion;
pub mod integration;
pub mod orchestration;
pub mod phase_causal_matrix;
pub mod platform;
#[cfg(feature = "pwsa")]
pub mod pwsa;
#[cfg(feature = "quantum_mlir_support")]
pub mod quantum_mlir;
pub mod resilience;
pub mod statistical_mechanics;
pub mod types;

// Re-export neuromorphic crate as a module
pub use neuromorphic_engine as neuromorphic;

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
#[cfg(feature = "pwsa")]
pub use orchestration::{OrchestratorConfig, PrismAIOrchestrator, UnifiedResponse};
pub use phase_causal_matrix::{PcmConfig, PhaseCausalMatrixProcessor};
pub use platform::NeuromorphicQuantumPlatform;
pub use types::*;

/// Placeholder binary entry point when compiled as `prism-ai`.
///
/// The platform is primarily exposed as a library. A no-op `main` keeps
/// the legacy binary target building cleanly while end-to-end orchestration
/// flows run from higher-level executors.
#[allow(dead_code)]
pub fn main() {
    env_logger::init();
    log::info!("PRISM foundation library loaded – no direct CLI actions.");
}
