//! Integration module

#[cfg(feature = "pwsa")]
pub mod pwsa_llm_bridge;
#[cfg(feature = "pwsa")]
pub mod mission_charlie_integration;
#[cfg(feature = "pwsa")]
pub mod prism_ai_integration;

#[cfg(feature = "pwsa")]
pub use pwsa_llm_bridge::{PwsaLLMFusionPlatform, CompleteIntelligence};
#[cfg(feature = "pwsa")]
pub use mission_charlie_integration::{
    MissionCharlieIntegration, IntegrationConfig, IntegratedResponse,
    ConsensusType, SystemStatus, DiagnosticReport,
};
#[cfg(feature = "pwsa")]
pub use prism_ai_integration::{
    PrismAIOrchestrator, OrchestratorConfig, UnifiedResponse,
    SensorContext, QuantumEnhancement, OrchestratorMetrics,
};
