//! Integration module

#[cfg(feature = "pwsa")]
pub mod mission_charlie_integration;
#[cfg(feature = "pwsa")]
pub mod prism_ai_integration;
#[cfg(feature = "pwsa")]
pub mod pwsa_llm_bridge;

#[cfg(feature = "pwsa")]
pub use mission_charlie_integration::{
    ConsensusType, DiagnosticReport, IntegratedResponse, IntegrationConfig,
    MissionCharlieIntegration, SystemStatus,
};
#[cfg(feature = "pwsa")]
pub use prism_ai_integration::{
    OrchestratorConfig, OrchestratorMetrics, PrismAIOrchestrator, QuantumEnhancement,
    SensorContext, UnifiedResponse,
};
#[cfg(feature = "pwsa")]
pub use pwsa_llm_bridge::{CompleteIntelligence, PwsaLLMFusionPlatform};
