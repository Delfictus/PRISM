//! Integration module

pub mod pwsa_llm_bridge;
pub mod mission_charlie_integration;
pub mod prism_ai_integration;

pub use pwsa_llm_bridge::{PwsaLLMFusionPlatform, CompleteIntelligence};
pub use mission_charlie_integration::{
    MissionCharlieIntegration, IntegrationConfig, IntegratedResponse,
    ConsensusType, SystemStatus, DiagnosticReport,
};
pub use prism_ai_integration::{
    PrismAIOrchestrator, OrchestratorConfig, UnifiedResponse,
    SensorContext, QuantumEnhancement, OrchestratorMetrics,
};
