//! Mission Charlie: Thermodynamic LLM Intelligence Fusion
//!
//! Multi-source intelligence fusion combining:
//! - PWSA sensor data (Mission Bravo)
//! - LLM-generated intelligence analysis
//! - Constitutional AI framework (Articles I, III, IV)
//!
//! Revolutionary Features:
//! - Transfer entropy causal LLM routing (patent-worthy)
//! - Active inference API clients (patent-worthy)
//! - Quantum semantic caching
//! - Thermodynamic consensus optimization

pub mod llm_clients;
pub mod thermodynamic;
pub mod causal_analysis;
pub mod active_inference;
pub mod synthesis;
pub mod integration;
pub mod privacy;
pub mod monitoring;
pub mod optimization;
pub mod caching;
pub mod routing;
pub mod validation;
pub mod semantic_analysis;
pub mod neuromorphic;
pub mod manifold;
pub mod multimodal;
pub mod production;
pub mod local_llm;

// New algorithm modules
pub mod cache;
pub mod consensus;
pub mod decomposition;
pub mod inference;
pub mod causality;
pub mod quantum;

// Core exports
pub use production::{ProductionErrorHandler, ProductionLogger, MissionCharlieConfig};

// Algorithm exports
pub use cache::quantum_cache::QuantumSemanticCache as QuantumApproximateCache;
pub use consensus::quantum_voting::QuantumConsensusOptimizer as QuantumVotingConsensus;
pub use thermodynamic::quantum_consensus::QuantumConsensusOptimizer as ThermodynamicConsensus;
pub use routing::transfer_entropy_router::TransferEntropyPromptRouter as TransferEntropyRouter;
pub use decomposition::pid_synergy::PIDSynergyDecomposition;
pub use inference::hierarchical_active_inference::HierarchicalActiveInference;
pub use inference::joint_active_inference::JointActiveInference;
pub use neuromorphic::unified_neuromorphic::UnifiedNeuromorphicProcessor;
pub use causality::bidirectional_causality::BidirectionalCausalityAnalyzer;
pub use optimization::geometric_manifold::GeometricManifoldOptimizer;
pub use quantum::quantum_entanglement_measures::QuantumEntanglementAnalyzer;
#[cfg(feature = "pwsa")]
pub use integration::mission_charlie_integration::MissionCharlieIntegration;
#[cfg(feature = "pwsa")]
pub use integration::prism_ai_integration::{PrismAIOrchestrator, OrchestratorConfig, UnifiedResponse};

// Main orchestrator
pub use llm_clients::LLMOrchestrator;

// Error handling
pub mod errors;
pub use errors::OrchestrationError;

// Common types
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
    pub confidence: f64,
    pub model: String,
    pub latency_ms: u64,
}
