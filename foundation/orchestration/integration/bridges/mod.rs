//! Integration Bridges
//!
//! This module contains bridges that connect different components of the PRISM-AI system
//! for unified operations and consensus mechanisms.

pub mod llm_consensus_bridge;      // Simplified 3-algorithm version
pub mod full_consensus_bridge;      // Complete 12-algorithm version

// Re-export simplified version for backward compatibility
pub use llm_consensus_bridge::{
    ConsensusRequest, ConsensusResponse, ModelResponse, 
    ConsensusWeights, ConsensusFusion, ConsensusMetrics,
    QuantumVoteResult, ThermodynamicConsensusResult, TransferEntropyRoutingResult,
    LLMResponse, Usage,
};

// Re-export full version for complete implementation
pub use full_consensus_bridge::{
    AlgorithmContributions,
    AlgorithmWeights,
    FullConsensusOrchestrator,
    // Individual algorithm results
    PIDResult,
    HierarchicalResult,
    NeuromorphicResult,
    CausalityResult,
    JointInferenceResult,
    ManifoldResult,
    EntanglementResult,
    FusedConsensus,
};