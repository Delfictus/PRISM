//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod ontology;
pub mod orchestrator;
pub mod reflexive;

pub use ontology::{ConceptAnchor, OntologyDigest, OntologyLedger};
pub use orchestrator::{
    EvolutionMetrics, EvolutionOutcome, EvolutionPlan, MetaOrchestrator, VariantEvaluation,
    VariantGenome, VariantParameter,
};
pub use reflexive::{
    GovernanceMode, ReflexiveConfig, ReflexiveController, ReflexiveDecision, ReflexiveMetric,
    ReflexiveSnapshot,
};
