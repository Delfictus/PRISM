//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod federated;
pub mod ontology;
pub mod orchestrator;

pub use ontology::{ConceptAnchor, OntologyDigest, OntologyLedger};
pub use orchestrator::{
    EvolutionMetrics, EvolutionOutcome, EvolutionPlan, MetaOrchestrator, VariantEvaluation,
    VariantGenome, VariantParameter,
};
