//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod ontology;
pub mod orchestrator;
pub mod reflexive;

pub use ontology::{
    alignment::{align_variants, AlignmentSummary, VariantAlignment},
    ConceptAnchor, FileOntologyStorage, OntologyDigest, OntologyLedger, OntologyService,
    OntologySnapshot,
};
pub use orchestrator::{
    EvolutionMetrics, EvolutionOutcome, EvolutionPlan, MetaOrchestrator, VariantEvaluation,
    VariantGenome, VariantParameter,
};
pub use reflexive::{
    LatticeSnapshot, ReflexiveController, ReflexiveMode, ReflexiveObservation, ReflexiveState,
};
