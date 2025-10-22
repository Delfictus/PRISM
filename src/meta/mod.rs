//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod ontology;
pub mod orchestrator;
#[path = "../../PRISM-AI-UNIFIED-VAULT/src/meta/plasticity/mod.rs"]
pub mod plasticity;

pub use ontology::{ConceptAnchor, OntologyDigest, OntologyLedger};
pub use orchestrator::{
    EvolutionMetrics, EvolutionOutcome, EvolutionPlan, MetaOrchestrator, VariantEvaluation,
    VariantGenome, VariantParameter,
};
pub use plasticity::{
    explainability_report, AdaptationEvent, AdaptationMetadata, AdapterError, AdapterMode,
    ConceptManifest, DriftError, DriftEvaluation, DriftMetrics, DriftStatus, RepresentationAdapter,
    RepresentationDataset, RepresentationManifest, RepresentationSnapshot, SemanticDriftDetector,
};
