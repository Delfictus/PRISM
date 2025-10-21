//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod ontology;
pub mod orchestrator;
pub mod registry;
pub mod telemetry;

pub use ontology::{
    AlignmentEngine, AlignmentResult, ConceptAnchor, OntologyDigest, OntologyLedger,
    OntologyService, OntologyServiceError,
};
pub use orchestrator::{
    EvolutionMetrics, EvolutionOutcome, EvolutionPlan, MetaOrchestrator, VariantEvaluation,
    VariantGenome, VariantParameter,
};
pub use registry::{RegistryError, SelectionReport};
pub use telemetry::{MetaReplayContext, MetaRuntimeMetrics, MetaTelemetryWriter};
