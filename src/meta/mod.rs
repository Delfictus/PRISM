//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod federated;
pub mod ontology;
pub mod orchestrator;
#[path = "../../PRISM-AI-UNIFIED-VAULT/src/meta/plasticity/mod.rs"]
pub mod plasticity;
pub mod reflexive;
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
pub use plasticity::{
    explainability_report, AdaptationEvent, AdaptationMetadata, AdapterError, AdapterMode,
    ConceptManifest, DriftError, DriftEvaluation, DriftMetrics, DriftStatus, RepresentationAdapter,
    RepresentationDataset, RepresentationManifest, RepresentationSnapshot, SemanticDriftDetector,
};
pub use reflexive::{GovernanceMode, ReflexiveConfig, ReflexiveController, ReflexiveSnapshot};
pub use registry::{RegistryError, SelectionReport};
pub use telemetry::{MetaReplayContext, MetaRuntimeMetrics, MetaTelemetryWriter};
