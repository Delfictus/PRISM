//! Meta Evolutionary Compute (MEC) scaffolding.

pub mod ontology;
pub mod orchestrator;

pub use ontology::{ConceptAnchor, OntologyDigest, OntologyLedger};
pub use orchestrator::{EvolutionPlan, MetaOrchestrator, VariantGenome, VariantParameter};
