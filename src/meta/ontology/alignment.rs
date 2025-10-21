use super::{ConceptAnchor, OntologyDigest, OntologyService, OntologyStorage, Result};
use crate::meta::orchestrator::{VariantGenome, VariantParameter};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantAlignment {
    pub variant_hash: String,
    pub matched_concepts: Vec<String>,
    pub uncovered_concepts: Vec<String>,
    pub coverage: f64,
    pub signals: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentSummary {
    pub digest: OntologyDigest,
    pub variant_alignments: Vec<VariantAlignment>,
    pub alignment_hash: String,
}

pub fn align_variants<S: OntologyStorage>(
    service: &OntologyService<S>,
    genomes: &[VariantGenome],
) -> Result<AlignmentSummary> {
    let snapshot = service.current_snapshot()?;
    let digest = snapshot.digest.clone();

    let mut variant_alignments = Vec::with_capacity(genomes.len());
    for genome in genomes {
        let tokens = collect_variant_tokens(genome);
        let mut matched = Vec::new();
        let mut uncovered = Vec::new();
        let mut signals = BTreeMap::new();
        let mut coverage_sum = 0.0;

        for concept in &snapshot.concepts {
            let concept_tokens = concept_tokens(concept);
            if concept_tokens.is_empty() {
                uncovered.push(concept.id.clone());
                signals.insert(concept.id.clone(), 0.0);
                continue;
            }
            let hits = concept_tokens
                .iter()
                .filter(|token| tokens.contains(*token))
                .count();
            let coverage = hits as f64 / concept_tokens.len() as f64;
            if coverage >= 0.5 {
                matched.push(concept.id.clone());
            } else {
                uncovered.push(concept.id.clone());
            }
            signals.insert(concept.id.clone(), coverage);
            coverage_sum += coverage;
        }

        let coverage = if snapshot.concepts.is_empty() {
            0.0
        } else {
            coverage_sum / snapshot.concepts.len() as f64
        };

        variant_alignments.push(VariantAlignment {
            variant_hash: genome.hash.clone(),
            matched_concepts: matched,
            uncovered_concepts: uncovered,
            coverage,
            signals,
        });
    }

    let alignment_hash = compute_alignment_hash(&digest, &variant_alignments)?;
    Ok(AlignmentSummary {
        digest,
        variant_alignments,
        alignment_hash,
    })
}

fn compute_alignment_hash(
    digest: &OntologyDigest,
    alignments: &[VariantAlignment],
) -> Result<String> {
    let mut hasher = Sha256::new();
    let digest_bytes = serde_json::to_vec(digest)?;
    hasher.update(digest_bytes);
    let alignment_bytes = serde_json::to_vec(alignments)?;
    hasher.update(alignment_bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn collect_variant_tokens(genome: &VariantGenome) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    tokens.insert(format!("variant:{}", genome.hash));
    for (name, param) in &genome.parameters {
        tokens.insert(format!("param:{}", name));
        tokens.insert(format!("param_value:{}={}", name, encode_param(param)));
        tokens.insert(format!("attr:{}={}", name, encode_param(param)));
    }
    for (name, enabled) in &genome.feature_toggles {
        if *enabled {
            tokens.insert(format!("feature:{}", name));
            tokens.insert(format!("related:{}", name));
        }
    }
    tokens
}

fn concept_tokens(concept: &ConceptAnchor) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    tokens.insert(format!("concept:{}", concept.id));
    for (key, value) in &concept.attributes {
        tokens.insert(format!("attr:{}={}", key, value));
    }
    for related in &concept.related {
        tokens.insert(format!("related:{}", related));
    }
    tokens
}

fn encode_param(param: &VariantParameter) -> String {
    match param {
        VariantParameter::Continuous { value, .. } => format!("{value:.6}"),
        VariantParameter::Discrete { value, .. } => value.to_string(),
        VariantParameter::Categorical { value, .. } => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::ontology::{ConceptAnchor, OntologyDigest, OntologyLedger, OntologySnapshot};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    #[derive(Clone)]
    struct MemoryStore {
        snapshot: Option<OntologySnapshot>,
    }

    impl OntologyStorage for MemoryStore {
        fn load(&self) -> Result<Option<OntologySnapshot>> {
            Ok(self.snapshot.clone())
        }

        fn save(&self, _: &OntologySnapshot) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn computes_alignment_hash() {
        let concepts = vec![ConceptAnchor {
            id: "chromatic_phase".to_string(),
            description: "Phase-coherent chromatic policy".to_string(),
            attributes: BTreeMap::from([
                ("domain".into(), "coloring".into()),
                ("trait".into(), "coherence".into()),
            ]),
            related: BTreeSet::from(["meta_generation".into()]),
        }];

        let digest = OntologyDigest::compute(&concepts).unwrap();
        let snapshot = OntologySnapshot {
            digest: digest.clone(),
            concepts,
        };

        let storage = MemoryStore {
            snapshot: Some(snapshot),
        };
        let ledger = OntologyLedger::new(PathBuf::from("unused"));
        let service = OntologyService::new(storage, ledger).unwrap();

        let genome = VariantGenome {
            seed: 42,
            parameters: BTreeMap::from([
                (
                    "trait".to_string(),
                    VariantParameter::Categorical {
                        value: "coherence".to_string(),
                        choices: vec!["coherence".into()],
                    },
                ),
                (
                    "domain".to_string(),
                    VariantParameter::Categorical {
                        value: "coloring".to_string(),
                        choices: vec!["coloring".into()],
                    },
                ),
                (
                    "instrumentation".to_string(),
                    VariantParameter::Categorical {
                        value: "cuda_graph".to_string(),
                        choices: vec!["cuda_graph".into()],
                    },
                ),
            ]),
            feature_toggles: BTreeMap::from([
                ("meta_generation".into(), true),
                ("free_energy".into(), true),
            ]),
            hash: "abc123".into(),
        };

        let summary = align_variants(&service, &[genome]).unwrap();
        assert_eq!(summary.digest.concept_root, digest.concept_root);
        assert_eq!(summary.variant_alignments.len(), 1);
        let alignment = &summary.variant_alignments[0];
        assert!(alignment.coverage > 0.5);
        assert!(alignment
            .matched_concepts
            .contains(&"chromatic_phase".into()));
        assert_eq!(summary.alignment_hash.len(), 64);
    }
}
