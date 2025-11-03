use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use thiserror::Error;

mod alignment;
pub use alignment::{AlignmentEngine, AlignmentResult, ConceptAlignment};

type Result<T> = std::result::Result<T, OntologyError>;

#[derive(Debug, Error)]
pub enum OntologyError {
    #[error("no concepts supplied for digest")]
    EmptyOntology,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConceptAnchor {
    pub id: String,
    pub description: String,
    pub attributes: BTreeMap<String, String>,
    pub related: BTreeSet<String>,
}

impl ConceptAnchor {
    pub fn canonical_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.description.as_bytes());
        for (k, v) in &self.attributes {
            hasher.update(k.as_bytes());
            hasher.update(v.as_bytes());
        }
        for rel in &self.related {
            hasher.update(rel.as_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDigest {
    pub version: u32,
    pub generated_at: DateTime<Utc>,
    pub concept_root: String,
    pub edge_root: String,
    pub manifest_hash: String,
}

impl OntologyDigest {
    pub fn compute(concepts: &[ConceptAnchor]) -> Result<Self> {
        if concepts.is_empty() {
            return Err(OntologyError::EmptyOntology);
        }
        let generated_at = Utc::now();
        let mut concept_hashes: Vec<String> = concepts
            .iter()
            .map(ConceptAnchor::canonical_fingerprint)
            .collect();
        concept_hashes.sort();
        let concept_root = merkle_root(&concept_hashes);

        let mut edge_hashes = Vec::new();
        for concept in concepts {
            for rel in &concept.related {
                let mut hasher = Sha256::new();
                hasher.update(concept.id.as_bytes());
                hasher.update(rel.as_bytes());
                edge_hashes.push(hex::encode(hasher.finalize()));
            }
        }
        edge_hashes.sort();
        let edge_root = merkle_root(&edge_hashes);

        let manifest = serde_json::to_string(concepts)?;
        let manifest_hash = hex::encode(Sha256::digest(manifest.as_bytes()));

        Ok(Self {
            version: 1,
            generated_at,
            concept_root,
            edge_root,
            manifest_hash,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LedgerEntry {
    digest: OntologyDigest,
    concepts: Vec<ConceptAnchor>,
}

#[derive(Debug, Clone)]
pub struct OntologyLedger {
    path: PathBuf,
}

impl OntologyLedger {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&self, concepts: Vec<ConceptAnchor>) -> Result<OntologyDigest> {
        let digest = OntologyDigest::compute(&concepts)?;
        let entry = LedgerEntry {
            digest: digest.clone(),
            concepts,
        };
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(&entry)?;
        use std::io::Write;
        writeln!(file, "{}", line)?;
        Ok(digest)
    }

    pub fn latest(&self) -> Result<Option<OntologyDigest>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&self.path)?;
        let last_line = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .last();
        if let Some(line) = last_line {
            let entry: LedgerEntry = serde_json::from_str(line)?;
            Ok(Some(entry.digest))
        } else {
            Ok(None)
        }
    }
}

fn merkle_root(nodes: &[String]) -> String {
    if nodes.is_empty() {
        return hex::encode(Sha256::digest(b"ontology-empty"));
    }
    let mut layer: Vec<[u8; 32]> = nodes
        .iter()
        .map(|node| Sha256::digest(node.as_bytes()).into())
        .collect();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity((layer.len() + 1) / 2);
        for chunk in layer.chunks(2) {
            let (left, right) = if chunk.len() == 2 {
                (chunk[0], chunk[1])
            } else {
                (chunk[0], chunk[0])
            };
            let mut hasher = Sha256::new();
            hasher.update(left);
            hasher.update(right);
            next.push(hasher.finalize().into());
        }
        layer = next;
    }
    hex::encode(layer[0])
}

#[derive(Debug, Error)]
pub enum OntologyServiceError {
    #[error("ontology ledger error: {0}")]
    Ledger(#[from] OntologyError),

    #[error("alignment error: {0}")]
    Alignment(String),

    #[error("concept '{0}' already exists in ontology")]
    DuplicateConcept(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type ServiceResult<T> = std::result::Result<T, OntologyServiceError>;

#[derive(Debug, Clone)]
pub struct OntologyService {
    ledger: OntologyLedger,
    alignment: AlignmentEngine,
    cache: Arc<RwLock<HashMap<String, ConceptAnchor>>>,
    digest: Arc<RwLock<Option<OntologyDigest>>>,
}

impl OntologyService {
    pub fn new(path: PathBuf) -> ServiceResult<Self> {
        let ledger = OntologyLedger::new(path);
        let mut service = Self {
            ledger,
            alignment: AlignmentEngine::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            digest: Arc::new(RwLock::new(None)),
        };
        if !service.ledger.path().exists() {
            service.bootstrap_defaults()?;
        }
        service.reload()?;
        Ok(service)
    }

    pub fn with_alignment(mut self, engine: AlignmentEngine) -> Self {
        self.alignment = engine;
        self
    }

    pub fn ingest(&self, concepts: Vec<ConceptAnchor>) -> ServiceResult<OntologyDigest> {
        if concepts.is_empty() {
            return Err(OntologyServiceError::Alignment(
                "empty ontology submission".into(),
            ));
        }
        let mut guard = self.cache.write().expect("ontology cache lock");
        for concept in &concepts {
            if guard.contains_key(&concept.id) {
                return Err(OntologyServiceError::DuplicateConcept(concept.id.clone()));
            }
        }
        let digest = self.ledger.append(concepts.clone())?;
        for concept in concepts {
            guard.insert(concept.id.clone(), concept);
        }
        drop(guard);
        *self.digest.write().expect("digest lock") = Some(digest.clone());
        Ok(digest)
    }

    pub fn align(&self, concept_id: &str) -> ServiceResult<AlignmentResult> {
        let cache = self.cache.read().expect("ontology cache read lock");
        let target = cache.get(concept_id).ok_or_else(|| {
            OntologyServiceError::Alignment(format!("concept '{concept_id}' not found"))
        })?;
        let corpus: Vec<_> = cache.values().cloned().collect();
        let result = self
            .alignment
            .align(target, &corpus)
            .map_err(|err| OntologyServiceError::Alignment(err.to_string()))?;
        Ok(result)
    }

    pub fn reload(&mut self) -> ServiceResult<()> {
        let mut latest_digest = None;
        if let Some(digest) = self.ledger.latest()? {
            latest_digest = Some(digest);
        }
        let mut index = HashMap::new();
        if self.ledger.path().exists() {
            let content = std::fs::read_to_string(self.ledger.path())?;
            for line in content.lines().filter(|l| !l.trim().is_empty()) {
                let entry: LedgerEntry = serde_json::from_str(line)?;
                for concept in entry.concepts {
                    index.insert(concept.id.clone(), concept);
                }
            }
        }
        *self.cache.write().expect("ontology cache lock") = index;
        *self.digest.write().expect("digest lock") = latest_digest;
        Ok(())
    }

    pub fn export(&self, path: &PathBuf) -> ServiceResult<()> {
        let snapshot = self.snapshot();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(&snapshot)?)?;
        Ok(())
    }

    pub fn digest(&self) -> Option<OntologyDigest> {
        self.digest.read().expect("digest lock").clone()
    }

    pub fn snapshot(&self) -> OntologyDigestSnapshot {
        let cache = self.cache.read().expect("cache read lock");
        OntologyDigestSnapshot {
            concepts: cache.values().cloned().collect(),
            digest: self.digest.read().expect("digest read lock").clone(),
        }
    }

    fn bootstrap_defaults(&self) -> ServiceResult<()> {
        let defaults = vec![
            ConceptAnchor {
                id: "meta_generation".into(),
                description:
                    "Deterministic variant orchestrator responsible for MEC generation cycles."
                        .into(),
                attributes: BTreeMap::from([
                    ("domain".into(), "meta".into()),
                    ("layer".into(), "orchestration".into()),
                    ("criticality".into(), "p0".into()),
                ]),
                related: BTreeSet::from(["telemetry".into(), "determinism".into()]),
            },
            ConceptAnchor {
                id: "ontology_bridge".into(),
                description:
                    "Ontology alignment service linking MEC variants to governance anchors.".into(),
                attributes: BTreeMap::from([
                    ("domain".into(), "meta".into()),
                    ("layer".into(), "ontology".into()),
                    ("criticality".into(), "p1".into()),
                ]),
                related: BTreeSet::from(["meta_generation".into(), "explainability".into()]),
            },
        ];
        self.ledger.append(defaults)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDigestSnapshot {
    pub concepts: Vec<ConceptAnchor>,
    pub digest: Option<OntologyDigest>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::ontology::alignment::AlignmentEngine;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    fn sample_concept(id: &str, related: &[&str]) -> ConceptAnchor {
        ConceptAnchor {
            id: id.into(),
            description: format!("Concept {id}"),
            attributes: BTreeMap::from([
                ("domain".into(), "meta".into()),
                ("type".into(), "feature".into()),
            ]),
            related: related.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn ontology_digest_stable() {
        let concept = ConceptAnchor {
            id: "chromatic_phase".into(),
            description: "Phase-coherent chromatic policy".into(),
            attributes: BTreeMap::from([
                ("domain".into(), "coloring".into()),
                ("trait".into(), "coherence".into()),
            ]),
            related: BTreeSet::from(["meta_generation".into()]),
        };
        let digest = OntologyDigest::compute(&[concept.clone()]).unwrap();
        let digest_again = OntologyDigest::compute(&[concept]).unwrap();
        assert_eq!(digest.concept_root, digest_again.concept_root);
    }

    #[test]
    fn service_ingest_and_align() {
        let dir = tempdir().expect("tempdir");
        let ledger_path = dir.path().join("ontology_ledger.jsonl");
        let service = OntologyService::new(ledger_path).expect("service");

        let digest = service
            .ingest(vec![
                sample_concept("reflexive_controller", &["free_energy", "meta_generation"]),
                sample_concept("federated_meta", &["federation", "meta_generation"]),
            ])
            .expect("ingest");
        assert_eq!(digest.version, 1);

        let alignment = service.align("federated_meta").expect("align");
        assert_eq!(alignment.target, "federated_meta");
        assert!(alignment.primary_match.is_some());
        assert!(!alignment.candidates.is_empty());
    }
}
