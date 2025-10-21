use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use thiserror::Error;

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

pub struct OntologyLedger {
    path: PathBuf,
}

impl OntologyLedger {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
