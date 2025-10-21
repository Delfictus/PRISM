pub mod alignment;
pub use alignment::{align_variants, AlignmentSummary, VariantAlignment};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use thiserror::Error;

type Result<T> = std::result::Result<T, OntologyError>;

#[derive(Debug, Error)]
pub enum OntologyError {
    #[error("no concepts supplied for digest")]
    EmptyOntology,

    #[error("ontology snapshot not initialized")]
    MissingSnapshot,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologySnapshot {
    pub digest: OntologyDigest,
    pub concepts: Vec<ConceptAnchor>,
}

impl OntologySnapshot {
    pub fn from_concepts(concepts: Vec<ConceptAnchor>) -> Result<Self> {
        let digest = OntologyDigest::compute(&concepts)?;
        Ok(Self { digest, concepts })
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

pub trait OntologyStorage: Send + Sync + 'static {
    fn load(&self) -> Result<Option<OntologySnapshot>>;
    fn save(&self, snapshot: &OntologySnapshot) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct FileOntologyStorage {
    snapshot_path: PathBuf,
}

impl FileOntologyStorage {
    pub fn new(snapshot_path: PathBuf) -> Self {
        Self { snapshot_path }
    }
}

impl OntologyStorage for FileOntologyStorage {
    fn load(&self) -> Result<Option<OntologySnapshot>> {
        if !self.snapshot_path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&self.snapshot_path)?;
        if content.trim().is_empty() {
            return Ok(None);
        }
        let snapshot: OntologySnapshot = serde_json::from_str(&content)?;
        Ok(Some(snapshot))
    }

    fn save(&self, snapshot: &OntologySnapshot) -> Result<()> {
        if let Some(parent) = self.snapshot_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(snapshot)?;
        std::fs::write(&self.snapshot_path, json)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct OntologyService<S: OntologyStorage> {
    store: S,
    ledger: OntologyLedger,
    cache: Arc<RwLock<Option<OntologySnapshot>>>,
}

impl<S: OntologyStorage> OntologyService<S> {
    pub fn new(store: S, ledger: OntologyLedger) -> Result<Self> {
        let snapshot = store.load()?;
        Ok(Self {
            store,
            ledger,
            cache: Arc::new(RwLock::new(snapshot)),
        })
    }

    pub fn current_snapshot(&self) -> Result<OntologySnapshot> {
        if let Some(snapshot) = self.cache.read().unwrap().clone() {
            return Ok(snapshot);
        }
        let snapshot = self.store.load()?.ok_or(OntologyError::MissingSnapshot)?;
        *self.cache.write().unwrap() = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub fn refresh(&self) -> Result<Option<OntologySnapshot>> {
        let snapshot = self.store.load()?;
        *self.cache.write().unwrap() = snapshot.clone();
        Ok(snapshot)
    }

    pub fn ingest(&self, new_concepts: Vec<ConceptAnchor>) -> Result<OntologySnapshot> {
        let mut map: BTreeMap<String, ConceptAnchor> = self
            .cache
            .read()
            .unwrap()
            .as_ref()
            .map(|snapshot| {
                snapshot
                    .concepts
                    .iter()
                    .map(|concept| (concept.id.clone(), concept.clone()))
                    .collect()
            })
            .unwrap_or_default();

        for concept in new_concepts {
            map.insert(concept.id.clone(), concept);
        }

        let mut concepts: Vec<ConceptAnchor> = map.into_values().collect();
        concepts.sort_by(|a, b| a.id.cmp(&b.id));
        let snapshot = OntologySnapshot::from_concepts(concepts)?;
        self.store.save(&snapshot)?;
        self.ledger.append(snapshot.concepts.clone())?;
        *self.cache.write().unwrap() = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub fn find_concept(&self, id: &str) -> Result<Option<ConceptAnchor>> {
        let snapshot = self.current_snapshot()?;
        Ok(snapshot
            .concepts
            .into_iter()
            .find(|concept| concept.id == id))
    }

    pub fn list_concepts(&self) -> Result<Vec<ConceptAnchor>> {
        let snapshot = self.current_snapshot()?;
        Ok(snapshot.concepts)
    }

    pub fn latest_digest(&self) -> Result<OntologyDigest> {
        self.current_snapshot().map(|snapshot| snapshot.digest)
    }

    pub fn export_snapshot(&self, path: impl AsRef<Path>) -> Result<()> {
        let snapshot = self.current_snapshot()?;
        let json = serde_json::to_string_pretty(&snapshot)?;
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl OntologyService<FileOntologyStorage> {
    pub fn with_file_paths(
        snapshot_path: PathBuf,
        ledger_path: PathBuf,
    ) -> Result<OntologyService<FileOntologyStorage>> {
        Self::new(
            FileOntologyStorage::new(snapshot_path),
            OntologyLedger::new(ledger_path),
        )
    }
}

#[derive(Clone)]
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct MemoryStore {
        snapshot: Arc<RwLock<Option<OntologySnapshot>>>,
    }

    impl OntologyStorage for MemoryStore {
        fn load(&self) -> Result<Option<OntologySnapshot>> {
            Ok(self.snapshot.read().unwrap().clone())
        }

        fn save(&self, snapshot: &OntologySnapshot) -> Result<()> {
            *self.snapshot.write().unwrap() = Some(snapshot.clone());
            Ok(())
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
    fn service_ingest_merges_concepts() {
        let store = MemoryStore::default();
        let ledger_path = std::env::temp_dir().join(format!(
            "ontology-ledger-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let service = OntologyService::new(store.clone(), OntologyLedger::new(ledger_path.clone()))
            .expect("create service");

        let concepts = vec![ConceptAnchor {
            id: "chromatic_phase".into(),
            description: "Chromatic phase alignment".into(),
            attributes: BTreeMap::from([("domain".into(), "coloring".into())]),
            related: BTreeSet::new(),
        }];

        let snapshot = service.ingest(concepts).expect("ingest concepts");
        assert_eq!(snapshot.concepts.len(), 1);
        assert!(snapshot
            .concepts
            .iter()
            .any(|concept| concept.id == "chromatic_phase"));

        // Merge new concept and ensure both exist.
        let snapshot = service
            .ingest(vec![ConceptAnchor {
                id: "meta_alignment".into(),
                description: "Meta alignment handshake".into(),
                attributes: BTreeMap::from([("domain".into(), "meta".into())]),
                related: BTreeSet::new(),
            }])
            .expect("merge concepts");
        assert_eq!(snapshot.concepts.len(), 2);
        assert!(snapshot
            .concepts
            .iter()
            .any(|concept| concept.id == "meta_alignment"));

        // Clean up ledger file created during the test.
        if ledger_path.exists() {
            let _ = std::fs::remove_file(&ledger_path);
        }
    }
}
