use crate::telemetry::{ComponentId, EventData, EventLevel, TelemetryEntry, TelemetrySink};
use chrono::{DateTime, SecondsFormat, Utc};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use thiserror::Error;

const META_FLAG_STORAGE: &str = "PRISM-AI-UNIFIED-VAULT/meta/meta_flags.json";
const META_LEDGER_DIR: &str = "PRISM-AI-UNIFIED-VAULT/meta/merkle";
const META_ARTIFACT_DIR: &str = "artifacts/merkle";
const MANIFEST_VERSION: u32 = 1;

lazy_static! {
    static ref GLOBAL_REGISTRY: MetaFeatureRegistry =
        MetaFeatureRegistry::load_default().expect("meta feature registry");
}

/// Access the process-wide registry instance.
pub fn registry() -> &'static MetaFeatureRegistry {
    &GLOBAL_REGISTRY
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaFeatureId {
    MetaGeneration,
    OntologyBridge,
    FreeEnergySnapshots,
    SemanticPlasticity,
    FederatedMeta,
    MetaProd,
}

impl Display for MetaFeatureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl MetaFeatureId {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetaFeatureId::MetaGeneration => "meta_generation",
            MetaFeatureId::OntologyBridge => "ontology_bridge",
            MetaFeatureId::FreeEnergySnapshots => "free_energy_snapshots",
            MetaFeatureId::SemanticPlasticity => "semantic_plasticity",
            MetaFeatureId::FederatedMeta => "federated_meta",
            MetaFeatureId::MetaProd => "meta_prod",
        }
    }

    pub fn variants() -> &'static [&'static str] {
        &[
            "meta_generation",
            "ontology_bridge",
            "free_energy_snapshots",
            "semantic_plasticity",
            "federated_meta",
            "meta_prod",
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MetaFeatureState {
    Disabled,
    Shadow {
        actor: String,
        planned_activation: Option<DateTime<Utc>>,
    },
    Gradual {
        actor: String,
        current_pct: f32,
        target_pct: f32,
        eta: Option<DateTime<Utc>>,
    },
    Enabled {
        actor: String,
        activated_at: DateTime<Utc>,
        justification: String,
    },
}

impl MetaFeatureState {
    fn promotion_index(&self) -> u8 {
        match self {
            MetaFeatureState::Disabled => 0,
            MetaFeatureState::Shadow { .. } => 1,
            MetaFeatureState::Gradual { .. } => 2,
            MetaFeatureState::Enabled { .. } => 3,
        }
    }

    fn actor(&self) -> Option<&str> {
        match self {
            MetaFeatureState::Disabled => None,
            MetaFeatureState::Shadow { actor, .. } => Some(actor.as_str()),
            MetaFeatureState::Gradual { actor, .. } => Some(actor.as_str()),
            MetaFeatureState::Enabled { actor, .. } => Some(actor.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaFlagRecord {
    pub id: MetaFeatureId,
    pub state: MetaFeatureState,
    pub updated_at: DateTime<Utc>,
    pub updated_by: String,
    pub rationale: String,
    pub evidence_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaFlagManifest {
    pub version: u32,
    pub generated_at: DateTime<Utc>,
    pub merkle_root: String,
    pub records: Vec<MetaFlagRecord>,
    pub invariant_snapshot: ManifestInvariants,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestInvariants {
    pub enforced: Vec<MetaFeatureId>,
    pub generation_entropy: String,
}

#[derive(Debug, Error)]
pub enum MetaFlagError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("invalid feature transition {feature}: {reason}")]
    InvalidTransition {
        feature: MetaFeatureId,
        reason: String,
    },

    #[error("feature {0} is not enabled")]
    NotEnabled(MetaFeatureId),

    #[error("unknown meta feature '{0}'")]
    UnknownFeature(String),
}

impl FromStr for MetaFeatureId {
    type Err = MetaFlagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "meta_generation" | "meta-generation" => Ok(MetaFeatureId::MetaGeneration),
            "ontology_bridge" | "ontology-bridge" => Ok(MetaFeatureId::OntologyBridge),
            "free_energy_snapshots" | "free-energy-snapshots" => {
                Ok(MetaFeatureId::FreeEnergySnapshots)
            }
            "semantic_plasticity" | "semantic-plasticity" => Ok(MetaFeatureId::SemanticPlasticity),
            "federated_meta" | "federated-meta" => Ok(MetaFeatureId::FederatedMeta),
            "meta_prod" | "meta-prod" => Ok(MetaFeatureId::MetaProd),
            other => Err(MetaFlagError::UnknownFeature(other.to_string())),
        }
    }
}

#[derive(Clone)]
pub struct MetaFeatureRegistry {
    storage_path: PathBuf,
    ledger_dir: PathBuf,
    artifacts_dir: PathBuf,
    records: Arc<RwLock<BTreeMap<MetaFeatureId, MetaFlagRecord>>>,
    telemetry: TelemetrySink,
}

impl MetaFeatureRegistry {
    pub fn load_default() -> Result<Self, MetaFlagError> {
        Self::new_with_paths(
            PathBuf::from(META_FLAG_STORAGE),
            PathBuf::from(META_LEDGER_DIR),
            PathBuf::from(META_ARTIFACT_DIR),
        )
    }

    fn new_with_paths(
        storage_path: PathBuf,
        ledger_dir: PathBuf,
        artifacts_dir: PathBuf,
    ) -> Result<Self, MetaFlagError> {
        fs::create_dir_all(ledger_dir.as_path())?;
        fs::create_dir_all(artifacts_dir.as_path())?;

        let telemetry = TelemetrySink::new("meta_feature_registry");
        let mut registry = Self {
            storage_path,
            ledger_dir,
            artifacts_dir,
            records: Arc::new(RwLock::new(BTreeMap::new())),
            telemetry,
        };
        if registry.storage_path.exists() {
            registry.load_from_disk()?;
        } else {
            registry.bootstrap_defaults()?;
        }
        Ok(registry)
    }

    #[cfg(test)]
    pub(crate) fn test_with_paths(
        storage_path: PathBuf,
        ledger_dir: PathBuf,
        artifacts_dir: PathBuf,
    ) -> Result<Self, MetaFlagError> {
        Self::new_with_paths(storage_path, ledger_dir, artifacts_dir)
    }

    fn load_from_disk(&mut self) -> Result<(), MetaFlagError> {
        let file = fs::File::open(&self.storage_path)?;
        let manifest: MetaFlagManifest = serde_json::from_reader(file)?;
        let expected_root = compute_merkle_root(&manifest.records);
        if manifest.merkle_root != expected_root {
            return Err(MetaFlagError::InvalidTransition {
                feature: MetaFeatureId::MetaGeneration,
                reason: format!(
                    "Merkle root mismatch: expected {}, found {}",
                    expected_root, manifest.merkle_root
                ),
            });
        }

        let mut guard = self.records.write().expect("records write lock");
        guard.clear();
        for record in manifest.records {
            guard.insert(record.id, record);
        }
        Ok(())
    }

    fn bootstrap_defaults(&mut self) -> Result<(), MetaFlagError> {
        let mut guard = self.records.write().expect("records write lock");
        guard.clear();
        let defaults = [
            MetaFeatureId::MetaGeneration,
            MetaFeatureId::OntologyBridge,
            MetaFeatureId::FreeEnergySnapshots,
            MetaFeatureId::SemanticPlasticity,
            MetaFeatureId::FederatedMeta,
            MetaFeatureId::MetaProd,
        ];
        let now = Utc::now();
        for feature in defaults {
            guard.insert(
                feature,
                MetaFlagRecord {
                    id: feature,
                    state: MetaFeatureState::Disabled,
                    updated_at: now,
                    updated_by: "bootstrap".into(),
                    rationale: "Initial baseline".into(),
                    evidence_uri: None,
                },
            );
        }
        drop(guard);
        self.persist("bootstrap", "Initialized registry", None)?;
        Ok(())
    }

    pub fn snapshot(&self) -> MetaFlagManifest {
        let guard = self.records.read().expect("records read lock");
        let records: Vec<MetaFlagRecord> = guard.values().cloned().collect();
        let merkle_root = compute_merkle_root(&records);
        MetaFlagManifest {
            version: MANIFEST_VERSION,
            generated_at: Utc::now(),
            merkle_root,
            invariant_snapshot: self.compute_invariants(&records),
            records,
        }
    }

    pub fn is_enabled(&self, feature: MetaFeatureId) -> bool {
        let guard = self.records.read().expect("records read lock");
        guard
            .get(&feature)
            .map(|record| matches!(record.state, MetaFeatureState::Enabled { .. }))
            .unwrap_or(false)
    }

    pub fn require_enabled(&self, feature: MetaFeatureId) -> Result<(), MetaFlagError> {
        if self.is_enabled(feature) {
            Ok(())
        } else {
            Err(MetaFlagError::NotEnabled(feature))
        }
    }

    pub fn update_state(
        &self,
        feature: MetaFeatureId,
        state: MetaFeatureState,
        updated_by: impl Into<String>,
        rationale: impl Into<String>,
        evidence_uri: Option<String>,
    ) -> Result<MetaFlagManifest, MetaFlagError> {
        let actor = updated_by.into();
        if let Some(name) = state.actor() {
            if name.trim().is_empty() {
                return Err(MetaFlagError::InvalidTransition {
                    feature,
                    reason: "actor must be non-empty".into(),
                });
            }
        }

        let mut guard = self.records.write().expect("records write lock");
        let current = guard.get(&feature);
        if let Some(existing) = current {
            let existing_order = existing.state.promotion_index();
            let new_order = state.promotion_index();
            if new_order < existing_order {
                return Err(MetaFlagError::InvalidTransition {
                    feature,
                    reason: format!(
                        "non-monotonic transition from {:?} to {:?}",
                        existing.state, state
                    ),
                });
            }
        }

        let updated_at = Utc::now();
        let record = MetaFlagRecord {
            id: feature,
            state: state.clone(),
            updated_at,
            updated_by: actor.clone(),
            rationale: rationale.into(),
            evidence_uri,
        };
        guard.insert(feature, record);
        drop(guard);

        let manifest = self.persist(&actor, "state change", None)?;
        self.emit_telemetry(feature, state, updated_at, &actor);
        Ok(manifest)
    }

    fn persist(
        &self,
        actor: &str,
        action: &str,
        audit_note: Option<&str>,
    ) -> Result<MetaFlagManifest, MetaFlagError> {
        let manifest = self.snapshot();
        let serialized = serde_json::to_vec_pretty(&manifest)?;
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.storage_path, &serialized)?;

        let ts = manifest
            .generated_at
            .to_rfc3339_opts(SecondsFormat::Millis, true)
            .replace(':', "-");
        let ledger_name = format!("meta_flags_{}_{}.json", ts, manifest.merkle_root);
        let ledger_path = self.ledger_dir.join(ledger_name);
        fs::write(ledger_path, &serialized)?;

        let audit_name = format!("meta_flags_{}_{}.json", ts, manifest.merkle_root);
        let audit_path = self.artifacts_dir.join(audit_name);
        fs::write(audit_path, &serialized)?;

        if let Some(note) = audit_note {
            let mut log_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("artifacts/audit/meta_flags.log")?;
            writeln!(
                log_file,
                "{} | {} | {} | {}",
                manifest.generated_at, actor, action, note
            )?;
        }

        Ok(manifest)
    }

    fn emit_telemetry(
        &self,
        feature: MetaFeatureId,
        state: MetaFeatureState,
        updated_at: DateTime<Utc>,
        actor: &str,
    ) {
        let payload = serde_json::json!({
            "feature": feature.to_string(),
            "state": state,
            "updated_at": updated_at.to_rfc3339(),
            "actor": actor,
        });
        let entry = TelemetryEntry::new(
            ComponentId::Custom("meta_feature_registry".into()),
            EventLevel::Info,
            EventData::Custom { payload },
            None,
            None,
        );
        self.telemetry.log(&entry);
    }

    fn compute_invariants(&self, records: &[MetaFlagRecord]) -> ManifestInvariants {
        let enforced = records
            .iter()
            .filter(|record| matches!(record.state, MetaFeatureState::Enabled { .. }))
            .map(|record| record.id)
            .collect();
        let entropy = {
            let mut hasher = Sha256::new();
            hasher.update(b"META_INVARIANT_SALT");
            for record in records {
                hasher.update(record.id.to_string().as_bytes());
                hasher.update(record.updated_at.timestamp_micros().to_be_bytes());
            }
            hex::encode(hasher.finalize())
        };
        ManifestInvariants {
            enforced,
            generation_entropy: entropy,
        }
    }
}

fn compute_merkle_root(records: &[MetaFlagRecord]) -> String {
    if records.is_empty() {
        return hex::encode(Sha256::digest(b"meta-empty"));
    }

    let mut leaves: Vec<[u8; 32]> = records
        .iter()
        .map(|record| {
            let serialized =
                serde_json::to_vec(record).expect("record serialization for merkle tree");
            let mut hasher = Sha256::new();
            hasher.update(serialized);
            hasher.finalize().into()
        })
        .collect();
    leaves.sort();

    while leaves.len() > 1 {
        let mut next_level = Vec::with_capacity((leaves.len() + 1) / 2);
        for chunk in leaves.chunks(2) {
            let combined = if chunk.len() == 2 {
                [chunk[0], chunk[1]]
            } else {
                [chunk[0], chunk[0]]
            };
            let mut hasher = Sha256::new();
            hasher.update(combined[0]);
            hasher.update(combined[1]);
            next_level.push(hasher.finalize().into());
        }
        leaves = next_level;
    }

    hex::encode(leaves[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn merkle_root_is_deterministic() {
        let now = Utc::now();
        let records = vec![
            MetaFlagRecord {
                id: MetaFeatureId::MetaGeneration,
                state: MetaFeatureState::Disabled,
                updated_at: now,
                updated_by: "a".into(),
                rationale: "r".into(),
                evidence_uri: None,
            },
            MetaFlagRecord {
                id: MetaFeatureId::OntologyBridge,
                state: MetaFeatureState::Disabled,
                updated_at: now,
                updated_by: "a".into(),
                rationale: "r".into(),
                evidence_uri: None,
            },
        ];
        let a = compute_merkle_root(&records);
        let b = compute_merkle_root(&records);
        assert_eq!(a, b);
    }

    #[test]
    fn reject_downgrade_transition() {
        let dir = tempdir().unwrap();
        let registry = MetaFeatureRegistry::test_with_paths(
            dir.path().join("meta_flags.json"),
            dir.path().join("merkle"),
            dir.path().join("artifacts"),
        )
        .unwrap();
        registry
            .update_state(
                MetaFeatureId::MetaGeneration,
                MetaFeatureState::Shadow {
                    actor: "tester".into(),
                    planned_activation: None,
                },
                "tester",
                "shadow deploy",
                None,
            )
            .unwrap();
        let result = registry.update_state(
            MetaFeatureId::MetaGeneration,
            MetaFeatureState::Disabled,
            "tester",
            "attempt downgrade",
            None,
        );
        assert!(matches!(
            result,
            Err(MetaFlagError::InvalidTransition { .. })
        ));
    }
}
