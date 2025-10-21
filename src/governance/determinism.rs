use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;

const COMPONENTS: &[&str] = &[
    "ensemble_generation",
    "thermodynamic",
    "quantum_pimc",
    "neuromorphic",
    "cma",
    "info_geom",
    "gpu_coloring",
];

/// Captures deterministic execution metadata for replay validation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeterminismProof {
    pub master_seed: u64,
    pub component_seeds: HashMap<String, u64>,
    pub input_hash: String,
    pub output_hash: String,
    pub intermediate_hashes: Vec<IntermediateHash>,
    pub timestamp: DateTime<Utc>,
    pub environment: EnvironmentFingerprint,
    pub meta: Option<MetaDeterminism>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntermediateHash {
    pub stage: String,
    pub hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentFingerprint {
    pub cuda_version: Option<String>,
    pub gpu_model: Option<String>,
    pub driver_version: Option<String>,
    pub rust_version: Option<String>,
    pub feature_flags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaDeterminism {
    pub meta_genome_hash: String,
    pub meta_merkle_root: String,
    pub ontology_hash: Option<String>,
    pub free_energy_hash: Option<String>,
    pub reflexive_mode: Option<String>,
    pub lattice_fingerprint: Option<String>,
}

impl EnvironmentFingerprint {
    pub fn capture() -> Self {
        Self {
            cuda_version: std::env::var("CUDA_VERSION").ok(),
            gpu_model: std::env::var("GPU_MODEL").ok(),
            driver_version: std::env::var("NVIDIA_DRIVER_VERSION").ok(),
            rust_version: capture_rust_version(),
            feature_flags: collect_feature_flags(),
        }
    }
}

fn capture_rust_version() -> Option<String> {
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    option_env!("RUST_VERSION").map(|s| s.to_string())
}

fn collect_feature_flags() -> Vec<String> {
    std::env::var("PRISM_FEATURE_FLAGS")
        .map(|flags| {
            flags
                .split(',')
                .map(|flag| flag.trim().to_string())
                .filter(|flag| !flag.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

impl DeterminismProof {
    pub fn new(master_seed: u64) -> Self {
        let mut proof = Self {
            master_seed,
            component_seeds: HashMap::new(),
            input_hash: String::new(),
            output_hash: String::new(),
            intermediate_hashes: Vec::new(),
            timestamp: Utc::now(),
            environment: EnvironmentFingerprint::capture(),
            meta: None,
        };
        proof.derive_component_seeds();
        proof
    }

    fn derive_component_seeds(&mut self) {
        let mut hasher = Sha256::new();
        hasher.update(self.master_seed.to_le_bytes());

        for (index, component) in COMPONENTS.iter().enumerate() {
            hasher.update((index as u64).to_le_bytes());
            let digest = hasher.finalize_reset();
            let seed = u64::from_le_bytes(digest[0..8].try_into().expect("hash slice"));
            self.component_seeds.insert(component.to_string(), seed);
        }
    }

    pub fn record_input<T: Serialize>(&mut self, value: &T) -> Result<()> {
        self.input_hash = compute_hash(value)?;
        Ok(())
    }

    pub fn record_output<T: Serialize>(&mut self, value: &T) -> Result<()> {
        self.output_hash = compute_hash(value)?;
        Ok(())
    }

    pub fn attach_meta(&mut self, meta: MetaDeterminism) {
        self.meta = Some(meta);
    }

    pub fn record_intermediate<T: Serialize>(
        &mut self,
        stage: impl Into<String>,
        value: &T,
    ) -> Result<()> {
        let stage = stage.into();
        let hash = compute_hash(value)?;
        self.intermediate_hashes
            .push(IntermediateHash { stage, hash });
        Ok(())
    }

    pub fn verify_replay(&self, other: &Self) -> Result<()> {
        if self.master_seed != other.master_seed {
            return Err(anyhow!(
                "Master seed mismatch: {} != {}",
                self.master_seed,
                other.master_seed
            ));
        }
        if self.input_hash != other.input_hash {
            return Err(anyhow!(
                "Input hash mismatch: {} != {}",
                self.input_hash,
                other.input_hash
            ));
        }
        if self.component_seeds != other.component_seeds {
            return Err(anyhow!("Component seed map mismatch"));
        }
        if self.intermediate_hashes.len() != other.intermediate_hashes.len() {
            return Err(anyhow!(
                "Intermediate hash length mismatch: {} vs {}",
                self.intermediate_hashes.len(),
                other.intermediate_hashes.len()
            ));
        }

        for (idx, (lhs, rhs)) in self
            .intermediate_hashes
            .iter()
            .zip(other.intermediate_hashes.iter())
            .enumerate()
        {
            if lhs.stage != rhs.stage || lhs.hash != rhs.hash {
                return Err(anyhow!(
                    "Intermediate divergence at {} ({} -> {} vs {})",
                    idx,
                    lhs.stage,
                    lhs.hash,
                    rhs.hash
                ));
            }
        }

        if self.output_hash != other.output_hash {
            return Err(anyhow!(
                "Output hash mismatch: {} != {}",
                self.output_hash,
                other.output_hash
            ));
        }

        Ok(())
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_json_string()?)?;
        Ok(())
    }
}

/// Helper to incrementally record determinism artefacts.
pub struct DeterminismRecorder {
    proof: DeterminismProof,
}

impl DeterminismRecorder {
    pub fn new(master_seed: u64) -> Self {
        Self {
            proof: DeterminismProof::new(master_seed),
        }
    }

    pub fn record_input<T: Serialize>(&mut self, value: &T) -> Result<()> {
        self.proof.record_input(value)
    }

    pub fn record_intermediate<T: Serialize>(
        &mut self,
        stage: impl Into<String>,
        value: &T,
    ) -> Result<()> {
        self.proof.record_intermediate(stage, value)
    }

    pub fn record_output<T: Serialize>(&mut self, value: &T) -> Result<()> {
        self.proof.record_output(value)
    }

    pub fn attach_meta(&mut self, meta: MetaDeterminism) {
        self.proof.meta = Some(meta);
    }

    pub fn finalize(self) -> DeterminismProof {
        self.proof
    }
}

/// Simplified audit trail that produces a Merkle-like root for stored proofs.
#[derive(Default)]
pub struct DeterminismAuditTrail {
    proofs: Vec<DeterminismProof>,
}

impl DeterminismAuditTrail {
    pub fn new() -> Self {
        Self { proofs: Vec::new() }
    }

    pub fn append(&mut self, proof: DeterminismProof) -> String {
        self.proofs.push(proof);
        self.root()
    }

    pub fn proofs(&self) -> &[DeterminismProof] {
        &self.proofs
    }

    pub fn root(&self) -> String {
        let mut leaves: Vec<[u8; 32]> = self.proofs.iter().map(hash_proof).collect();
        if leaves.is_empty() {
            return String::new();
        }

        while leaves.len() > 1 {
            let mut next = Vec::with_capacity((leaves.len() + 1) / 2);
            for chunk in leaves.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(chunk[0]);
                if chunk.len() == 2 {
                    hasher.update(chunk[1]);
                } else {
                    hasher.update(chunk[0]);
                }
                let digest = hasher.finalize();
                next.push(digest.try_into().expect("digest len"));
            }
            leaves = next;
        }

        hex::encode(leaves[0])
    }

    pub fn verify_inclusion(&self, proof: &DeterminismProof) -> bool {
        self.proofs
            .iter()
            .any(|existing| hash_proof(existing) == hash_proof(proof))
    }
}

fn compute_hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn hash_proof(proof: &DeterminismProof) -> [u8; 32] {
    let bytes = bincode::serialize(proof).expect("serialize proof");
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_roundtrip_and_verification() {
        let mut recorder_a = DeterminismRecorder::new(42);
        recorder_a.record_input(&vec![1u32, 2, 3]).unwrap();
        recorder_a.record_intermediate("stage1", &"abc").unwrap();
        recorder_a.record_output(&vec![0u32, 1, 2]).unwrap();
        let proof_a = recorder_a.finalize();

        let mut recorder_b = DeterminismRecorder::new(42);
        recorder_b.record_input(&vec![1u32, 2, 3]).unwrap();
        recorder_b.record_intermediate("stage1", &"abc").unwrap();
        recorder_b.record_output(&vec![0u32, 1, 2]).unwrap();
        let proof_b = recorder_b.finalize();

        proof_a.verify_replay(&proof_b).unwrap();
        assert_eq!(proof_a.input_hash, proof_b.input_hash);

        let json = proof_a.to_json_string().unwrap();
        assert!(json.contains("master_seed"));
    }

    #[test]
    fn audit_trail_root_changes_with_entries() {
        let mut trail = DeterminismAuditTrail::new();
        assert_eq!(trail.root(), "");

        let mut recorder = DeterminismRecorder::new(1);
        recorder.record_input(&vec![1u32]).unwrap();
        recorder.record_output(&vec![2u32]).unwrap();
        let root1 = trail.append(recorder.finalize());

        let mut recorder = DeterminismRecorder::new(2);
        recorder.record_input(&vec![3u32]).unwrap();
        recorder.record_output(&vec![4u32]).unwrap();
        let root2 = trail.append(recorder.finalize());

        assert_ne!(root1, root2);
        assert!(!root2.is_empty());
    }
}
