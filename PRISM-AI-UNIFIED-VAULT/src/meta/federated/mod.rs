//! Federated readiness module (Phase M5).
//!
//! This module encapsulates the core protocol structures required to run meta
//! orchestration across multiple federation nodes. It includes data types for
//! handshake metadata, per-node updates, consensus aggregation, and a simple
//! simulation harness used by the compliance tooling.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Identifier for a federation node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeFingerprint(pub String);

impl fmt::Display for NodeFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Configuration used during federation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FederationConfig {
    pub policy_hash: String,
    pub epoch: u64,
    pub required_quorum: usize,
}

impl FederationConfig {
    pub fn new(policy_hash: impl Into<String>, epoch: u64, required_quorum: usize) -> Self {
        Self {
            policy_hash: policy_hash.into(),
            epoch,
            required_quorum,
        }
    }
}

/// Describes a federation node's capabilities and stake.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FederatedNode {
    pub fingerprint: NodeFingerprint,
    pub software_version: String,
    pub capabilities: BTreeSet<String>,
    pub stake_weight: f32,
}

impl FederatedNode {
    pub fn new(
        fingerprint: impl Into<String>,
        software_version: impl Into<String>,
        capabilities: impl IntoIterator<Item = impl Into<String>>,
        stake_weight: f32,
    ) -> Self {
        Self {
            fingerprint: NodeFingerprint(fingerprint.into()),
            software_version: software_version.into(),
            capabilities: capabilities
                .into_iter()
                .map(|cap| cap.into())
                .collect::<BTreeSet<String>>(),
            stake_weight,
        }
    }
}

/// Signed update emitted by a federation node at the end of a local MEC cycle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FederatedUpdate {
    pub node: NodeFingerprint,
    pub epoch: u64,
    pub update_hash: String,
    pub ledger_block_id: String,
    pub payload_hash: String,
    pub stake_weight: f32,
}

/// Outcome of consensus evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusDecision {
    pub passed: bool,
    pub votes_for: usize,
    pub votes_against: usize,
    pub quorum: usize,
    pub aggregated_hash: String,
}

/// Alignment proof for a node after reconciliation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignmentProof {
    pub node: NodeFingerprint,
    pub update_hash: String,
    pub aligned_hash: String,
}

/// Summary manifest for a federated epoch (persisted under artifacts/mec/M5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FederatedPlan {
    pub policy_hash: String,
    pub epoch: u64,
    pub timestamp: String,
    pub participants: Vec<FederatedNode>,
    pub consensus: ConsensusDecision,
    pub alignments: Vec<AlignmentProof>,
}

/// Federated simulation harness.
pub struct FederationSimulator {
    config: FederationConfig,
    nodes: Vec<FederatedNode>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FederationError {
    #[error("unknown node fingerprint: {0}")]
    UnknownNode(String),

    #[error("update epoch mismatch (got {found}, expected {expected})")]
    EpochMismatch { found: u64, expected: u64 },

    #[error("duplicate update for node {0}")]
    DuplicateUpdate(String),
}

impl FederationSimulator {
    pub fn new(config: FederationConfig, nodes: Vec<FederatedNode>) -> Self {
        Self { config, nodes }
    }

    pub fn nodes(&self) -> &[FederatedNode] {
        &self.nodes
    }

    /// Simulate an epoch given a set of node updates.
    pub fn simulate_epoch(
        &self,
        updates: &[FederatedUpdate],
        timestamp: impl Into<String>,
    ) -> Result<FederatedPlan, FederationError> {
        let mut seen = BTreeSet::new();
        let mut alignments = Vec::new();
        let mut votes_for = 0usize;
        let mut weighted_hash_inputs = Vec::new();
        let participants = BTreeMap::from_iter(
            self.nodes
                .iter()
                .map(|node| (node.fingerprint.clone(), node.clone())),
        );

        for update in updates {
            if !participants.contains_key(&update.node) {
                return Err(FederationError::UnknownNode(update.node.0.clone()));
            }

            if update.epoch != self.config.epoch {
                return Err(FederationError::EpochMismatch {
                    found: update.epoch,
                    expected: self.config.epoch,
                });
            }

            if !seen.insert(update.node.clone()) {
                return Err(FederationError::DuplicateUpdate(update.node.0.clone()));
            }

            // Align hash by combining update hash with ledger block and payload.
            let aligned_hash = align_update_hash(
                &update.update_hash,
                &update.ledger_block_id,
                &update.payload_hash,
            );
            alignments.push(AlignmentProof {
                node: update.node.clone(),
                update_hash: update.update_hash.clone(),
                aligned_hash: aligned_hash.clone(),
            });

            // Weighted votes (for this simulation all updates are considered valid).
            weighted_hash_inputs.push((aligned_hash, update.stake_weight));
            votes_for += 1;

            // stake weight already encoded in updates for aggregation
        }

        let votes_against = self.nodes.len().saturating_sub(votes_for);
        let aggregated_hash = aggregate_hash(&weighted_hash_inputs);
        let consensus = ConsensusDecision {
            passed: votes_for >= self.config.required_quorum,
            votes_for,
            votes_against,
            quorum: self.config.required_quorum,
            aggregated_hash,
        };

        Ok(FederatedPlan {
            policy_hash: self.config.policy_hash.clone(),
            epoch: self.config.epoch,
            timestamp: timestamp.into(),
            participants: participants.into_values().collect(),
            consensus,
            alignments,
        })
    }

    /// Produce a deterministic sample plan using canonical updates derived from node fingerprints.
    pub fn sample_plan(&self) -> FederatedPlan {
        let updates: Vec<FederatedUpdate> = self
            .nodes
            .iter()
            .map(|node| sample_update(node, self.config.epoch))
            .collect();

        self.simulate_epoch(&updates, iso_timestamp()).expect("sample plan generation should not fail")
    }
}

fn align_update_hash(update_hash: &str, ledger_block: &str, payload_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(update_hash.as_bytes());
    hasher.update(ledger_block.as_bytes());
    hasher.update(payload_hash.as_bytes());
    hex::encode(hasher.finalize())
}

fn aggregate_hash(inputs: &[(String, f32)]) -> String {
    let mut hasher = Sha256::new();
    // Stable ordering by hash string for determinism.
    let mut sorted = inputs.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (hash, weight) in sorted {
        hasher.update(hash.as_bytes());
        hasher.update(weight.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn sample_update(node: &FederatedNode, epoch: u64) -> FederatedUpdate {
    // Derive hashes deterministically from the node fingerprint and epoch.
    let seed = format!("{}-{}", node.fingerprint.0, epoch);
    let update_hash = deterministic_hash(&seed);
    let ledger_block_id = deterministic_hash(&format!("ledger-{seed}"));
    let payload_hash = deterministic_hash(&format!("payload-{seed}"));

    FederatedUpdate {
        node: node.fingerprint.clone(),
        epoch,
        update_hash,
        ledger_block_id,
        payload_hash,
        stake_weight: node.stake_weight,
    }
}

fn deterministic_hash(input: &str) -> String {
    hex::encode(Sha256::digest(input.as_bytes()))
}

fn iso_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nodes() -> Vec<FederatedNode> {
        vec![
            FederatedNode::new(
                "node-A",
                "1.0.0",
                ["mec", "ledger", "consensus"],
                1.0,
            ),
            FederatedNode::new("node-B", "1.0.0", ["mec", "consensus"], 0.8),
            FederatedNode::new("node-C", "1.0.0", ["mec"], 0.6),
        ]
    }

    fn simulator() -> FederationSimulator {
        let config = FederationConfig::new("policy-hash", 42, 2);
        FederationSimulator::new(config, sample_nodes())
    }

    #[test]
    fn sample_plan_passes_consensus() {
        let sim = simulator();
        let plan = sim.sample_plan();
        assert!(plan.consensus.passed);
        assert_eq!(plan.consensus.votes_for, 3);
        assert_eq!(plan.alignments.len(), 3);
    }

    #[test]
    fn simulate_epoch_rejects_unknown_node() {
        let sim = simulator();
        let bogus = FederatedUpdate {
            node: NodeFingerprint("unknown".into()),
            epoch: 42,
            update_hash: "abc".into(),
            ledger_block_id: "def".into(),
            payload_hash: "ghi".into(),
            stake_weight: 1.0,
        };
        let err = sim
            .simulate_epoch(&[bogus], "2025-01-01T00:00:00Z")
            .unwrap_err();
        assert!(matches!(err, FederationError::UnknownNode(_)));
    }

    #[test]
    fn aggregate_hash_is_deterministic() {
        let input = vec![
            ("hash-a".to_string(), 1.0),
            ("hash-b".to_string(), 0.5),
            ("hash-c".to_string(), 0.25),
        ];
        let first = aggregate_hash(&input);
        let second = aggregate_hash(&input);
        assert_eq!(first, second);
    }

    #[test]
    fn duplicate_updates_are_rejected() {
        let sim = simulator();
        let update = sample_update(&sim.nodes()[0], sim.config.epoch);
        let err = sim
            .simulate_epoch(&[update.clone(), update], "2025-01-01T00:00:00Z")
            .unwrap_err();
        assert!(matches!(err, FederationError::DuplicateUpdate(_)));
    }

    #[test]
    fn epoch_mismatch_is_rejected() {
        let sim = simulator();
        let mut update = sample_update(&sim.nodes()[0], sim.config.epoch);
        update.epoch = 999;
        let err = sim
            .simulate_epoch(&[update], "2025-01-01T00:00:00Z")
            .unwrap_err();
        assert!(matches!(err, FederationError::EpochMismatch { .. }));
    }
}
