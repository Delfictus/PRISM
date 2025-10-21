//! Federated readiness primitives (Phase M5 target).
//!
//! The goal of this module is to provide a minimal-yet-meaningful scaffold for
//! coordinating meta nodes across sites. It does **not** attempt to model the
//! full PBFT stack; instead it captures the key data flows that governance and
//! compliance automation expect:
//!   * deterministic node alignment for downstream aggregation;
//!   * per-node ledger anchors suitable for Merkle inclusion proofs; and
//!   * simulation hooks that generate summarized reports for auditing.

use std::fmt;

/// Configuration parameters that govern a federated epoch.
#[derive(Clone, Debug)]
pub struct FederationConfig {
    /// Minimum number of validator nodes that must participate in an epoch.
    pub quorum: usize,
    /// Maximum tolerated ledger drift before a node is quarantined.
    pub max_ledger_drift: u64,
    /// Epoch identifier used for synthetic simulations.
    pub epoch: u64,
}

impl FederationConfig {
    /// Constructs a configuration with sensible defaults for local testing.
    pub fn placeholder() -> Self {
        Self {
            quorum: 2,
            max_ledger_drift: 2,
            epoch: 1,
        }
    }
}

/// Describes the role a node plays inside the federation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeRole {
    /// Core validators participate in consensus and anchor ledger changes.
    CoreValidator,
    /// Edge participants contribute local learning updates but do not vote.
    EdgeParticipant,
}

impl fmt::Display for NodeRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            NodeRole::CoreValidator => "core",
            NodeRole::EdgeParticipant => "edge",
        };
        f.write_str(label)
    }
}

impl NodeRole {
    fn is_validator(&self) -> bool {
        matches!(self, NodeRole::CoreValidator)
    }
}

/// Immutable fingerprint describing a node.
#[derive(Clone, Debug)]
pub struct NodeProfile {
    pub id: String,
    pub region: String,
    pub role: NodeRole,
    /// Stake weight (arbitrary units) used for aggregation heuristics.
    pub stake: u32,
}

impl NodeProfile {
    pub fn new(id: impl Into<String>, region: impl Into<String>, role: NodeRole, stake: u32) -> Self {
        Self {
            id: id.into(),
            region: region.into(),
            role,
            stake,
        }
    }
}

/// Mutable per-node state maintained across epochs.
#[derive(Clone, Debug)]
struct NodeState {
    profile: NodeProfile,
    ledger_height: u64,
    last_anchor: String,
}

impl NodeState {
    fn new(profile: NodeProfile) -> Self {
        Self {
            profile,
            ledger_height: 0,
            last_anchor: String::new(),
        }
    }

    fn record_anchor(&mut self, anchor: String) {
        self.ledger_height += 1;
        self.last_anchor = anchor;
    }
}

/// Represents a deterministic update emitted by a federated node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeUpdate {
    pub node_id: String,
    pub ledger_height: u64,
    pub delta_score: i64,
    pub anchor_hash: String,
}

/// Ledger entry persisted for governance proofs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    pub epoch: u64,
    pub node_id: String,
    pub anchor_hash: String,
}

/// Aggregated simulation output used by compliance automation.
#[derive(Clone, Debug, Default)]
pub struct FederatedSimulationReport {
    pub epoch: u64,
    pub quorum_reached: bool,
    pub aligned_updates: Vec<NodeUpdate>,
    pub ledger_entries: Vec<LedgerEntry>,
    /// Weighted sum across updates—it acts as a deterministic aggregate.
    pub aggregated_delta: i64,
    /// Placeholder signature derived from the ledger Merkle root.
    pub signature: String,
}

/// Primary entry point for coordinating federated meta nodes.
pub struct FederatedInterface {
    config: FederationConfig,
    nodes: Vec<NodeState>,
}

impl FederatedInterface {
    /// Builds a new interface from configuration + node profiles.
    pub fn new(config: FederationConfig, profiles: Vec<NodeProfile>) -> Self {
        let nodes = profiles.into_iter().map(NodeState::new).collect();
        Self { config, nodes }
    }

    /// Convenience constructor for tests and command-line tooling.
    pub fn placeholder() -> Self {
        let config = FederationConfig::placeholder();
        let profiles = vec![
            NodeProfile::new("validator-a", "us-west", NodeRole::CoreValidator, 5),
            NodeProfile::new("validator-b", "eu-central", NodeRole::CoreValidator, 4),
            NodeProfile::new("edge-c", "ap-south", NodeRole::EdgeParticipant, 2),
        ];
        Self::new(config, profiles)
    }

    /// Produces a deterministic simulation report for the current epoch.
    ///
    /// The implementation intentionally keeps the computation straightforward so
    /// that auditors can reason about the numbers without executing complex
    /// kernels. Once Phase M5 graduates this scaffold, the simulation will be
    /// replaced with genuine federation logic.
    pub fn simulate_epoch(&mut self) -> FederatedSimulationReport {
        let epoch = self.config.epoch;
        self.config.epoch += 1;

        let mut updates = Vec::with_capacity(self.nodes.len());
        let mut ledger_entries = Vec::with_capacity(self.nodes.len());

        for node in &mut self.nodes {
            let delta = Self::derive_delta(epoch, node);
            let anchor = stable_hash(&format!(
                "{}:{}:{}:{}",
                node.profile.id, node.profile.region, epoch, node.ledger_height + 1
            ));
            node.record_anchor(anchor.clone());
            updates.push(NodeUpdate {
                node_id: node.profile.id.clone(),
                ledger_height: node.ledger_height,
                delta_score: delta,
                anchor_hash: anchor.clone(),
            });
            ledger_entries.push(LedgerEntry {
                epoch,
                node_id: node.profile.id.clone(),
                anchor_hash: anchor,
            });
        }

        updates.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        ledger_entries.sort_by(|a, b| a.node_id.cmp(&b.node_id));

        let aggregated_delta = updates.iter().map(|u| u.delta_score).sum::<i64>();
        let signature = compute_ledger_merkle(&ledger_entries);
        let quorum_reached = self
            .nodes
            .iter()
            .filter(|n| n.profile.role.is_validator())
            .count()
            >= self.config.quorum;

        FederatedSimulationReport {
            epoch,
            quorum_reached,
            aligned_updates: updates,
            ledger_entries,
            aggregated_delta,
            signature,
        }
    }

    /// Calculates the deterministic score delta for a node.
    fn derive_delta(epoch: u64, node: &NodeState) -> i64 {
        // Toy deterministic function:
        let base = i64::from(node.profile.stake);
        let regional_bias = stable_hash(&node.profile.region);
        let bias = i64::from_str_radix(&regional_bias[0..4], 16).unwrap_or(0) % 11;
        let epoch_penalty = (epoch % 5) as i64;
        base
            + i64::from(node.profile.role.is_validator() as u8) * 3
            - epoch_penalty
            + bias
    }

    /// Returns true when all nodes are within the configured ledger drift.
    pub fn ledger_within_drift(&self) -> bool {
        let min_height = self.nodes.iter().map(|n| n.ledger_height).min().unwrap_or(0);
        let max_height = self.nodes.iter().map(|n| n.ledger_height).max().unwrap_or(0);
        max_height - min_height <= self.config.max_ledger_drift
    }

    /// Exposes the current epoch number (mainly for inspection/testing).
    pub fn epoch(&self) -> u64 {
        self.config.epoch
    }
}

/// Simple deterministic hash for generating reproducible anchors.
fn stable_hash(input: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// Compute a deterministic Merkle-style hash for ledger entries.
pub fn compute_ledger_merkle(entries: &[LedgerEntry]) -> String {
    if entries.is_empty() {
        return stable_hash("ledger-empty");
    }
    let mut leaves: Vec<String> = entries
        .iter()
        .map(|entry| stable_hash(&format!("{}:{}", entry.node_id, entry.anchor_hash)))
        .collect();
    leaves.sort();

    let mut level: Vec<String> = leaves;
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for chunk in level.chunks(2) {
            let combined = if chunk.len() == 2 {
                format!("{}{}", chunk[0], chunk[1])
            } else {
                format!("{}{}", chunk[0], chunk[0])
            };
            next.push(stable_hash(&combined));
        }
        level = next;
    }
    level[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_is_deterministic() {
        let config = FederationConfig {
            quorum: 2,
            max_ledger_drift: 1,
            epoch: 42,
        };
        let nodes = vec![
            NodeProfile::new("core-a", "us-west", NodeRole::CoreValidator, 7),
            NodeProfile::new("core-b", "eu-central", NodeRole::CoreValidator, 5),
            NodeProfile::new("edge-c", "ap-south", NodeRole::EdgeParticipant, 3),
        ];

        let mut interface = FederatedInterface::new(config.clone(), nodes.clone());
        let report_a = interface.simulate_epoch();
        // Reset and run the same configuration again—we should get the same report.
        let mut interface_b = FederatedInterface::new(config, nodes);
        let report_b = interface_b.simulate_epoch();

        assert_eq!(report_a.epoch, 42);
        assert_eq!(report_a.epoch, report_b.epoch);
        assert_eq!(report_a.aligned_updates.len(), 3);
        assert_eq!(report_a.aligned_updates, report_b.aligned_updates);
        assert_eq!(report_a.ledger_entries, report_b.ledger_entries);
        assert!(report_a.quorum_reached);
    }

    #[test]
    fn ledger_drift_detection() {
        let mut interface = FederatedInterface::placeholder();
        assert!(interface.ledger_within_drift());

        // Run multiple epochs to advance ledger heights.
        for _ in 0..3 {
            interface.simulate_epoch();
        }
        assert!(interface.ledger_within_drift());
    }

    #[test]
    fn ledger_merkle_is_deterministic() {
        let entries = vec![
            LedgerEntry {
                epoch: 1,
                node_id: "a".into(),
                anchor_hash: "1234".into(),
            },
            LedgerEntry {
                epoch: 1,
                node_id: "b".into(),
                anchor_hash: "5678".into(),
            },
        ];
        let root_a = compute_ledger_merkle(&entries);
        let mut shuffled = entries.clone();
        shuffled.reverse();
        let root_b = compute_ledger_merkle(&shuffled);
        assert_eq!(root_a, root_b);
    }
}
