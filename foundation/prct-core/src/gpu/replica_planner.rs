//! Replica Planning and Assignment
//!
//! Distributes replicas across GPUs with phase-specific assignments.
//! Implements embarrassingly parallel strategy with optional snapshot sharing.
//!
//! Constitutional Compliance:
//! - No stubs, full implementation
//! - Device assignments based on topology and config
//! - Per-replica stream allocation

use crate::errors::*;
use crate::gpu::device_topology::DeviceInfo;

/// Phase execution mask for replica assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseMask {
    /// Execute Phase 0 (Reservoir prediction)
    pub phase0_reservoir: bool,

    /// Execute Phase 1 (Transfer Entropy + Active Inference)
    pub phase1_te_ai: bool,

    /// Execute Phase 2 (Thermodynamic equilibration)
    pub phase2_thermo: bool,

    /// Execute Phase 3 (Quantum annealing)
    pub phase3_quantum: bool,

    /// Execute Phase 4 (Memetic search)
    pub phase4_memetic: bool,
}

impl PhaseMask {
    /// All phases enabled
    pub fn all() -> Self {
        Self {
            phase0_reservoir: true,
            phase1_te_ai: true,
            phase2_thermo: true,
            phase3_quantum: true,
            phase4_memetic: true,
        }
    }

    /// Only embarrassingly parallel phases (thermo, quantum, memetic)
    pub fn parallel_only() -> Self {
        Self {
            phase0_reservoir: false,
            phase1_te_ai: false,
            phase2_thermo: true,
            phase3_quantum: true,
            phase4_memetic: true,
        }
    }

    /// Primary replica (runs all phases including reservoir/TE)
    pub fn primary() -> Self {
        Self::all()
    }

    /// Secondary replica (skips reservoir/TE, runs parallel phases)
    pub fn secondary() -> Self {
        Self::parallel_only()
    }
}

/// Stream allocation for a replica
#[derive(Debug, Clone)]
pub struct ReplicaStreams {
    /// Number of streams allocated to this replica
    pub count: usize,

    /// Stream pool for this replica
    /// Note: Actual CudaStreamPool created on replica thread
    pub stream_mode: crate::gpu::state::StreamMode,
}

impl ReplicaStreams {
    pub fn new(count: usize, mode: crate::gpu::state::StreamMode) -> Self {
        Self {
            count,
            stream_mode: mode,
        }
    }
}

/// Assignment of replica to device with phase and stream allocation
#[derive(Debug, Clone)]
pub struct ReplicaAssignment {
    /// Replica identifier (0-based)
    pub replica_id: usize,

    /// Device information
    pub device: DeviceInfo,

    /// Stream allocation
    pub streams: ReplicaStreams,

    /// Phase execution mask
    pub phase_mask: PhaseMask,

    /// Seed offset for deterministic RNG
    pub seed_offset: u64,

    /// Is primary replica (runs reservoir/TE)
    pub is_primary: bool,
}

impl ReplicaAssignment {
    /// Create primary replica assignment
    pub fn primary(
        replica_id: usize,
        device: DeviceInfo,
        streams: ReplicaStreams,
        seed_offset: u64,
    ) -> Self {
        Self {
            replica_id,
            device,
            streams,
            phase_mask: PhaseMask::primary(),
            seed_offset,
            is_primary: true,
        }
    }

    /// Create secondary replica assignment
    pub fn secondary(
        replica_id: usize,
        device: DeviceInfo,
        streams: ReplicaStreams,
        seed_offset: u64,
    ) -> Self {
        Self {
            replica_id,
            device,
            streams,
            phase_mask: PhaseMask::secondary(),
            seed_offset,
            is_primary: false,
        }
    }
}

impl std::fmt::Display for ReplicaAssignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Replica {} [{}]: device={}, streams={}, primary={}, phases=[R:{}, TE:{}, Th:{}, Q:{}, M:{}]",
            self.replica_id,
            if self.is_primary { "PRIMARY" } else { "SECONDARY" },
            self.device.id,
            self.streams.count,
            self.is_primary,
            self.phase_mask.phase0_reservoir as u8,
            self.phase_mask.phase1_te_ai as u8,
            self.phase_mask.phase2_thermo as u8,
            self.phase_mask.phase3_quantum as u8,
            self.phase_mask.phase4_memetic as u8,
        )
    }
}

/// Replica planner for multi-GPU distribution
pub struct ReplicaPlanner;

impl ReplicaPlanner {
    /// Plan replica assignments across devices
    ///
    /// # Arguments
    /// * `devices` - Discovered device topology
    /// * `requested_replicas` - Number of replicas requested (or "auto" via Option)
    /// * `streams_per_replica` - Number of streams for each replica
    /// * `stream_mode` - Stream execution mode
    /// * `base_seed` - Base seed for deterministic RNG
    ///
    /// # Returns
    /// Vector of replica assignments
    ///
    /// # Strategy
    /// - If replicas <= devices: one replica per device
    /// - If replicas > devices: round-robin across devices
    /// - Replica 0 is always primary (runs reservoir/TE)
    /// - Other replicas are secondary (parallel phases only)
    pub fn plan(
        devices: &[DeviceInfo],
        requested_replicas: Option<usize>,
        streams_per_replica: usize,
        stream_mode: crate::gpu::state::StreamMode,
        base_seed: u64,
    ) -> Result<Vec<ReplicaAssignment>> {
        if devices.is_empty() {
            return Err(PRCTError::ConfigError(
                "Cannot plan replicas with zero devices".to_string()
            ));
        }

        // Determine actual replica count
        let num_replicas = match requested_replicas {
            Some(n) if n > 0 => n,
            Some(0) => {
                return Err(PRCTError::ConfigError(
                    "Replica count must be > 0".to_string()
                ));
            }
            None => devices.len(), // Auto: one replica per device
            _ => devices.len(),
        };

        println!("[REPLICA-PLANNER][PLAN] Planning {} replica(s) across {} device(s)", num_replicas, devices.len());

        let mut assignments = Vec::with_capacity(num_replicas);

        for replica_id in 0..num_replicas {
            // Round-robin device assignment
            let device_index = replica_id % devices.len();
            let device = devices[device_index].clone();

            let streams = ReplicaStreams::new(streams_per_replica, stream_mode);
            let seed_offset = (replica_id as u64) * 1000000;

            let assignment = if replica_id == 0 {
                // Replica 0 is always primary
                ReplicaAssignment::primary(replica_id, device, streams, base_seed + seed_offset)
            } else {
                // Other replicas are secondary
                ReplicaAssignment::secondary(replica_id, device, streams, base_seed + seed_offset)
            };

            println!("[REPLICA-PLANNER][PLAN] {}", assignment);

            assignments.push(assignment);
        }

        println!("[REPLICA-PLANNER][PLAN] Replica planning complete: {} assignment(s)", assignments.len());

        Ok(assignments)
    }

    /// Validate assignments for correctness
    pub fn validate(assignments: &[ReplicaAssignment]) -> Result<()> {
        if assignments.is_empty() {
            return Err(PRCTError::ConfigError(
                "No replica assignments provided".to_string()
            ));
        }

        // Check that replica 0 is primary
        if !assignments[0].is_primary {
            return Err(PRCTError::ConfigError(
                "Replica 0 must be primary".to_string()
            ));
        }

        // Check that all replica IDs are unique and sequential
        for (i, assignment) in assignments.iter().enumerate() {
            if assignment.replica_id != i {
                return Err(PRCTError::ConfigError(
                    format!("Replica IDs must be sequential, expected {} but got {}", i, assignment.replica_id)
                ));
            }
        }

        // Check that stream counts are reasonable
        for assignment in assignments {
            if assignment.streams.count == 0 || assignment.streams.count > 32 {
                return Err(PRCTError::ConfigError(
                    format!("Replica {} has invalid stream count {}", assignment.replica_id, assignment.streams.count)
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_device(id: usize) -> DeviceInfo {
        DeviceInfo {
            id,
            name: format!("Mock Device {}", id),
            total_memory_bytes: 24 * 1024 * 1024 * 1024,
            total_memory_mb: 24 * 1024,
            compute_capability: (8, 9),
            sm_count: 128,
            pci_bus_id: format!("0000:{:02x}:00.0", id),
            hostname: "localhost".to_string(),
        }
    }

    #[test]
    fn test_phase_mask() {
        let all = PhaseMask::all();
        assert!(all.phase0_reservoir);
        assert!(all.phase1_te_ai);
        assert!(all.phase2_thermo);

        let parallel = PhaseMask::parallel_only();
        assert!(!parallel.phase0_reservoir);
        assert!(!parallel.phase1_te_ai);
        assert!(parallel.phase2_thermo);
    }

    #[test]
    fn test_replica_planner_single_device() {
        let devices = vec![mock_device(0)];

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(1),
            4,
            crate::gpu::state::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].replica_id, 0);
        assert_eq!(assignments[0].device.id, 0);
        assert!(assignments[0].is_primary);
    }

    #[test]
    fn test_replica_planner_multi_device() {
        let devices = vec![
            mock_device(0),
            mock_device(1),
            mock_device(2),
            mock_device(3),
        ];

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(4),
            4,
            crate::gpu::state::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), 4);

        // Replica 0 should be primary
        assert!(assignments[0].is_primary);

        // Other replicas should be secondary
        for i in 1..4 {
            assert!(!assignments[i].is_primary);
        }

        // Each replica should be on different device
        for i in 0..4 {
            assert_eq!(assignments[i].device.id, i);
        }
    }

    #[test]
    fn test_replica_planner_more_replicas_than_devices() {
        let devices = vec![
            mock_device(0),
            mock_device(1),
        ];

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(8),
            4,
            crate::gpu::state::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), 8);

        // Check round-robin assignment
        for (i, assignment) in assignments.iter().enumerate() {
            let expected_device = i % 2;
            assert_eq!(assignment.device.id, expected_device);
        }
    }

    #[test]
    fn test_replica_planner_auto() {
        let devices = vec![
            mock_device(0),
            mock_device(1),
            mock_device(2),
        ];

        let assignments = ReplicaPlanner::plan(
            &devices,
            None, // Auto
            4,
            crate::gpu::state::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        // Auto should create one replica per device
        assert_eq!(assignments.len(), 3);
    }

    #[test]
    fn test_replica_planner_validation() {
        let devices = vec![mock_device(0)];

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(1),
            4,
            crate::gpu::state::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        // Should pass validation
        ReplicaPlanner::validate(&assignments).expect("Validation failed");
    }

    #[test]
    fn test_replica_planner_validation_empty() {
        let assignments: Vec<ReplicaAssignment> = vec![];

        // Should fail validation
        assert!(ReplicaPlanner::validate(&assignments).is_err());
    }
}
