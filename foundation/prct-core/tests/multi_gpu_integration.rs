//! Multi-GPU Integration Tests
//!
//! Tests device discovery, replica planning, and distributed runtime.
//! Uses CUDA_VISIBLE_DEVICES to simulate multi-GPU environments.

#[cfg(feature = "cuda")]
mod multi_gpu_tests {
    use prct_core::gpu::{
        discover_devices, DeviceInfo, ReplicaPlanner, ReplicaAssignment,
        DistributedRuntime, ReplicaCommand, DeviceProfile, load_device_profile,
    };
    use prct_core::errors::*;
    use std::path::Path;

    #[test]
    fn test_device_discovery_single() {
        let filters = vec!["cuda:0".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover device 0");

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, 0);
        assert!(devices[0].total_memory_mb > 0);
        assert!(!devices[0].name.is_empty());

        println!("Discovered device: {}", devices[0]);
    }

    #[test]
    fn test_device_discovery_all() {
        let filters = vec!["cuda:*".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover all devices");

        assert!(!devices.is_empty());

        for (i, device) in devices.iter().enumerate() {
            assert_eq!(device.id, i);
            println!("Device {}: {}", i, device);
        }
    }

    #[test]
    fn test_replica_planner_single_device() {
        let filters = vec!["cuda:0".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover devices");

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(1),
            4,
            prct_core::gpu::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].replica_id, 0);
        assert_eq!(assignments[0].device.id, 0);
        assert!(assignments[0].is_primary);

        println!("Assignment: {}", assignments[0]);

        ReplicaPlanner::validate(&assignments).expect("Validation failed");
    }

    #[test]
    fn test_replica_planner_multi_replica() {
        let filters = vec!["cuda:0".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover devices");

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(4), // 4 replicas on 1 device
            4,
            prct_core::gpu::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), 4);

        // Replica 0 should be primary
        assert!(assignments[0].is_primary);

        // Other replicas should be secondary
        for i in 1..4 {
            assert!(!assignments[i].is_primary);
        }

        // All should be on same device
        for assignment in &assignments {
            assert_eq!(assignment.device.id, 0);
        }

        ReplicaPlanner::validate(&assignments).expect("Validation failed");
    }

    #[test]
    #[ignore] // Only run when multiple GPUs available
    fn test_replica_planner_multi_device() {
        let filters = vec!["cuda:*".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover devices");

        if devices.len() < 2 {
            println!("Skipping multi-device test (only {} GPU(s) available)", devices.len());
            return;
        }

        let assignments = ReplicaPlanner::plan(
            &devices,
            None, // Auto: one replica per device
            4,
            prct_core::gpu::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        assert_eq!(assignments.len(), devices.len());

        // Each replica should be on different device
        for (i, assignment) in assignments.iter().enumerate() {
            assert_eq!(assignment.device.id, i);
        }

        ReplicaPlanner::validate(&assignments).expect("Validation failed");
    }

    #[test]
    fn test_distributed_runtime_single_replica() {
        let filters = vec!["cuda:0".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover devices");

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(1),
            4,
            prct_core::gpu::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        let runtime = DistributedRuntime::new(assignments, 100)
            .expect("Runtime creation failed");

        assert_eq!(runtime.replicas().len(), 1);

        println!("Runtime initialized with {} replica(s)", runtime.replicas().len());

        // Test broadcast command
        runtime.broadcast_command(ReplicaCommand::Shutdown)
            .expect("Broadcast failed");

        runtime.shutdown().expect("Shutdown failed");

        println!("Runtime shutdown successful");
    }

    #[test]
    #[ignore] // Only run when multiple GPUs available
    fn test_distributed_runtime_multi_replica() {
        let filters = vec!["cuda:*".to_string()];
        let devices = discover_devices(&filters).expect("Failed to discover devices");

        if devices.len() < 2 {
            println!("Skipping multi-replica test (only {} GPU(s) available)", devices.len());
            return;
        }

        let num_replicas = devices.len().min(4); // Use up to 4 replicas

        let assignments = ReplicaPlanner::plan(
            &devices,
            Some(num_replicas),
            4,
            prct_core::gpu::StreamMode::Parallel,
            12345,
        ).expect("Planning failed");

        let runtime = DistributedRuntime::new(assignments, 100)
            .expect("Runtime creation failed");

        assert_eq!(runtime.replicas().len(), num_replicas);

        println!("Runtime initialized with {} replica(s)", runtime.replicas().len());

        // Check health
        let health = runtime.health_check(std::time::Duration::from_secs(5));
        println!("Health check: {:?}", health);

        for (replica_id, is_healthy) in health {
            assert!(is_healthy, "Replica {} is unhealthy", replica_id);
        }

        // Test broadcast command
        runtime.broadcast_command(ReplicaCommand::Shutdown)
            .expect("Broadcast failed");

        runtime.shutdown().expect("Shutdown failed");

        println!("Runtime shutdown successful");
    }

    #[test]
    fn test_device_profile_loading() {
        // Test loading rtx5070 profile
        let profile_path = Path::new("foundation/prct-core/configs/device_profiles.toml");

        if !profile_path.exists() {
            println!("Skipping profile test (config file not found)");
            return;
        }

        let (name, profile) = load_device_profile(Some(profile_path), Some("rtx5070"))
            .expect("Failed to load rtx5070 profile");

        assert_eq!(name, "rtx5070");
        assert_eq!(profile.mode, prct_core::gpu::ProfileMode::Local);
        assert_eq!(profile.device_filter, vec!["cuda:0"]);

        println!("Loaded profile: {} - mode={:?}, replicas={:?}", name, profile.mode, profile.replicas);
    }

    #[test]
    fn test_device_profile_validation() {
        use prct_core::gpu::{ProfileMode, ProfileStrategy, ReplicaCount};

        // Valid profile
        let valid_profile = DeviceProfile {
            mode: ProfileMode::Local,
            replicas: ReplicaCount::Fixed(1),
            device_filter: vec!["cuda:0".to_string()],
            sync_interval_ms: 0,
            allow_remote: false,
            enable_peer_access: false,
            strategy: ProfileStrategy::SingleDevice,
            verbose: false,
        };

        assert!(valid_profile.validate().is_ok());

        // Invalid profile (empty filter)
        let invalid_profile = DeviceProfile {
            mode: ProfileMode::Local,
            replicas: ReplicaCount::Fixed(1),
            device_filter: vec![], // Empty
            sync_interval_ms: 0,
            allow_remote: false,
            enable_peer_access: false,
            strategy: ProfileStrategy::SingleDevice,
            verbose: false,
        };

        assert!(invalid_profile.validate().is_err());
    }

    #[test]
    fn test_replica_count_resolution() {
        use prct_core::gpu::ReplicaCount;

        let auto = ReplicaCount::Auto;
        assert_eq!(auto.resolve(8), 8);
        assert_eq!(auto.resolve(1), 1);

        let fixed = ReplicaCount::Fixed(4);
        assert_eq!(fixed.resolve(8), 4);
        assert_eq!(fixed.resolve(1), 4);
    }
}
