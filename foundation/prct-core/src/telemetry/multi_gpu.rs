//! Multi-GPU Telemetry Extensions
//!
//! Tracks per-replica metrics, device utilization, and coordination overhead.
//! Extends base telemetry with distributed runtime monitoring.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Per-replica metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaTelemetry {
    pub replica_id: usize,
    pub device_id: usize,
    pub phases_executed: usize,
    pub total_runtime_ms: u64,
    pub best_chromatic: Option<usize>,
    pub last_heartbeat_ms: u64,
    pub queue_depth: usize,
    pub last_kernel_ms: f64,
    pub is_healthy: bool,
}

/// Device utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceUtilization {
    pub device_id: usize,
    pub device_name: String,
    pub assigned_replicas: Vec<usize>,
    pub total_phases_executed: usize,
    pub cumulative_runtime_ms: u64,
    pub memory_allocated_mb: f64,
    pub sm_utilization_percent: f64,
}

/// Coordination overhead metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub sync_interval_ms: u64,
    pub total_syncs: usize,
    pub sync_overhead_ms: u64,
    pub snapshot_broadcasts: usize,
    pub command_broadcasts: usize,
    pub event_messages: usize,
}

/// Multi-GPU runtime summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGpuSummary {
    pub profile_name: String,
    pub num_replicas: usize,
    pub num_devices: usize,
    pub replicas: Vec<ReplicaTelemetry>,
    pub devices: Vec<DeviceUtilization>,
    pub coordination: CoordinationMetrics,
    pub global_best_chromatic: Option<usize>,
    pub global_best_conflicts: usize,
    pub total_runtime_ms: u64,
    pub parallel_speedup: f64,
}

impl MultiGpuSummary {
    /// Calculate parallel speedup (theoretical vs actual)
    pub fn calculate_speedup(&mut self, baseline_runtime_ms: u64) {
        if baseline_runtime_ms > 0 && self.total_runtime_ms > 0 {
            self.parallel_speedup = baseline_runtime_ms as f64 / self.total_runtime_ms as f64;
        } else {
            self.parallel_speedup = 1.0;
        }
    }

    /// Format summary for display
    pub fn format_summary(&self) -> String {
        let mut lines = vec![
            format!("Multi-GPU Runtime Summary"),
            format!("========================="),
            format!("Profile: {}", self.profile_name),
            format!("Replicas: {} across {} device(s)", self.num_replicas, self.num_devices),
            format!("Total Runtime: {:.2}s", self.total_runtime_ms as f64 / 1000.0),
            format!("Parallel Speedup: {:.2}x", self.parallel_speedup),
            format!(""),
            format!("Global Best:"),
            format!("  Chromatic: {:?}", self.global_best_chromatic),
            format!("  Conflicts: {}", self.global_best_conflicts),
            format!(""),
            format!("Replica Performance:"),
        ];

        for replica in &self.replicas {
            lines.push(format!(
                "  Replica {} [GPU {}]: {} phases, {:.2}s, best={:?}, healthy={}",
                replica.replica_id,
                replica.device_id,
                replica.phases_executed,
                replica.total_runtime_ms as f64 / 1000.0,
                replica.best_chromatic,
                replica.is_healthy,
            ));
        }

        lines.push(format!(""));
        lines.push(format!("Device Utilization:"));

        for device in &self.devices {
            lines.push(format!(
                "  GPU {} [{}]: {} replicas, {} phases, {:.2}s",
                device.device_id,
                device.device_name,
                device.assigned_replicas.len(),
                device.total_phases_executed,
                device.cumulative_runtime_ms as f64 / 1000.0,
            ));
        }

        lines.push(format!(""));
        lines.push(format!("Coordination Overhead:"));
        lines.push(format!(
            "  Syncs: {} (interval={}ms, overhead={:.2}s)",
            self.coordination.total_syncs,
            self.coordination.sync_interval_ms,
            self.coordination.sync_overhead_ms as f64 / 1000.0,
        ));
        lines.push(format!(
            "  Messages: {} snapshots, {} commands, {} events",
            self.coordination.snapshot_broadcasts,
            self.coordination.command_broadcasts,
            self.coordination.event_messages,
        ));

        lines.join("\n")
    }
}

/// Watchdog for replica health monitoring
#[derive(Debug)]
pub struct ReplicaWatchdog {
    timeout_ms: u64,
    last_check: Instant,
    unhealthy_replicas: Vec<usize>,
}

impl ReplicaWatchdog {
    /// Create new watchdog with timeout
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            last_check: Instant::now(),
            unhealthy_replicas: Vec::new(),
        }
    }

    /// Check replica health
    pub fn check_health(&mut self, replicas: &[ReplicaTelemetry]) -> Vec<usize> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_check).as_millis() as u64;

        self.unhealthy_replicas.clear();

        for replica in replicas {
            if replica.last_heartbeat_ms > self.timeout_ms {
                self.unhealthy_replicas.push(replica.replica_id);
            }
        }

        self.last_check = now;

        self.unhealthy_replicas.clone()
    }

    /// Get current unhealthy replicas
    pub fn unhealthy_replicas(&self) -> &[usize] {
        &self.unhealthy_replicas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_summary() {
        let summary = MultiGpuSummary {
            profile_name: "test".to_string(),
            num_replicas: 2,
            num_devices: 2,
            replicas: vec![
                ReplicaTelemetry {
                    replica_id: 0,
                    device_id: 0,
                    phases_executed: 5,
                    total_runtime_ms: 1000,
                    best_chromatic: Some(100),
                    last_heartbeat_ms: 50,
                    queue_depth: 0,
                    last_kernel_ms: 10.0,
                    is_healthy: true,
                },
                ReplicaTelemetry {
                    replica_id: 1,
                    device_id: 1,
                    phases_executed: 5,
                    total_runtime_ms: 1000,
                    best_chromatic: Some(95),
                    last_heartbeat_ms: 50,
                    queue_depth: 0,
                    last_kernel_ms: 10.0,
                    is_healthy: true,
                },
            ],
            devices: vec![],
            coordination: CoordinationMetrics {
                sync_interval_ms: 100,
                total_syncs: 10,
                sync_overhead_ms: 50,
                snapshot_broadcasts: 5,
                command_broadcasts: 15,
                event_messages: 30,
            },
            global_best_chromatic: Some(95),
            global_best_conflicts: 0,
            total_runtime_ms: 1100,
            parallel_speedup: 1.0,
        };

        let formatted = summary.format_summary();
        assert!(formatted.contains("Multi-GPU Runtime Summary"));
        assert!(formatted.contains("Replicas: 2 across 2 device(s)"));
    }

    #[test]
    fn test_watchdog() {
        let mut watchdog = ReplicaWatchdog::new(5000);

        let replicas = vec![
            ReplicaTelemetry {
                replica_id: 0,
                device_id: 0,
                phases_executed: 5,
                total_runtime_ms: 1000,
                best_chromatic: Some(100),
                last_heartbeat_ms: 500, // Healthy
                queue_depth: 0,
                last_kernel_ms: 10.0,
                is_healthy: true,
            },
            ReplicaTelemetry {
                replica_id: 1,
                device_id: 1,
                phases_executed: 5,
                total_runtime_ms: 1000,
                best_chromatic: Some(95),
                last_heartbeat_ms: 10000, // Unhealthy
                queue_depth: 0,
                last_kernel_ms: 10.0,
                is_healthy: false,
            },
        ];

        let unhealthy = watchdog.check_health(&replicas);
        assert_eq!(unhealthy.len(), 1);
        assert_eq!(unhealthy[0], 1);
    }
}
