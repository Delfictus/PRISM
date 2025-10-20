//! Health check and monitoring endpoints for the ingestion system

use super::engine::IngestionStats;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Overall system health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// System is healthy and operating normally
    Healthy,
    /// System is degraded but still operational
    Degraded,
    /// System is unhealthy and may not be functioning correctly
    Unhealthy,
}

/// Comprehensive health check report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system health status
    pub status: HealthStatus,
    /// Timestamp of health check
    pub timestamp: DateTime<Utc>,
    /// Component-level health checks
    pub components: HashMap<String, ComponentHealth>,
    /// System-level metrics
    pub metrics: HealthMetrics,
    /// Any warnings or issues
    pub warnings: Vec<String>,
}

/// Health status of an individual component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component health status
    pub status: HealthStatus,
    /// Human-readable message
    pub message: String,
    /// Component-specific details
    pub details: HashMap<String, serde_json::Value>,
}

/// Key health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Total data points ingested
    pub total_points: usize,
    /// Current ingestion rate (points/sec)
    pub ingestion_rate: f64,
    /// Number of active sources
    pub active_sources: usize,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// Buffer utilization (percentage)
    pub buffer_utilization: f64,
    /// Average latency (milliseconds)
    pub avg_latency_ms: Option<f64>,
}

impl HealthReport {
    /// Create a health report from ingestion stats
    pub fn from_stats(stats: IngestionStats, buffer_size: usize, buffer_capacity: usize) -> Self {
        let mut components = HashMap::new();
        let mut warnings = Vec::new();

        // Check ingestion rate
        let ingestion_health = if stats.average_rate_hz > 10.0 {
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Ingestion rate: {:.1} points/sec", stats.average_rate_hz),
                details: HashMap::new(),
            }
        } else if stats.average_rate_hz > 1.0 {
            warnings.push("Low ingestion rate detected".to_string());
            ComponentHealth {
                status: HealthStatus::Degraded,
                message: format!(
                    "Low ingestion rate: {:.1} points/sec",
                    stats.average_rate_hz
                ),
                details: HashMap::new(),
            }
        } else {
            warnings.push("Very low or no ingestion".to_string());
            ComponentHealth {
                status: HealthStatus::Unhealthy,
                message: "Ingestion stalled or stopped".to_string(),
                details: HashMap::new(),
            }
        };
        components.insert("ingestion".to_string(), ingestion_health);

        // Check error rate
        let error_rate = if stats.total_points > 0 {
            (stats.error_count as f64 / stats.total_points as f64) * 100.0
        } else {
            0.0
        };

        let error_health = if error_rate < 1.0 {
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Error rate: {:.2}%", error_rate),
                details: HashMap::new(),
            }
        } else if error_rate < 5.0 {
            warnings.push(format!("Elevated error rate: {:.2}%", error_rate));
            ComponentHealth {
                status: HealthStatus::Degraded,
                message: format!("Elevated errors: {:.2}%", error_rate),
                details: HashMap::new(),
            }
        } else {
            warnings.push(format!("High error rate: {:.2}%", error_rate));
            ComponentHealth {
                status: HealthStatus::Unhealthy,
                message: format!("High error rate: {:.2}%", error_rate),
                details: HashMap::new(),
            }
        };
        components.insert("errors".to_string(), error_health);

        // Check circuit breakers
        let mut cb_open_count = 0;
        for (source, state) in &stats.circuit_breaker_states {
            if state.contains("open") {
                cb_open_count += 1;
                warnings.push(format!("Circuit breaker open for: {}", source));
            }
        }

        let cb_health = if cb_open_count == 0 {
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: "All circuit breakers closed".to_string(),
                details: HashMap::new(),
            }
        } else {
            ComponentHealth {
                status: HealthStatus::Degraded,
                message: format!("{} circuit breaker(s) open", cb_open_count),
                details: HashMap::new(),
            }
        };
        components.insert("circuit_breakers".to_string(), cb_health);

        // Check buffer utilization
        let buffer_pct = if buffer_capacity > 0 {
            (buffer_size as f64 / buffer_capacity as f64) * 100.0
        } else {
            0.0
        };

        let buffer_health = if buffer_pct < 80.0 {
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Buffer: {:.1}% full", buffer_pct),
                details: HashMap::new(),
            }
        } else if buffer_pct < 95.0 {
            warnings.push(format!("Buffer utilization high: {:.1}%", buffer_pct));
            ComponentHealth {
                status: HealthStatus::Degraded,
                message: format!("Buffer nearly full: {:.1}%", buffer_pct),
                details: HashMap::new(),
            }
        } else {
            warnings.push("Buffer nearly full".to_string());
            ComponentHealth {
                status: HealthStatus::Unhealthy,
                message: format!("Buffer full: {:.1}%", buffer_pct),
                details: HashMap::new(),
            }
        };
        components.insert("buffer".to_string(), buffer_health);

        // Determine overall status
        let overall_status = if components
            .values()
            .all(|c| c.status == HealthStatus::Healthy)
        {
            HealthStatus::Healthy
        } else if components
            .values()
            .any(|c| c.status == HealthStatus::Unhealthy)
        {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        Self {
            status: overall_status,
            timestamp: Utc::now(),
            components,
            metrics: HealthMetrics {
                total_points: stats.total_points,
                ingestion_rate: stats.average_rate_hz,
                active_sources: stats.active_sources,
                error_rate,
                buffer_utilization: buffer_pct,
                avg_latency_ms: None, // Can be added if tracked
            },
            warnings,
        }
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }

    /// Get a summary message
    pub fn summary(&self) -> String {
        format!(
            "{:?} - {} sources, {:.1} pts/sec, {:.2}% errors",
            self.status,
            self.metrics.active_sources,
            self.metrics.ingestion_rate,
            self.metrics.error_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_healthy_report() {
        let mut stats = IngestionStats {
            total_points: 1000,
            total_bytes: 0,
            last_update: Instant::now(),
            average_rate_hz: 100.0,
            active_sources: 3,
            error_count: 5,
            retry_success_count: 0,
            retry_failed_count: 0,
            circuit_breaker_states: HashMap::new(),
        };

        stats
            .circuit_breaker_states
            .insert("test".to_string(), "closed".to_string());

        let report = HealthReport::from_stats(stats, 100, 10000);

        assert_eq!(report.status, HealthStatus::Healthy);
        assert!(report.is_healthy());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_degraded_report() {
        let mut stats = IngestionStats {
            total_points: 1000,
            total_bytes: 0,
            last_update: Instant::now(),
            average_rate_hz: 5.0, // Low rate
            active_sources: 3,
            error_count: 30, // 3% error rate
            retry_success_count: 0,
            retry_failed_count: 0,
            circuit_breaker_states: HashMap::new(),
        };

        stats
            .circuit_breaker_states
            .insert("test".to_string(), "closed".to_string());

        let report = HealthReport::from_stats(stats, 100, 10000);

        assert_eq!(report.status, HealthStatus::Degraded);
        assert!(!report.is_healthy());
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_unhealthy_report() {
        let mut stats = IngestionStats {
            total_points: 1000,
            total_bytes: 0,
            last_update: Instant::now(),
            average_rate_hz: 0.5, // Very low
            active_sources: 3,
            error_count: 100, // 10% error rate
            retry_success_count: 0,
            retry_failed_count: 0,
            circuit_breaker_states: HashMap::new(),
        };

        stats
            .circuit_breaker_states
            .insert("test".to_string(), "open".to_string());

        let report = HealthReport::from_stats(stats, 9500, 10000); // Buffer nearly full

        assert_eq!(report.status, HealthStatus::Unhealthy);
        assert!(!report.is_healthy());
        assert!(report.warnings.len() >= 3);
    }
}
