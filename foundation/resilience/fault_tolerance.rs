//! Health Monitoring and Fault Tolerance
//!
//! This module implements the centralized health monitoring system that acts as the platform's
//! "nervous system," tracking component health and enabling graceful degradation.
//!
//! # Architecture
//!
//! The `HealthMonitor` uses lock-free concurrent data structures (DashMap) to track the health
//! of individual components and aggregate them into a global system state.
//!
//! # State Machine
//!
//! ## Component States
//! - `Healthy`: Component operating normally
//! - `Degraded`: Component experiencing issues but functional
//! - `Unhealthy`: Component non-functional
//!
//! ## System States
//! - `Running`: All critical components healthy
//! - `Degraded`: Some components degraded, non-essential load shed
//! - `Critical`: Multiple failures, system barely functional
//!
//! # Mathematical Foundation
//!
//! System health is quantified using an availability metric:
//!
//! ```text
//! A(t) = (1/N) Σᵢ wᵢ · hᵢ(t)
//! ```
//!
//! where:
//! - N = number of components
//! - wᵢ = weight/criticality of component i
//! - hᵢ(t) ∈ {0, 0.5, 1} = health value (Unhealthy, Degraded, Healthy)
//!
//! System state transitions based on aggregate availability:
//! - A ≥ 0.9 → Running
//! - 0.5 ≤ A < 0.9 → Degraded
//! - A < 0.5 → Critical

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Component health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component operating normally
    Healthy,
    /// Component experiencing issues but functional
    Degraded,
    /// Component non-functional
    Unhealthy,
}

impl HealthStatus {
    /// Convert health status to numeric availability (0.0 to 1.0)
    pub fn availability(&self) -> f64 {
        match self {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 0.0,
        }
    }
}

/// Global system state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemState {
    /// All critical components healthy (A ≥ 0.9)
    Running,
    /// Some components degraded, graceful degradation active (0.5 ≤ A < 0.9)
    Degraded,
    /// Multiple failures, system barely functional (A < 0.5)
    Critical,
}

/// Component health information
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Current health status
    pub status: HealthStatus,
    /// Component criticality weight (0.0 to 1.0)
    pub weight: f64,
    /// Last update timestamp
    pub last_update: Instant,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Total failures since startup
    pub total_failures: u64,
    /// Component uptime
    pub uptime: Duration,
}

impl ComponentHealth {
    /// Create new healthy component
    pub fn new(weight: f64) -> Self {
        Self {
            status: HealthStatus::Healthy,
            weight: weight.clamp(0.0, 1.0),
            last_update: Instant::now(),
            failure_count: 0,
            total_failures: 0,
            uptime: Duration::ZERO,
        }
    }

    /// Update health status
    pub fn update_status(&mut self, status: HealthStatus) {
        let now = Instant::now();
        self.uptime += now.duration_since(self.last_update);
        self.last_update = now;

        if status != HealthStatus::Healthy && self.status == HealthStatus::Healthy {
            // Transition from healthy to unhealthy/degraded
            self.failure_count += 1;
            self.total_failures += 1;
        } else if status == HealthStatus::Healthy && self.status != HealthStatus::Healthy {
            // Recovery
            self.failure_count = 0;
        }

        self.status = status;
    }

    /// Check if component is stale (no update for >30s)
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_update.elapsed() > timeout
    }
}

/// Health monitoring system
///
/// Thread-safe health tracker using DashMap for lock-free concurrent access.
pub struct HealthMonitor {
    /// Component health map
    components: Arc<DashMap<String, ComponentHealth>>,
    /// Stale component timeout
    stale_timeout: Duration,
    /// Minimum availability for degraded state
    degraded_threshold: f64,
    /// Minimum availability for critical state
    critical_threshold: f64,
}

impl HealthMonitor {
    /// Create new health monitor
    ///
    /// # Parameters
    /// - `stale_timeout`: Duration before component considered stale (default 30s)
    /// - `degraded_threshold`: Availability threshold for degraded state (default 0.9)
    /// - `critical_threshold`: Availability threshold for critical state (default 0.5)
    pub fn new(stale_timeout: Duration, degraded_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            components: Arc::new(DashMap::new()),
            stale_timeout,
            degraded_threshold,
            critical_threshold,
        }
    }

    /// Create health monitor with default thresholds
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30), 0.9, 0.5)
    }

    /// Register a new component
    ///
    /// # Parameters
    /// - `name`: Component identifier
    /// - `weight`: Component criticality (0.0 = non-critical, 1.0 = critical)
    pub fn register_component(&self, name: impl Into<String>, weight: f64) {
        let name = name.into();
        self.components.insert(name, ComponentHealth::new(weight));
    }

    /// Update component health status
    ///
    /// # Parameters
    /// - `name`: Component identifier
    /// - `status`: New health status
    ///
    /// # Returns
    /// - `Ok(())` if component exists
    /// - `Err(String)` if component not registered
    pub fn update_health(&self, name: &str, status: HealthStatus) -> Result<(), String> {
        self.components
            .get_mut(name)
            .map(|mut health| health.update_status(status))
            .ok_or_else(|| format!("Component '{}' not registered", name))
    }

    /// Mark component as healthy (convenience method)
    pub fn mark_healthy(&self, name: &str) -> Result<(), String> {
        self.update_health(name, HealthStatus::Healthy)
    }

    /// Mark component as degraded (convenience method)
    pub fn mark_degraded(&self, name: &str) -> Result<(), String> {
        self.update_health(name, HealthStatus::Degraded)
    }

    /// Mark component as unhealthy (convenience method)
    pub fn mark_unhealthy(&self, name: &str) -> Result<(), String> {
        self.update_health(name, HealthStatus::Unhealthy)
    }

    /// Get component health
    pub fn get_health(&self, name: &str) -> Option<ComponentHealth> {
        self.components.get(name).map(|health| health.clone())
    }

    /// Calculate system-wide availability
    ///
    /// # Formula
    /// ```text
    /// A = (1/N) Σᵢ wᵢ · hᵢ
    /// ```
    ///
    /// where wᵢ is normalized so Σᵢ wᵢ = N
    pub fn system_availability(&self) -> f64 {
        if self.components.is_empty() {
            return 1.0; // No components = healthy system
        }

        let mut total_weighted_health = 0.0;
        let mut total_weight = 0.0;

        for entry in self.components.iter() {
            let health = entry.value();

            // Mark stale components as unhealthy
            let effective_status = if health.is_stale(self.stale_timeout) {
                HealthStatus::Unhealthy
            } else {
                health.status
            };

            total_weighted_health += health.weight * effective_status.availability();
            total_weight += health.weight;
        }

        if total_weight > 0.0 {
            total_weighted_health / total_weight
        } else {
            1.0 // All components have zero weight
        }
    }

    /// Get global system state
    pub fn system_state(&self) -> SystemState {
        let availability = self.system_availability();

        if availability >= self.degraded_threshold {
            SystemState::Running
        } else if availability >= self.critical_threshold {
            SystemState::Degraded
        } else {
            SystemState::Critical
        }
    }

    /// Get list of unhealthy components
    pub fn unhealthy_components(&self) -> Vec<String> {
        self.components
            .iter()
            .filter(|entry| {
                let health = entry.value();
                health.status == HealthStatus::Unhealthy || health.is_stale(self.stale_timeout)
            })
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get list of degraded components
    pub fn degraded_components(&self) -> Vec<String> {
        self.components
            .iter()
            .filter(|entry| entry.value().status == HealthStatus::Degraded)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get health report
    pub fn health_report(&self) -> HealthReport {
        let availability = self.system_availability();
        let state = self.system_state();

        let unhealthy = self.unhealthy_components();
        let degraded = self.degraded_components();

        let total_components = self.components.len();
        let healthy_count = total_components - unhealthy.len() - degraded.len();

        HealthReport {
            availability,
            state,
            total_components,
            healthy_count,
            degraded_count: degraded.len(),
            unhealthy_count: unhealthy.len(),
            unhealthy_components: unhealthy,
            degraded_components: degraded,
        }
    }

    /// Reset all component health (for testing)
    pub fn reset(&self) {
        self.components.clear();
    }
}

/// Health report summary
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// System-wide availability (0.0 to 1.0)
    pub availability: f64,
    /// Global system state
    pub state: SystemState,
    /// Total number of components
    pub total_components: usize,
    /// Number of healthy components
    pub healthy_count: usize,
    /// Number of degraded components
    pub degraded_count: usize,
    /// Number of unhealthy components
    pub unhealthy_count: usize,
    /// List of unhealthy component names
    pub unhealthy_components: Vec<String>,
    /// List of degraded component names
    pub degraded_components: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_availability() {
        assert_eq!(HealthStatus::Healthy.availability(), 1.0);
        assert_eq!(HealthStatus::Degraded.availability(), 0.5);
        assert_eq!(HealthStatus::Unhealthy.availability(), 0.0);
    }

    #[test]
    fn test_component_health_creation() {
        let health = ComponentHealth::new(0.8);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.weight, 0.8);
        assert_eq!(health.failure_count, 0);
        assert_eq!(health.total_failures, 0);
    }

    #[test]
    fn test_component_health_update() {
        let mut health = ComponentHealth::new(1.0);

        health.update_status(HealthStatus::Degraded);
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.failure_count, 1);
        assert_eq!(health.total_failures, 1);

        health.update_status(HealthStatus::Healthy);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.failure_count, 0);
        assert_eq!(health.total_failures, 1);
    }

    #[test]
    fn test_health_monitor_registration() {
        let monitor = HealthMonitor::default();
        monitor.register_component("test_component", 1.0);

        let health = monitor.get_health("test_component");
        assert!(health.is_some());
        assert_eq!(health.unwrap().status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_monitor_updates() {
        let monitor = HealthMonitor::default();
        monitor.register_component("comp1", 1.0);

        assert!(monitor.mark_healthy("comp1").is_ok());
        assert!(monitor.mark_degraded("comp1").is_ok());
        assert!(monitor.mark_unhealthy("comp1").is_ok());

        // Non-existent component
        assert!(monitor.mark_healthy("nonexistent").is_err());
    }

    #[test]
    fn test_system_availability_all_healthy() {
        let monitor = HealthMonitor::default();
        monitor.register_component("comp1", 1.0);
        monitor.register_component("comp2", 1.0);
        monitor.register_component("comp3", 1.0);

        assert_eq!(monitor.system_availability(), 1.0);
        assert_eq!(monitor.system_state(), SystemState::Running);
    }

    #[test]
    fn test_system_availability_mixed() {
        let monitor = HealthMonitor::default();
        monitor.register_component("comp1", 1.0);
        monitor.register_component("comp2", 1.0);
        monitor.register_component("comp3", 1.0);

        monitor.mark_degraded("comp1").unwrap();
        monitor.mark_unhealthy("comp2").unwrap();

        // A = (1.0*0.5 + 1.0*0.0 + 1.0*1.0) / 3.0 = 0.5
        let availability = monitor.system_availability();
        assert!((availability - 0.5).abs() < 1e-6);
        assert_eq!(monitor.system_state(), SystemState::Degraded);
    }

    #[test]
    fn test_system_state_transitions() {
        let monitor = HealthMonitor::default();
        monitor.register_component("comp1", 1.0);
        monitor.register_component("comp2", 1.0);

        // All healthy: Running
        assert_eq!(monitor.system_state(), SystemState::Running);

        // One degraded: Degraded (A = 0.75)
        monitor.mark_degraded("comp1").unwrap();
        assert_eq!(monitor.system_state(), SystemState::Degraded);

        // One unhealthy: Critical (A = 0.5)
        monitor.mark_unhealthy("comp1").unwrap();
        assert_eq!(monitor.system_state(), SystemState::Degraded);

        // Both unhealthy: Critical (A = 0.0)
        monitor.mark_unhealthy("comp2").unwrap();
        assert_eq!(monitor.system_state(), SystemState::Critical);
    }

    #[test]
    fn test_health_report() {
        let monitor = HealthMonitor::default();
        monitor.register_component("comp1", 1.0);
        monitor.register_component("comp2", 1.0);
        monitor.register_component("comp3", 1.0);

        monitor.mark_degraded("comp1").unwrap();
        monitor.mark_unhealthy("comp2").unwrap();

        let report = monitor.health_report();
        assert_eq!(report.total_components, 3);
        assert_eq!(report.healthy_count, 1);
        assert_eq!(report.degraded_count, 1);
        assert_eq!(report.unhealthy_count, 1);
        assert_eq!(report.state, SystemState::Degraded);
    }

    #[test]
    fn test_weighted_availability() {
        let monitor = HealthMonitor::default();
        monitor.register_component("critical", 1.0);
        monitor.register_component("noncritical", 0.1);

        // Only non-critical fails
        monitor.mark_unhealthy("noncritical").unwrap();

        // A = (1.0*1.0 + 0.1*0.0) / 1.1 ≈ 0.909
        let availability = monitor.system_availability();
        assert!(availability > 0.9); // Still in Running state
        assert_eq!(monitor.system_state(), SystemState::Running);
    }
}
