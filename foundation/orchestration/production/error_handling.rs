//! Production Error Handling with Graceful Degradation
//!
//! Mission Charlie: Phase 4 (Essential production features)
//!
//! Implements comprehensive error handling, circuit breakers,
//! fallback strategies, and recovery mechanisms for production deployment.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Production-grade Error Handler with Multiple Fallback Strategies
pub struct ProductionErrorHandler {
    circuit_breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
    fallback_registry: Arc<Mutex<FallbackRegistry>>,
    error_stats: Arc<Mutex<ErrorStatistics>>,
}

impl ProductionErrorHandler {
    pub fn new() -> Self {
        Self {
            circuit_breakers: Arc::new(Mutex::new(HashMap::new())),
            fallback_registry: Arc::new(Mutex::new(FallbackRegistry::new())),
            error_stats: Arc::new(Mutex::new(ErrorStatistics::new())),
        }
    }

    /// Handle LLM failures with sophisticated recovery strategies
    pub fn handle_llm_failure(&self, llm_name: &str, error: &str) -> Result<RecoveryAction> {
        // Record error statistics
        self.record_error(llm_name, error);

        // Check circuit breaker
        let mut breakers = self.circuit_breakers.lock().unwrap();
        let breaker = breakers
            .entry(llm_name.to_string())
            .or_insert_with(CircuitBreaker::new);

        if breaker.is_open() {
            return Ok(RecoveryAction::UseAlternateLLM);
        }

        // Classify error and determine recovery
        let recovery = self.classify_error(error);

        // Update circuit breaker based on error
        if matches!(recovery, RecoveryAction::FallbackToHeuristic) {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }

        Ok(recovery)
    }

    /// Classify error and determine optimal recovery strategy
    fn classify_error(&self, error: &str) -> RecoveryAction {
        // Rate limiting errors
        if error.contains("rate_limit") || error.contains("429") {
            return RecoveryAction::RetryAfterDelay(Duration::from_secs(60));
        }

        // Timeout errors
        if error.contains("timeout") || error.contains("timed out") {
            return RecoveryAction::RetryWithIncreasedTimeout(Duration::from_secs(120));
        }

        // Authentication errors
        if error.contains("auth") || error.contains("401") || error.contains("403") {
            return RecoveryAction::RefreshCredentials;
        }

        // Service unavailable
        if error.contains("503") || error.contains("unavailable") {
            return RecoveryAction::UseAlternateLLM;
        }

        // Model overloaded
        if error.contains("overloaded") || error.contains("capacity") {
            return RecoveryAction::UseCachedResponse;
        }

        // Content policy violations
        if error.contains("content_policy") || error.contains("moderation") {
            return RecoveryAction::ReformulateQuery;
        }

        // Network errors
        if error.contains("network") || error.contains("connection") {
            return RecoveryAction::RetryWithBackoff {
                initial_delay: Duration::from_secs(1),
                max_retries: 3,
            };
        }

        // Invalid request
        if error.contains("invalid") || error.contains("400") {
            return RecoveryAction::ValidateAndRetry;
        }

        // Default: fallback to heuristic
        RecoveryAction::FallbackToHeuristic
    }

    /// Record error for statistics
    fn record_error(&self, llm_name: &str, error: &str) {
        let mut stats = self.error_stats.lock().unwrap();
        stats.record(llm_name, error);
    }

    /// Get error statistics for monitoring
    pub fn get_error_stats(&self) -> HashMap<String, ErrorStats> {
        let stats = self.error_stats.lock().unwrap();
        stats.get_all()
    }

    /// Register a fallback handler
    pub fn register_fallback<F>(&self, error_type: &str, handler: F)
    where
        F: Fn(&str) -> Result<String> + Send + Sync + 'static,
    {
        let mut registry = self.fallback_registry.lock().unwrap();
        registry.register(error_type.to_string(), Box::new(handler));
    }

    /// Execute fallback strategy
    pub fn execute_fallback(&self, error_type: &str, context: &str) -> Result<String> {
        let registry = self.fallback_registry.lock().unwrap();
        registry.execute(error_type, context)
    }

    /// Reset circuit breaker for an LLM
    pub fn reset_circuit_breaker(&self, llm_name: &str) {
        let mut breakers = self.circuit_breakers.lock().unwrap();
        if let Some(breaker) = breakers.get_mut(llm_name) {
            breaker.reset();
        }
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self, llm_name: &str) -> CircuitBreakerStatus {
        let breakers = self.circuit_breakers.lock().unwrap();
        breakers
            .get(llm_name)
            .map(|b| b.status())
            .unwrap_or(CircuitBreakerStatus::Closed)
    }
}

/// Circuit Breaker Pattern Implementation
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    state: CircuitBreakerState,
    failure_threshold: usize,
    timeout: Duration,
    success_threshold: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Blocking requests
    HalfOpen, // Testing if service recovered
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerStatus {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            state: CircuitBreakerState::Closed,
            failure_threshold: 5,             // Open after 5 failures
            timeout: Duration::from_secs(60), // Try again after 60s
            success_threshold: 2,             // Close after 2 successes in half-open
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Failure in half-open state: go back to open
                self.state = CircuitBreakerState::Open;
                self.success_count = 0;
            }
            CircuitBreakerState::Open => {}
        }
    }

    fn record_success(&mut self) {
        self.success_count += 1;

        match self.state {
            CircuitBreakerState::HalfOpen => {
                if self.success_count >= self.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitBreakerState::Closed => {
                // Reset failure count on success
                if self.success_count >= 10 {
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitBreakerState::Open => {}
        }
    }

    fn is_open(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.timeout {
                        self.state = CircuitBreakerState::HalfOpen;
                        self.success_count = 0;
                        return false;
                    }
                }
                true
            }
            CircuitBreakerState::HalfOpen => false,
            CircuitBreakerState::Closed => false,
        }
    }

    fn reset(&mut self) {
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
        self.state = CircuitBreakerState::Closed;
    }

    fn status(&self) -> CircuitBreakerStatus {
        match self.state {
            CircuitBreakerState::Closed => CircuitBreakerStatus::Closed,
            CircuitBreakerState::Open => CircuitBreakerStatus::Open,
            CircuitBreakerState::HalfOpen => CircuitBreakerStatus::HalfOpen,
        }
    }
}

/// Recovery Action Strategies
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry after specified delay
    RetryAfterDelay(Duration),

    /// Retry with increased timeout
    RetryWithIncreasedTimeout(Duration),

    /// Retry with exponential backoff
    RetryWithBackoff {
        initial_delay: Duration,
        max_retries: usize,
    },

    /// Use cached response if available
    UseCachedResponse,

    /// Switch to alternate LLM
    UseAlternateLLM,

    /// Fallback to heuristic/rule-based approach
    FallbackToHeuristic,

    /// Refresh authentication credentials
    RefreshCredentials,

    /// Reformulate query to avoid content policy
    ReformulateQuery,

    /// Validate request and retry
    ValidateAndRetry,
}

/// Fallback Function Registry
struct FallbackRegistry {
    handlers: HashMap<String, Box<dyn Fn(&str) -> Result<String> + Send + Sync>>,
}

impl FallbackRegistry {
    fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    fn register(
        &mut self,
        error_type: String,
        handler: Box<dyn Fn(&str) -> Result<String> + Send + Sync>,
    ) {
        self.handlers.insert(error_type, handler);
    }

    fn execute(&self, error_type: &str, context: &str) -> Result<String> {
        if let Some(handler) = self.handlers.get(error_type) {
            handler(context)
        } else {
            Err(anyhow!(
                "No fallback handler registered for: {}",
                error_type
            ))
        }
    }
}

/// Error Statistics Tracking
#[derive(Debug, Clone)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub last_error_time: Option<Instant>,
    pub error_types: HashMap<String, usize>,
}

struct ErrorStatistics {
    stats: HashMap<String, ErrorStats>,
}

impl ErrorStatistics {
    fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    fn record(&mut self, llm_name: &str, error: &str) {
        // Classify error type first
        let error_type = Self::classify_error_type(error);

        let entry = self
            .stats
            .entry(llm_name.to_string())
            .or_insert_with(|| ErrorStats {
                total_errors: 0,
                last_error_time: None,
                error_types: HashMap::new(),
            });

        entry.total_errors += 1;
        entry.last_error_time = Some(Instant::now());
        *entry.error_types.entry(error_type).or_insert(0) += 1;
    }

    fn classify_error_type(error: &str) -> String {
        if error.contains("rate_limit") || error.contains("429") {
            "rate_limit".to_string()
        } else if error.contains("timeout") {
            "timeout".to_string()
        } else if error.contains("auth") {
            "authentication".to_string()
        } else if error.contains("503") {
            "service_unavailable".to_string()
        } else if error.contains("network") {
            "network".to_string()
        } else {
            "other".to_string()
        }
    }

    fn get_all(&self) -> HashMap<String, ErrorStats> {
        self.stats.clone()
    }
}

/// Monitoring Module with Prometheus-style Metrics
pub struct ProductionMonitoring {
    metrics: Arc<Mutex<MetricsCollector>>,
}

impl ProductionMonitoring {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(MetricsCollector::new())),
        }
    }

    /// Record a metric value
    pub fn record_metric(&self, name: &str, value: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.record(name, value);
    }

    /// Record a counter increment
    pub fn increment_counter(&self, name: &str) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.increment(name);
    }

    /// Record latency measurement
    pub fn record_latency(&self, operation: &str, duration: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.record_latency(operation, duration);
    }

    /// Get all metrics for export
    pub fn get_metrics(&self) -> HashMap<String, MetricValue> {
        let metrics = self.metrics.lock().unwrap();
        metrics.get_all()
    }

    /// Get metrics summary
    pub fn get_summary(&self) -> MetricsSummary {
        let metrics = self.metrics.lock().unwrap();
        metrics.get_summary()
    }
}

/// Metrics Collector
struct MetricsCollector {
    gauges: HashMap<String, f64>,
    counters: HashMap<String, u64>,
    histograms: HashMap<String, Vec<f64>>,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            gauges: HashMap::new(),
            counters: HashMap::new(),
            histograms: HashMap::new(),
        }
    }

    fn record(&mut self, name: &str, value: f64) {
        self.gauges.insert(name.to_string(), value);
    }

    fn increment(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
    }

    fn record_latency(&mut self, operation: &str, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;
        self.histograms
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(ms);
    }

    fn get_all(&self) -> HashMap<String, MetricValue> {
        let mut all = HashMap::new();

        for (name, value) in &self.gauges {
            all.insert(name.clone(), MetricValue::Gauge(*value));
        }

        for (name, value) in &self.counters {
            all.insert(name.clone(), MetricValue::Counter(*value));
        }

        for (name, values) in &self.histograms {
            all.insert(name.clone(), MetricValue::Histogram(values.clone()));
        }

        all
    }

    fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_gauges: self.gauges.len(),
            total_counters: self.counters.len(),
            total_histograms: self.histograms.len(),
            counter_sum: self.counters.values().sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Gauge(f64),
    Counter(u64),
    Histogram(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_gauges: usize,
    pub total_counters: usize,
    pub total_histograms: usize,
    pub counter_sum: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new();

        // Initially closed
        assert!(!breaker.is_open());

        // Record failures to open circuit
        for _ in 0..5 {
            breaker.record_failure();
        }

        assert!(breaker.is_open());
    }

    #[test]
    fn test_error_classification() {
        let handler = ProductionErrorHandler::new();

        let recovery = handler.classify_error("rate_limit exceeded");
        assert!(matches!(recovery, RecoveryAction::RetryAfterDelay(_)));

        let recovery = handler.classify_error("connection timeout");
        assert!(matches!(
            recovery,
            RecoveryAction::RetryWithIncreasedTimeout(_)
        ));
    }

    #[test]
    fn test_metrics_collection() {
        let monitoring = ProductionMonitoring::new();

        monitoring.record_metric("test_gauge", 42.0);
        monitoring.increment_counter("test_counter");
        monitoring.record_latency("test_op", Duration::from_millis(100));

        let summary = monitoring.get_summary();
        assert_eq!(summary.total_gauges, 1);
        assert_eq!(summary.total_counters, 1);
        assert_eq!(summary.total_histograms, 1);
    }
}
