//! Circuit Breaker Pattern for Cascading Failure Prevention
//!
//! This module implements the circuit breaker pattern to isolate failing components and
//! prevent cascading failures across the system.
//!
//! # State Machine
//!
//! ```text
//!                    failure_threshold exceeded
//!                    ┌─────────────────────┐
//!                    │                     │
//!                    ▼                     │
//!              ┌──────────┐          ┌──────────┐
//!              │  Closed  │          │   Open   │
//!              │ (normal) │          │(blocking)│
//!              └──────────┘          └──────────┘
//!                    ▲                     │
//!                    │                     │
//!                    │    recovery_timeout │
//!                    │                     │
//!                    │                     ▼
//!                    │              ┌──────────┐
//!                    │              │HalfOpen  │
//!                    │              │ (trial)  │
//!                    │              └──────────┘
//!                    │                     │
//!                    │  success            │ failure
//!                    └─────────────────────┘
//! ```
//!
//! # Mathematical Model
//!
//! The circuit breaker uses exponential moving average to track failure rate:
//!
//! ```text
//! λ(t) = α · f(t) + (1-α) · λ(t-1)
//! ```
//!
//! where:
//! - λ(t) = failure rate at time t
//! - f(t) = current operation outcome (0 = success, 1 = failure)
//! - α = smoothing factor (default 0.1)
//!
//! Circuit opens when: λ(t) > threshold

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit closed, requests pass through normally
    Closed,
    /// Circuit open, requests blocked immediately
    Open,
    /// Circuit half-open, single trial request allowed
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold (0.0 to 1.0) before opening
    pub failure_threshold: f64,
    /// Number of consecutive failures before opening (alternative to rate-based)
    pub consecutive_failure_threshold: u32,
    /// Duration to wait before attempting recovery (HalfOpen)
    pub recovery_timeout: Duration,
    /// Exponential moving average smoothing factor
    pub ema_alpha: f64,
    /// Minimum number of calls before circuit can open
    pub min_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 0.5,
            consecutive_failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            ema_alpha: 0.1,
            min_calls: 10,
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current state
    pub state: CircuitState,
    /// Total successful calls
    pub success_count: u64,
    /// Total failed calls
    pub failure_count: u64,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Current failure rate (EMA)
    pub failure_rate: f64,
    /// Time circuit opened (if open)
    pub opened_at: Option<Instant>,
    /// Time since last state transition
    pub last_transition: Instant,
}

/// Circuit breaker for fault isolation
///
/// Wraps fallible operations and tracks their success/failure rate.
/// Opens the circuit when failure rate exceeds threshold, preventing
/// cascading failures to downstream components.
pub struct CircuitBreaker {
    /// Configuration
    config: CircuitBreakerConfig,
    /// Internal state (protected by mutex)
    state: Arc<Mutex<CircuitBreakerState>>,
}

/// Internal mutable state
struct CircuitBreakerState {
    /// Current circuit state
    state: CircuitState,
    /// Success counter
    success_count: u64,
    /// Failure counter
    failure_count: u64,
    /// Consecutive failures
    consecutive_failures: u32,
    /// Exponential moving average of failure rate
    failure_rate: f64,
    /// Time circuit opened
    opened_at: Option<Instant>,
    /// Last state transition
    last_transition: Instant,
}

impl CircuitBreaker {
    /// Create new circuit breaker with configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(CircuitBreakerState {
                state: CircuitState::Closed,
                success_count: 0,
                failure_count: 0,
                consecutive_failures: 0,
                failure_rate: 0.0,
                opened_at: None,
                last_transition: Instant::now(),
            })),
        }
    }

    /// Create circuit breaker with default configuration
    pub fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Execute operation through circuit breaker
    ///
    /// # Returns
    /// - `Ok(T)` if operation succeeds
    /// - `Err(CircuitBreakerError::Open)` if circuit is open
    /// - `Err(CircuitBreakerError::Operation(E))` if operation fails
    pub fn call<T, E, F>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        // Check if circuit allows call
        let mut state = self.state.lock().unwrap();

        match state.state {
            CircuitState::Open => {
                // Check if recovery timeout elapsed
                if let Some(opened_at) = state.opened_at {
                    if opened_at.elapsed() >= self.config.recovery_timeout {
                        // Transition to HalfOpen for trial
                        state.state = CircuitState::HalfOpen;
                        state.last_transition = Instant::now();
                        drop(state); // Release lock before operation
                    } else {
                        // Circuit still open, reject immediately
                        return Err(CircuitBreakerError::Open);
                    }
                } else {
                    return Err(CircuitBreakerError::Open);
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {
                drop(state); // Release lock before operation
            }
        }

        // Execute operation
        let result = operation();

        // Record result and update state
        let mut state = self.state.lock().unwrap();
        match result {
            Ok(value) => {
                self.record_success(&mut state);
                Ok(value)
            }
            Err(error) => {
                self.record_failure(&mut state);
                Err(CircuitBreakerError::Operation(error))
            }
        }
    }

    /// Record successful operation
    fn record_success(&self, state: &mut CircuitBreakerState) {
        state.success_count += 1;
        state.consecutive_failures = 0;

        // Update failure rate EMA
        state.failure_rate = (1.0 - self.config.ema_alpha) * state.failure_rate;

        match state.state {
            CircuitState::HalfOpen => {
                // Trial succeeded, close circuit
                state.state = CircuitState::Closed;
                state.opened_at = None;
                state.last_transition = Instant::now();
            }
            _ => {}
        }
    }

    /// Record failed operation
    fn record_failure(&self, state: &mut CircuitBreakerState) {
        state.failure_count += 1;
        state.consecutive_failures += 1;

        // Update failure rate EMA
        state.failure_rate =
            self.config.ema_alpha + (1.0 - self.config.ema_alpha) * state.failure_rate;

        // Check if circuit should open
        let total_calls = state.success_count + state.failure_count;
        let should_open = total_calls >= self.config.min_calls as u64
            && (state.failure_rate > self.config.failure_threshold
                || state.consecutive_failures >= self.config.consecutive_failure_threshold);

        match state.state {
            CircuitState::Closed => {
                if should_open {
                    // Open circuit
                    state.state = CircuitState::Open;
                    state.opened_at = Some(Instant::now());
                    state.last_transition = Instant::now();
                }
            }
            CircuitState::HalfOpen => {
                // Trial failed, reopen circuit
                state.state = CircuitState::Open;
                state.opened_at = Some(Instant::now());
                state.last_transition = Instant::now();
            }
            CircuitState::Open => {}
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.lock().unwrap().state
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        let state = self.state.lock().unwrap();
        CircuitBreakerStats {
            state: state.state,
            success_count: state.success_count,
            failure_count: state.failure_count,
            consecutive_failures: state.consecutive_failures,
            failure_rate: state.failure_rate,
            opened_at: state.opened_at,
            last_transition: state.last_transition,
        }
    }

    /// Reset circuit breaker (for testing)
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.state = CircuitState::Closed;
        state.success_count = 0;
        state.failure_count = 0;
        state.consecutive_failures = 0;
        state.failure_rate = 0.0;
        state.opened_at = None;
        state.last_transition = Instant::now();
    }

    /// Manually open circuit (for testing)
    pub fn force_open(&self) {
        let mut state = self.state.lock().unwrap();
        state.state = CircuitState::Open;
        state.opened_at = Some(Instant::now());
        state.last_transition = Instant::now();
    }
}

/// Circuit breaker error
#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    /// Circuit is open, request rejected
    Open,
    /// Operation failed
    Operation(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::Open => write!(f, "Circuit breaker is open"),
            CircuitBreakerError::Operation(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::error::Error> std::error::Error for CircuitBreakerError<E> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_circuit_breaker_closed_success() {
        let breaker = CircuitBreaker::default();

        let result = breaker.call(|| Ok::<i32, String>(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_consecutive_failures() {
        let config = CircuitBreakerConfig {
            consecutive_failure_threshold: 3,
            min_calls: 0,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Fail 3 times
        for _ in 0..3 {
            let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        }

        // Circuit should be open
        assert_eq!(breaker.state(), CircuitState::Open);

        // Next call should be rejected immediately
        let result = breaker.call(|| Ok::<i32, String>(42));
        assert!(matches!(result, Err(CircuitBreakerError::Open)));
    }

    #[test]
    fn test_circuit_breaker_rate_based_opening() {
        let config = CircuitBreakerConfig {
            failure_threshold: 0.5,
            min_calls: 10,
            ema_alpha: 1.0, // Immediate response
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // 5 successes, 5 failures
        for i in 0..10 {
            if i % 2 == 0 {
                let _ = breaker.call(|| Ok::<i32, String>(42));
            } else {
                let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
            }
        }

        // Circuit should be open (failure rate = 0.5)
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            consecutive_failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            min_calls: 0,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Fail twice to open circuit
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for recovery timeout
        thread::sleep(Duration::from_millis(150));

        // Circuit should transition to HalfOpen and allow trial
        let result = breaker.call(|| Ok::<i32, String>(42));
        assert!(result.is_ok());
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let config = CircuitBreakerConfig {
            consecutive_failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            min_calls: 0,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open circuit
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));

        // Wait for recovery
        thread::sleep(Duration::from_millis(150));

        // Trial fails, circuit reopens
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_stats() {
        let breaker = CircuitBreaker::default();

        let _ = breaker.call(|| Ok::<i32, String>(1));
        let _ = breaker.call(|| Ok::<i32, String>(2));
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));

        let stats = breaker.stats();
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
        assert_eq!(stats.consecutive_failures, 1);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let breaker = CircuitBreaker::default();

        // Generate some activity
        let _ = breaker.call(|| Ok::<i32, String>(1));
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));

        breaker.reset();

        let stats = breaker.stats();
        assert_eq!(stats.state, CircuitState::Closed);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
    }

    #[test]
    fn test_circuit_breaker_ema_smoothing() {
        let config = CircuitBreakerConfig {
            ema_alpha: 0.5, // 50% smoothing
            failure_threshold: 0.6,
            min_calls: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Alternate success/failure
        let _ = breaker.call(|| Err::<i32, String>("error".to_string()));
        let stats1 = breaker.stats();
        assert!(stats1.failure_rate > 0.0);

        let _ = breaker.call(|| Ok::<i32, String>(1));
        let stats2 = breaker.stats();
        assert!(stats2.failure_rate < stats1.failure_rate);
    }
}
