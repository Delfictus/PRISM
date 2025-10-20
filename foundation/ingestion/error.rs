//! Error types and handling for the ingestion system

use std::fmt;

/// Errors that can occur during data ingestion
#[derive(Debug, Clone)]
pub enum IngestionError {
    /// Source connection failed
    ConnectionFailed {
        source: String,
        reason: String,
        retryable: bool,
    },
    /// Failed to read data from source
    ReadFailed {
        source: String,
        reason: String,
        retryable: bool,
    },
    /// Data parsing/validation failed
    ParseError { source: String, reason: String },
    /// Source exceeded error threshold (circuit breaker)
    CircuitBreakerOpen {
        source: String,
        error_count: usize,
        threshold: usize,
    },
    /// Channel closed unexpectedly
    ChannelClosed { source: String },
    /// Timeout waiting for data
    Timeout { source: String, timeout_ms: u64 },
    /// Configuration error
    ConfigError { reason: String },
}

impl fmt::Display for IngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestionError::ConnectionFailed {
                source,
                reason,
                retryable,
            } => {
                write!(
                    f,
                    "Connection failed for '{}': {} (retryable: {})",
                    source, reason, retryable
                )
            }
            IngestionError::ReadFailed {
                source,
                reason,
                retryable,
            } => {
                write!(
                    f,
                    "Read failed for '{}': {} (retryable: {})",
                    source, reason, retryable
                )
            }
            IngestionError::ParseError { source, reason } => {
                write!(f, "Parse error for '{}': {}", source, reason)
            }
            IngestionError::CircuitBreakerOpen {
                source,
                error_count,
                threshold,
            } => {
                write!(
                    f,
                    "Circuit breaker open for '{}': {} errors (threshold: {})",
                    source, error_count, threshold
                )
            }
            IngestionError::ChannelClosed { source } => {
                write!(f, "Channel closed for '{}'", source)
            }
            IngestionError::Timeout { source, timeout_ms } => {
                write!(f, "Timeout for '{}' after {}ms", source, timeout_ms)
            }
            IngestionError::ConfigError { reason } => {
                write!(f, "Configuration error: {}", reason)
            }
        }
    }
}

impl std::error::Error for IngestionError {}

impl IngestionError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            IngestionError::ConnectionFailed { retryable, .. } => *retryable,
            IngestionError::ReadFailed { retryable, .. } => *retryable,
            IngestionError::Timeout { .. } => true,
            IngestionError::CircuitBreakerOpen { .. } => false,
            IngestionError::ChannelClosed { .. } => false,
            IngestionError::ParseError { .. } => false,
            IngestionError::ConfigError { .. } => false,
        }
    }

    /// Get the source name associated with this error
    pub fn source_name(&self) -> Option<&str> {
        match self {
            IngestionError::ConnectionFailed { source, .. }
            | IngestionError::ReadFailed { source, .. }
            | IngestionError::ParseError { source, .. }
            | IngestionError::CircuitBreakerOpen { source, .. }
            | IngestionError::ChannelClosed { source }
            | IngestionError::Timeout { source, .. } => Some(source),
            IngestionError::ConfigError { .. } => None,
        }
    }
}

/// Convert from anyhow::Error to IngestionError
impl From<anyhow::Error> for IngestionError {
    fn from(err: anyhow::Error) -> Self {
        IngestionError::ReadFailed {
            source: "unknown".to_string(),
            reason: err.to_string(),
            retryable: true,
        }
    }
}

/// Retry policy for handling transient failures
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier (exponential backoff)
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Calculate backoff delay for the given attempt number
    pub fn backoff_delay(&self, attempt: usize) -> u64 {
        if attempt == 0 {
            return self.initial_backoff_ms;
        }

        let delay =
            self.initial_backoff_ms as f64 * self.backoff_multiplier.powi(attempt as i32 - 1);

        delay.min(self.max_backoff_ms as f64) as u64
    }

    /// Create a retry policy with no retries
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
            backoff_multiplier: 1.0,
        }
    }

    /// Create an aggressive retry policy
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 10000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Circuit breaker for protecting against cascading failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Number of consecutive errors before opening
    pub error_threshold: usize,
    /// Time in milliseconds before attempting to close
    pub timeout_ms: u64,
    /// Current error count
    error_count: usize,
    /// Circuit breaker state
    state: CircuitBreakerState,
    /// Last error timestamp
    last_error_time: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are blocked
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            error_threshold: 5,
            timeout_ms: 30000, // 30 seconds
            error_count: 0,
            state: CircuitBreakerState::Closed,
            last_error_time: None,
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(error_threshold: usize, timeout_ms: u64) -> Self {
        Self {
            error_threshold,
            timeout_ms,
            error_count: 0,
            state: CircuitBreakerState::Closed,
            last_error_time: None,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        self.error_count = 0;
        self.state = CircuitBreakerState::Closed;
        self.last_error_time = None;
    }

    /// Record a failed operation
    pub fn record_failure(&mut self) {
        self.error_count += 1;
        self.last_error_time = Some(std::time::Instant::now());

        if self.error_count >= self.error_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }

    /// Check if the circuit breaker allows requests
    pub fn is_closed(&self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::HalfOpen => true,
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed
                if let Some(last_error) = self.last_error_time {
                    let elapsed = last_error.elapsed().as_millis() as u64;
                    elapsed >= self.timeout_ms
                } else {
                    false
                }
            }
        }
    }

    /// Transition to half-open state (for testing)
    pub fn half_open(&mut self) {
        self.state = CircuitBreakerState::HalfOpen;
        self.error_count = 0;
    }

    /// Get current state
    pub fn state(&self) -> CircuitBreakerState {
        self.state
    }

    /// Get current error count
    pub fn error_count(&self) -> usize {
        self.error_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_backoff() {
        let policy = RetryPolicy::default();

        assert_eq!(policy.backoff_delay(0), 100);
        assert_eq!(policy.backoff_delay(1), 100);
        assert_eq!(policy.backoff_delay(2), 200);
        assert_eq!(policy.backoff_delay(3), 400);
        assert_eq!(policy.backoff_delay(10), 5000); // Capped at max
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let mut cb = CircuitBreaker::new(3, 1000);

        assert!(cb.is_closed());

        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_closed()); // Still closed

        cb.record_failure();
        assert!(!cb.is_closed()); // Now open
    }

    #[test]
    fn test_circuit_breaker_recovers() {
        let mut cb = CircuitBreaker::new(2, 1000);

        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_closed());

        cb.record_success();
        assert!(cb.is_closed());
        assert_eq!(cb.error_count(), 0);
    }

    #[test]
    fn test_ingestion_error_retryable() {
        let err = IngestionError::ConnectionFailed {
            source: "test".to_string(),
            reason: "timeout".to_string(),
            retryable: true,
        };
        assert!(err.is_retryable());

        let err = IngestionError::CircuitBreakerOpen {
            source: "test".to_string(),
            error_count: 5,
            threshold: 3,
        };
        assert!(!err.is_retryable());
    }
}
