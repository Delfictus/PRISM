//! Configuration management for the ingestion system
//!
//! Provides structured configuration for engines, sources, and policies

use super::error::{CircuitBreaker, RetryPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Complete configuration for an ingestion engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Engine settings
    pub engine: EngineConfig,
    /// Retry policy settings
    pub retry: RetryConfig,
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
    /// Source-specific configurations
    #[serde(default)]
    pub sources: HashMap<String, SourceConfig>,
}

/// Engine-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Size of the async channel buffer
    #[serde(default = "default_channel_size")]
    pub channel_size: usize,
    /// Size of the historical circular buffer
    #[serde(default = "default_history_size")]
    pub history_size: usize,
    /// Maximum number of concurrent sources
    #[serde(default = "default_max_sources")]
    pub max_sources: usize,
}

fn default_channel_size() -> usize {
    1000
}

fn default_history_size() -> usize {
    10000
}

fn default_max_sources() -> usize {
    100
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            channel_size: default_channel_size(),
            history_size: default_history_size(),
            max_sources: default_max_sources(),
        }
    }
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    #[serde(default = "default_max_attempts")]
    pub max_attempts: usize,
    /// Initial backoff delay in milliseconds
    #[serde(default = "default_initial_backoff")]
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    #[serde(default = "default_max_backoff")]
    pub max_backoff_ms: u64,
    /// Backoff multiplier (exponential backoff)
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
}

fn default_max_attempts() -> usize {
    3
}

fn default_initial_backoff() -> u64 {
    100
}

fn default_max_backoff() -> u64 {
    5000
}

fn default_backoff_multiplier() -> f64 {
    2.0
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: default_max_attempts(),
            initial_backoff_ms: default_initial_backoff(),
            max_backoff_ms: default_max_backoff(),
            backoff_multiplier: default_backoff_multiplier(),
        }
    }
}

impl From<RetryConfig> for RetryPolicy {
    fn from(config: RetryConfig) -> Self {
        Self {
            max_attempts: config.max_attempts,
            initial_backoff_ms: config.initial_backoff_ms,
            max_backoff_ms: config.max_backoff_ms,
            backoff_multiplier: config.backoff_multiplier,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive errors before opening
    #[serde(default = "default_error_threshold")]
    pub error_threshold: usize,
    /// Time in milliseconds before attempting to close
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_error_threshold() -> usize {
    5
}

fn default_timeout() -> u64 {
    30000
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            error_threshold: default_error_threshold(),
            timeout_ms: default_timeout(),
        }
    }
}

impl From<CircuitBreakerConfig> for CircuitBreaker {
    fn from(config: CircuitBreakerConfig) -> Self {
        Self::new(config.error_threshold, config.timeout_ms)
    }
}

/// Source-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Enable/disable this source
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Source type (e.g., "synthetic", "alpaca", "sensor")
    pub source_type: String,
    /// Source-specific parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Custom retry policy for this source (overrides global)
    pub retry: Option<RetryConfig>,
    /// Custom circuit breaker for this source (overrides global)
    pub circuit_breaker: Option<CircuitBreakerConfig>,
}

fn default_enabled() -> bool {
    true
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            sources: HashMap::new(),
        }
    }
}

impl IngestionConfig {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a TOML file
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load configuration from a JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn to_toml_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Save configuration to a JSON file
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.engine.channel_size == 0 {
            return Err("channel_size must be greater than 0".to_string());
        }

        if self.engine.history_size == 0 {
            return Err("history_size must be greater than 0".to_string());
        }

        if self.retry.max_attempts == 0 {
            return Err("max_attempts must be greater than 0".to_string());
        }

        if self.retry.initial_backoff_ms == 0 {
            return Err("initial_backoff_ms must be greater than 0".to_string());
        }

        if self.circuit_breaker.error_threshold == 0 {
            return Err("error_threshold must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Create a high-performance configuration preset
    pub fn high_performance() -> Self {
        Self {
            engine: EngineConfig {
                channel_size: 10000,
                history_size: 100000,
                max_sources: 200,
            },
            retry: RetryConfig {
                max_attempts: 5,
                initial_backoff_ms: 50,
                max_backoff_ms: 10000,
                backoff_multiplier: 2.0,
            },
            circuit_breaker: CircuitBreakerConfig {
                error_threshold: 10,
                timeout_ms: 60000,
            },
            sources: HashMap::new(),
        }
    }

    /// Create a low-latency configuration preset
    pub fn low_latency() -> Self {
        Self {
            engine: EngineConfig {
                channel_size: 5000,
                history_size: 50000,
                max_sources: 100,
            },
            retry: RetryConfig {
                max_attempts: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 1000,
                backoff_multiplier: 1.5,
            },
            circuit_breaker: CircuitBreakerConfig {
                error_threshold: 5,
                timeout_ms: 10000,
            },
            sources: HashMap::new(),
        }
    }

    /// Create a conservative configuration preset (high reliability)
    pub fn conservative() -> Self {
        Self {
            engine: EngineConfig {
                channel_size: 2000,
                history_size: 20000,
                max_sources: 50,
            },
            retry: RetryConfig {
                max_attempts: 10,
                initial_backoff_ms: 200,
                max_backoff_ms: 30000,
                backoff_multiplier: 2.5,
            },
            circuit_breaker: CircuitBreakerConfig {
                error_threshold: 3,
                timeout_ms: 120000,
            },
            sources: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IngestionConfig::default();
        assert_eq!(config.engine.channel_size, 1000);
        assert_eq!(config.retry.max_attempts, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_performance_preset() {
        let config = IngestionConfig::high_performance();
        assert_eq!(config.engine.channel_size, 10000);
        assert_eq!(config.retry.max_attempts, 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_low_latency_preset() {
        let config = IngestionConfig::low_latency();
        assert_eq!(config.retry.initial_backoff_ms, 10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_conservative_preset() {
        let config = IngestionConfig::conservative();
        assert_eq!(config.retry.max_attempts, 10);
        assert_eq!(config.circuit_breaker.error_threshold, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation() {
        let mut config = IngestionConfig::default();
        assert!(config.validate().is_ok());

        config.engine.channel_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_retry_policy_conversion() {
        let config = RetryConfig::default();
        let policy: RetryPolicy = config.into();
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.initial_backoff_ms, 100);
    }

    #[test]
    fn test_circuit_breaker_conversion() {
        let config = CircuitBreakerConfig::default();
        let cb: CircuitBreaker = config.into();
        assert_eq!(cb.error_threshold, 5);
    }
}
