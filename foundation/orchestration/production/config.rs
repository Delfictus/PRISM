//! Production Configuration Management
//!
//! Mission Charlie: Phase 4 (Essential production features)
//!
//! Centralized configuration system for Mission Charlie with
//! environment variable support, validation, and defaults.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Master Configuration for Mission Charlie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionCharlieConfig {
    /// LLM client configurations
    pub llm_config: LLMConfiguration,

    /// Caching configurations
    pub cache_config: CacheConfiguration,

    /// Consensus configurations
    pub consensus_config: ConsensusConfiguration,

    /// Error handling configurations
    pub error_config: ErrorConfiguration,

    /// Logging configurations
    pub logging_config: LoggingConfiguration,

    /// Performance tuning
    pub performance_config: PerformanceConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfiguration {
    /// OpenAI configuration
    pub openai: LLMClientConfig,

    /// Claude configuration
    pub claude: LLMClientConfig,

    /// Gemini configuration
    pub gemini: LLMClientConfig,

    /// Grok configuration
    pub grok: LLMClientConfig,

    /// Global timeout for LLM requests (seconds)
    pub global_timeout_secs: u64,

    /// Maximum retries for failed requests
    pub max_retries: usize,

    /// Enable cost tracking
    pub enable_cost_tracking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMClientConfig {
    /// Enable this LLM client
    pub enabled: bool,

    /// API key (can be loaded from environment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Environment variable name for API key
    pub api_key_env: String,

    /// Model name
    pub model_name: String,

    /// Default temperature
    pub temperature: f64,

    /// Max tokens per request
    pub max_tokens: usize,

    /// Rate limit (requests per minute)
    pub rate_limit_rpm: usize,

    /// Initial quality score for bandit
    pub initial_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfiguration {
    /// Enable semantic caching
    pub enable_semantic_cache: bool,

    /// Similarity threshold for cache hits
    pub similarity_threshold: f64,

    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,

    /// Cache TTL (time-to-live) in seconds
    pub cache_ttl_secs: u64,

    /// LSH hash count for quantum cache
    pub lsh_hash_count: usize,

    /// LSH bucket count
    pub lsh_bucket_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfiguration {
    /// Enable quantum voting consensus
    pub enable_quantum_voting: bool,

    /// Enable thermodynamic consensus
    pub enable_thermodynamic: bool,

    /// Enable neuromorphic spike consensus
    pub enable_neuromorphic: bool,

    /// Minimum LLMs for consensus (2-4)
    pub min_llms_for_consensus: usize,

    /// Confidence threshold for consensus acceptance
    pub confidence_threshold: f64,

    /// Temperature parameter for thermodynamic methods
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorConfiguration {
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,

    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout_secs: u64,

    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,

    /// Fallback to heuristics on total failure
    pub enable_heuristic_fallback: bool,

    /// Maximum error recovery attempts
    pub max_recovery_attempts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfiguration {
    /// Minimum log level
    pub min_level: String,

    /// Enable structured logging
    pub enable_structured_logging: bool,

    /// Enable JSON format
    pub enable_json: bool,

    /// Log buffer size
    pub buffer_size: usize,

    /// Enable request tracing
    pub enable_request_tracing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfiguration {
    /// Enable GPU acceleration where possible
    pub enable_gpu: bool,

    /// Number of worker threads for async operations
    pub worker_threads: usize,

    /// Enable prompt compression (MDL)
    pub enable_prompt_compression: bool,

    /// Compression quality (0-10, higher = more compression)
    pub compression_level: i32,

    /// Enable parallel LLM queries
    pub enable_parallel_queries: bool,

    /// Maximum parallel queries
    pub max_parallel_queries: usize,
}

impl Default for MissionCharlieConfig {
    fn default() -> Self {
        Self {
            llm_config: LLMConfiguration::default(),
            cache_config: CacheConfiguration::default(),
            consensus_config: ConsensusConfiguration::default(),
            error_config: ErrorConfiguration::default(),
            logging_config: LoggingConfiguration::default(),
            performance_config: PerformanceConfiguration::default(),
        }
    }
}

impl Default for LLMConfiguration {
    fn default() -> Self {
        Self {
            openai: LLMClientConfig {
                enabled: true,
                api_key: None,
                api_key_env: "OPENAI_API_KEY".to_string(),
                model_name: "gpt-4".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                rate_limit_rpm: 500,
                initial_quality: 0.8,
            },
            claude: LLMClientConfig {
                enabled: true,
                api_key: None,
                api_key_env: "ANTHROPIC_API_KEY".to_string(),
                model_name: "claude-3-5-sonnet-20250110".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                rate_limit_rpm: 500,
                initial_quality: 0.85,
            },
            gemini: LLMClientConfig {
                enabled: true,
                api_key: None,
                api_key_env: "GEMINI_API_KEY".to_string(),
                model_name: "gemini-2.0-flash-exp".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                rate_limit_rpm: 500,
                initial_quality: 0.75,
            },
            grok: LLMClientConfig {
                enabled: true,
                api_key: None,
                api_key_env: "XAI_API_KEY".to_string(),
                model_name: "grok-2-1212".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                rate_limit_rpm: 500,
                initial_quality: 0.7,
            },
            global_timeout_secs: 60,
            max_retries: 3,
            enable_cost_tracking: true,
        }
    }
}

impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            enable_semantic_cache: true,
            similarity_threshold: 0.85,
            max_cache_size: 10000,
            cache_ttl_secs: 3600,
            lsh_hash_count: 5,
            lsh_bucket_count: 100,
        }
    }
}

impl Default for ConsensusConfiguration {
    fn default() -> Self {
        Self {
            enable_quantum_voting: true,
            enable_thermodynamic: true,
            enable_neuromorphic: false, // Experimental
            min_llms_for_consensus: 3,
            confidence_threshold: 0.7,
            temperature: 1.0,
        }
    }
}

impl Default for ErrorConfiguration {
    fn default() -> Self {
        Self {
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_secs: 60,
            enable_graceful_degradation: true,
            enable_heuristic_fallback: true,
            max_recovery_attempts: 3,
        }
    }
}

impl Default for LoggingConfiguration {
    fn default() -> Self {
        Self {
            min_level: "info".to_string(),
            enable_structured_logging: true,
            enable_json: false,
            buffer_size: 1000,
            enable_request_tracing: true,
        }
    }
}

impl Default for PerformanceConfiguration {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            worker_threads: 4,
            enable_prompt_compression: true,
            compression_level: 3,
            enable_parallel_queries: true,
            max_parallel_queries: 4,
        }
    }
}

impl MissionCharlieConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with environment variable overrides
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut config = Self::from_file(path)?;
        config.load_env_overrides()?;
        Ok(config)
    }

    /// Create from environment variables only
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        config.load_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Load API keys from environment variables
    fn load_env_overrides(&mut self) -> Result<()> {
        // Load OpenAI key
        if let Ok(key) = env::var(&self.llm_config.openai.api_key_env) {
            self.llm_config.openai.api_key = Some(key);
        }

        // Load Claude key
        if let Ok(key) = env::var(&self.llm_config.claude.api_key_env) {
            self.llm_config.claude.api_key = Some(key);
        }

        // Load Gemini key
        if let Ok(key) = env::var(&self.llm_config.gemini.api_key_env) {
            self.llm_config.gemini.api_key = Some(key);
        }

        // Load Grok key
        if let Ok(key) = env::var(&self.llm_config.grok.api_key_env) {
            self.llm_config.grok.api_key = Some(key);
        }

        // Override configurations from environment
        if let Ok(val) = env::var("MISSION_CHARLIE_ENABLE_GPU") {
            self.performance_config.enable_gpu = val.parse().unwrap_or(true);
        }

        if let Ok(val) = env::var("MISSION_CHARLIE_LOG_LEVEL") {
            self.logging_config.min_level = val;
        }

        if let Ok(val) = env::var("MISSION_CHARLIE_CACHE_SIZE") {
            self.cache_config.max_cache_size = val.parse().unwrap_or(10000);
        }

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate at least one LLM is enabled
        let enabled_llms = vec![
            self.llm_config.openai.enabled,
            self.llm_config.claude.enabled,
            self.llm_config.gemini.enabled,
            self.llm_config.grok.enabled,
        ];

        if !enabled_llms.iter().any(|&e| e) {
            return Err(anyhow!("At least one LLM must be enabled"));
        }

        // Validate enabled LLMs have API keys
        if self.llm_config.openai.enabled && self.llm_config.openai.api_key.is_none() {
            return Err(anyhow!("OpenAI enabled but no API key provided"));
        }

        if self.llm_config.claude.enabled && self.llm_config.claude.api_key.is_none() {
            return Err(anyhow!("Claude enabled but no API key provided"));
        }

        if self.llm_config.gemini.enabled && self.llm_config.gemini.api_key.is_none() {
            return Err(anyhow!("Gemini enabled but no API key provided"));
        }

        if self.llm_config.grok.enabled && self.llm_config.grok.api_key.is_none() {
            return Err(anyhow!("Grok enabled but no API key provided"));
        }

        // Validate consensus configuration
        if self.consensus_config.min_llms_for_consensus < 2 {
            return Err(anyhow!("Minimum LLMs for consensus must be at least 2"));
        }

        if self.consensus_config.min_llms_for_consensus > 4 {
            return Err(anyhow!(
                "Minimum LLMs for consensus cannot exceed 4 (total available)"
            ));
        }

        // Validate confidence threshold
        if self.consensus_config.confidence_threshold < 0.0
            || self.consensus_config.confidence_threshold > 1.0
        {
            return Err(anyhow!("Confidence threshold must be between 0.0 and 1.0"));
        }

        // Validate cache configuration
        if self.cache_config.similarity_threshold < 0.0
            || self.cache_config.similarity_threshold > 1.0
        {
            return Err(anyhow!(
                "Cache similarity threshold must be between 0.0 and 1.0"
            ));
        }

        // Validate performance configuration
        if self.performance_config.compression_level < 0
            || self.performance_config.compression_level > 10
        {
            return Err(anyhow!("Compression level must be between 0 and 10"));
        }

        if self.performance_config.max_parallel_queries < 1 {
            return Err(anyhow!("Max parallel queries must be at least 1"));
        }

        Ok(())
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_to_toml_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml = toml::to_string_pretty(&self)?;
        fs::write(path, toml)?;
        Ok(())
    }

    /// Get count of enabled LLMs
    pub fn enabled_llm_count(&self) -> usize {
        let mut count = 0;
        if self.llm_config.openai.enabled {
            count += 1;
        }
        if self.llm_config.claude.enabled {
            count += 1;
        }
        if self.llm_config.gemini.enabled {
            count += 1;
        }
        if self.llm_config.grok.enabled {
            count += 1;
        }
        count
    }

    /// Check if configuration allows consensus
    pub fn can_use_consensus(&self) -> bool {
        self.enabled_llm_count() >= self.consensus_config.min_llms_for_consensus
    }

    /// Get list of enabled LLM names
    pub fn enabled_llm_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.llm_config.openai.enabled {
            names.push("openai".to_string());
        }
        if self.llm_config.claude.enabled {
            names.push("claude".to_string());
        }
        if self.llm_config.gemini.enabled {
            names.push("gemini".to_string());
        }
        if self.llm_config.grok.enabled {
            names.push("grok".to_string());
        }
        names
    }
}

/// Configuration Builder for Programmatic Construction
pub struct ConfigBuilder {
    config: MissionCharlieConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: MissionCharlieConfig::default(),
        }
    }

    pub fn with_openai_key(mut self, key: String) -> Self {
        self.config.llm_config.openai.api_key = Some(key);
        self
    }

    pub fn with_claude_key(mut self, key: String) -> Self {
        self.config.llm_config.claude.api_key = Some(key);
        self
    }

    pub fn with_gemini_key(mut self, key: String) -> Self {
        self.config.llm_config.gemini.api_key = Some(key);
        self
    }

    pub fn with_grok_key(mut self, key: String) -> Self {
        self.config.llm_config.grok.api_key = Some(key);
        self
    }

    pub fn enable_gpu(mut self, enable: bool) -> Self {
        self.config.performance_config.enable_gpu = enable;
        self
    }

    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.config.cache_config.max_cache_size = size;
        self
    }

    pub fn with_log_level(mut self, level: String) -> Self {
        self.config.logging_config.min_level = level;
        self
    }

    pub fn build(self) -> Result<MissionCharlieConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let mut config = MissionCharlieConfig::default();

        // Set API keys for validation
        config.llm_config.openai.api_key = Some("test".to_string());
        config.llm_config.claude.api_key = Some("test".to_string());
        config.llm_config.gemini.api_key = Some("test".to_string());
        config.llm_config.grok.api_key = Some("test".to_string());

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_enabled_llm_count() {
        let config = MissionCharlieConfig::default();
        assert_eq!(config.enabled_llm_count(), 4);
    }

    #[test]
    fn test_can_use_consensus() {
        let config = MissionCharlieConfig::default();
        assert!(config.can_use_consensus());
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .with_openai_key("key1".to_string())
            .with_claude_key("key2".to_string())
            .with_gemini_key("key3".to_string())
            .with_grok_key("key4".to_string())
            .enable_gpu(false)
            .with_cache_size(5000)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.cache_config.max_cache_size, 5000);
        assert_eq!(config.performance_config.enable_gpu, false);
    }

    #[test]
    fn test_invalid_consensus_config() {
        let mut config = MissionCharlieConfig::default();
        config.consensus_config.min_llms_for_consensus = 10;
        assert!(config.validate().is_err());
    }
}
