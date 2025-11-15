//! Production Logging System with Structured Logging
//!
//! Mission Charlie: Phase 4 (Essential production features)
//!
//! Implements comprehensive structured logging with multiple severity levels,
//! context enrichment, and integration with standard logging frameworks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Production Logger with Structured Logging
pub struct ProductionLogger {
    config: LogConfig,
    context: Arc<Mutex<LogContext>>,
    buffer: Arc<Mutex<Vec<LogEntry>>>,
}

#[derive(Debug, Clone)]
pub struct LogConfig {
    pub min_level: LogLevel,
    pub enable_timestamps: bool,
    pub enable_context: bool,
    pub buffer_size: usize,
    pub output_format: LogFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Critical = 5,
}

#[derive(Debug, Clone, Copy)]
pub enum LogFormat {
    Json,
    Plain,
    Structured,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            enable_timestamps: true,
            enable_context: true,
            buffer_size: 1000,
            output_format: LogFormat::Structured,
        }
    }
}

impl ProductionLogger {
    pub fn new(config: LogConfig) -> Self {
        let buffer_size = config.buffer_size;
        Self {
            config,
            context: Arc::new(Mutex::new(LogContext::new())),
            buffer: Arc::new(Mutex::new(Vec::with_capacity(buffer_size))),
        }
    }

    /// Log a trace message
    pub fn trace(&self, message: &str) {
        self.log(LogLevel::Trace, message, None);
    }

    /// Log a debug message
    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message, None);
    }

    /// Log an info message
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message, None);
    }

    /// Log a warning message
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message, None);
    }

    /// Log an error message
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message, None);
    }

    /// Log a critical message
    pub fn critical(&self, message: &str) {
        self.log(LogLevel::Critical, message, None);
    }

    /// Log with additional metadata
    pub fn log_with_metadata(
        &self,
        level: LogLevel,
        message: &str,
        metadata: HashMap<String, String>,
    ) {
        self.log(level, message, Some(metadata));
    }

    /// Core logging function
    fn log(&self, level: LogLevel, message: &str, metadata: Option<HashMap<String, String>>) {
        // Check if level meets minimum threshold
        if level < self.config.min_level {
            return;
        }

        let entry = self.create_log_entry(level, message, metadata);

        // Write to buffer
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push(entry.clone());

        // Rotate buffer if full
        if buffer.len() >= self.config.buffer_size {
            let drain_size = buffer.len() / 2;
            buffer.drain(0..drain_size);
        }

        // Output log entry
        self.output_log(&entry);
    }

    /// Create a log entry with context
    fn create_log_entry(
        &self,
        level: LogLevel,
        message: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> LogEntry {
        let timestamp = if self.config.enable_timestamps {
            Some(Self::get_timestamp())
        } else {
            None
        };

        let context = if self.config.enable_context {
            let ctx = self.context.lock().unwrap();
            Some(ctx.clone())
        } else {
            None
        };

        LogEntry {
            timestamp,
            level,
            message: message.to_string(),
            metadata,
            context,
        }
    }

    /// Output log entry in configured format
    fn output_log(&self, entry: &LogEntry) {
        match self.config.output_format {
            LogFormat::Json => {
                if let Ok(json) = serde_json::to_string(&entry) {
                    eprintln!("{}", json);
                }
            }
            LogFormat::Plain => {
                eprintln!("{}", self.format_plain(entry));
            }
            LogFormat::Structured => {
                eprintln!("{}", self.format_structured(entry));
            }
        }
    }

    /// Format log entry as plain text
    fn format_plain(&self, entry: &LogEntry) -> String {
        let timestamp = entry
            .timestamp
            .map(|ts| format!("[{}] ", ts))
            .unwrap_or_default();

        let level = format!("{:?}", entry.level).to_uppercase();

        format!("{}[{}] {}", timestamp, level, entry.message)
    }

    /// Format log entry as structured text
    fn format_structured(&self, entry: &LogEntry) -> String {
        let mut parts = Vec::new();

        if let Some(ts) = entry.timestamp {
            parts.push(format!("timestamp={}", ts));
        }

        parts.push(format!("level={:?}", entry.level));
        parts.push(format!("message=\"{}\"", entry.message));

        if let Some(ref metadata) = entry.metadata {
            for (key, value) in metadata {
                parts.push(format!("{}=\"{}\"", key, value));
            }
        }

        if let Some(ref ctx) = entry.context {
            if let Some(ref req_id) = ctx.request_id {
                parts.push(format!("request_id={}", req_id));
            }
            if let Some(ref user_id) = ctx.user_id {
                parts.push(format!("user_id={}", user_id));
            }
            if let Some(ref operation) = ctx.operation {
                parts.push(format!("operation={}", operation));
            }
        }

        parts.join(" ")
    }

    /// Get current timestamp
    fn get_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Add context field
    pub fn add_context(&self, key: &str, value: &str) {
        let mut ctx = self.context.lock().unwrap();
        ctx.add_field(key, value);
    }

    /// Set request ID for tracing
    pub fn set_request_id(&self, request_id: &str) {
        let mut ctx = self.context.lock().unwrap();
        ctx.request_id = Some(request_id.to_string());
    }

    /// Set user ID for context
    pub fn set_user_id(&self, user_id: &str) {
        let mut ctx = self.context.lock().unwrap();
        ctx.user_id = Some(user_id.to_string());
    }

    /// Set operation name
    pub fn set_operation(&self, operation: &str) {
        let mut ctx = self.context.lock().unwrap();
        ctx.operation = Some(operation.to_string());
    }

    /// Clear context
    pub fn clear_context(&self) {
        let mut ctx = self.context.lock().unwrap();
        *ctx = LogContext::new();
    }

    /// Get buffered logs
    pub fn get_logs(&self) -> Vec<LogEntry> {
        let buffer = self.buffer.lock().unwrap();
        buffer.clone()
    }

    /// Get logs by level
    pub fn get_logs_by_level(&self, level: LogLevel) -> Vec<LogEntry> {
        let buffer = self.buffer.lock().unwrap();
        buffer
            .iter()
            .filter(|entry| entry.level == level)
            .cloned()
            .collect()
    }

    /// Get error count
    pub fn get_error_count(&self) -> usize {
        let buffer = self.buffer.lock().unwrap();
        buffer
            .iter()
            .filter(|entry| entry.level >= LogLevel::Error)
            .count()
    }

    /// Clear buffer
    pub fn clear_buffer(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.clear();
    }
}

/// Log Entry Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
    pub level: LogLevel,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<LogContext>,
}

impl Serialize for LogLevel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:?}", self))
    }
}

impl<'de> Deserialize<'de> for LogLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "Trace" => Ok(LogLevel::Trace),
            "Debug" => Ok(LogLevel::Debug),
            "Info" => Ok(LogLevel::Info),
            "Warn" => Ok(LogLevel::Warn),
            "Error" => Ok(LogLevel::Error),
            "Critical" => Ok(LogLevel::Critical),
            _ => Err(serde::de::Error::custom("Invalid log level")),
        }
    }
}

/// Log Context for Request Tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub custom_fields: HashMap<String, String>,
}

impl LogContext {
    fn new() -> Self {
        Self {
            request_id: None,
            user_id: None,
            operation: None,
            custom_fields: HashMap::new(),
        }
    }

    fn add_field(&mut self, key: &str, value: &str) {
        self.custom_fields
            .insert(key.to_string(), value.to_string());
    }
}

/// LLM Operation Logger (specialized for LLM calls)
pub struct LLMOperationLogger {
    logger: Arc<ProductionLogger>,
}

impl LLMOperationLogger {
    pub fn new(logger: Arc<ProductionLogger>) -> Self {
        Self { logger }
    }

    /// Log LLM query
    pub fn log_query(&self, llm_name: &str, query: &str, query_type: &str) {
        let mut metadata = HashMap::new();
        metadata.insert("llm".to_string(), llm_name.to_string());
        metadata.insert("query_length".to_string(), query.len().to_string());
        metadata.insert("query_type".to_string(), query_type.to_string());

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!("LLM query to {}", llm_name),
            metadata,
        );
    }

    /// Log LLM response
    pub fn log_response(&self, llm_name: &str, response_length: usize, latency_ms: f64) {
        let mut metadata = HashMap::new();
        metadata.insert("llm".to_string(), llm_name.to_string());
        metadata.insert("response_length".to_string(), response_length.to_string());
        metadata.insert("latency_ms".to_string(), format!("{:.2}", latency_ms));

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!("LLM response from {} ({:.2}ms)", llm_name, latency_ms),
            metadata,
        );
    }

    /// Log LLM error
    pub fn log_error(&self, llm_name: &str, error: &str, recovery_action: &str) {
        let mut metadata = HashMap::new();
        metadata.insert("llm".to_string(), llm_name.to_string());
        metadata.insert("error_type".to_string(), error.to_string());
        metadata.insert("recovery_action".to_string(), recovery_action.to_string());

        self.logger.log_with_metadata(
            LogLevel::Error,
            &format!("LLM error from {}: {}", llm_name, error),
            metadata,
        );
    }

    /// Log cache hit
    pub fn log_cache_hit(&self, cache_type: &str, key_hash: &str) {
        let mut metadata = HashMap::new();
        metadata.insert("cache_type".to_string(), cache_type.to_string());
        metadata.insert("key_hash".to_string(), key_hash.to_string());

        self.logger.log_with_metadata(
            LogLevel::Debug,
            &format!("Cache hit: {}", cache_type),
            metadata,
        );
    }

    /// Log consensus decision
    pub fn log_consensus(&self, method: &str, llm_count: usize, consensus_confidence: f64) {
        let mut metadata = HashMap::new();
        metadata.insert("consensus_method".to_string(), method.to_string());
        metadata.insert("llm_count".to_string(), llm_count.to_string());
        metadata.insert(
            "confidence".to_string(),
            format!("{:.3}", consensus_confidence),
        );

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!(
                "Consensus achieved via {} ({:.1}% confidence)",
                method,
                consensus_confidence * 100.0
            ),
            metadata,
        );
    }
}

/// Integration Logger (specialized for Mission Bravo + Charlie integration)
pub struct IntegrationLogger {
    logger: Arc<ProductionLogger>,
}

impl IntegrationLogger {
    pub fn new(logger: Arc<ProductionLogger>) -> Self {
        Self { logger }
    }

    /// Log sensor fusion event
    pub fn log_sensor_fusion(&self, source_count: usize, fusion_time_ms: f64, threat_level: f64) {
        let mut metadata = HashMap::new();
        metadata.insert("source_count".to_string(), source_count.to_string());
        metadata.insert(
            "fusion_time_ms".to_string(),
            format!("{:.3}", fusion_time_ms),
        );
        metadata.insert("threat_level".to_string(), format!("{:.2}", threat_level));

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!(
                "Sensor fusion complete: {} sources, threat {:.2}",
                source_count, threat_level
            ),
            metadata,
        );
    }

    /// Log LLM context enrichment
    pub fn log_context_enrichment(
        &self,
        sensor_events: usize,
        llm_responses: usize,
        enrichment_time_ms: f64,
    ) {
        let mut metadata = HashMap::new();
        metadata.insert("sensor_events".to_string(), sensor_events.to_string());
        metadata.insert("llm_responses".to_string(), llm_responses.to_string());
        metadata.insert(
            "enrichment_time_ms".to_string(),
            format!("{:.3}", enrichment_time_ms),
        );

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!(
                "Context enrichment: {} sensors + {} LLM responses",
                sensor_events, llm_responses
            ),
            metadata,
        );
    }

    /// Log complete intelligence fusion
    pub fn log_complete_fusion(&self, total_time_ms: f64, confidence: f64) {
        let mut metadata = HashMap::new();
        metadata.insert("total_time_ms".to_string(), format!("{:.3}", total_time_ms));
        metadata.insert("confidence".to_string(), format!("{:.3}", confidence));

        self.logger.log_with_metadata(
            LogLevel::Info,
            &format!(
                "Complete intelligence fusion ({:.2}ms, {:.1}% confidence)",
                total_time_ms,
                confidence * 100.0
            ),
            metadata,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_logging() {
        let logger = ProductionLogger::new(LogConfig::default());

        logger.info("Test info message");
        logger.warn("Test warning");
        logger.error("Test error");

        let logs = logger.get_logs();
        assert_eq!(logs.len(), 3);
    }

    #[test]
    fn test_log_filtering() {
        let mut config = LogConfig::default();
        config.min_level = LogLevel::Warn;

        let logger = ProductionLogger::new(config);

        logger.info("Should be filtered");
        logger.debug("Should be filtered");
        logger.warn("Should appear");
        logger.error("Should appear");

        let logs = logger.get_logs();
        assert_eq!(logs.len(), 2);
    }

    #[test]
    fn test_context_enrichment() {
        let logger = ProductionLogger::new(LogConfig::default());

        logger.set_request_id("req-12345");
        logger.set_user_id("user-67890");
        logger.set_operation("test_operation");

        logger.info("Test with context");

        let logs = logger.get_logs();
        assert_eq!(logs.len(), 1);

        let entry = &logs[0];
        assert!(entry.context.is_some());

        let ctx = entry.context.as_ref().unwrap();
        assert_eq!(ctx.request_id.as_deref(), Some("req-12345"));
        assert_eq!(ctx.user_id.as_deref(), Some("user-67890"));
        assert_eq!(ctx.operation.as_deref(), Some("test_operation"));
    }

    #[test]
    fn test_llm_operation_logger() {
        let logger = Arc::new(ProductionLogger::new(LogConfig::default()));
        let llm_logger = LLMOperationLogger::new(logger.clone());

        llm_logger.log_query("gpt-4", "test query", "reasoning");
        llm_logger.log_response("gpt-4", 100, 250.5);
        llm_logger.log_error("claude", "timeout", "retry");

        let logs = logger.get_logs();
        assert_eq!(logs.len(), 3);
    }
}
