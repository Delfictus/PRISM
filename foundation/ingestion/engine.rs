//! Ingestion engine for managing multiple data sources

use super::buffer::CircularBuffer;
use super::error::{CircuitBreaker, IngestionError, RetryPolicy};
use super::types::{DataPoint, DataSource};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Statistics for ingestion performance monitoring
#[derive(Debug, Clone)]
pub struct IngestionStats {
    /// Total data points ingested
    pub total_points: usize,
    /// Total bytes ingested
    pub total_bytes: usize,
    /// Last update timestamp
    pub last_update: Instant,
    /// Average ingestion rate (points/sec)
    pub average_rate_hz: f64,
    /// Number of active sources
    pub active_sources: usize,
    /// Number of errors encountered
    pub error_count: usize,
    /// Number of successful retries
    pub retry_success_count: usize,
    /// Number of failed retries
    pub retry_failed_count: usize,
    /// Circuit breaker states by source
    pub circuit_breaker_states: HashMap<String, String>,
}

impl Default for IngestionStats {
    fn default() -> Self {
        Self {
            total_points: 0,
            total_bytes: 0,
            last_update: Instant::now(),
            average_rate_hz: 0.0,
            active_sources: 0,
            error_count: 0,
            retry_success_count: 0,
            retry_failed_count: 0,
            circuit_breaker_states: HashMap::new(),
        }
    }
}

/// Main ingestion engine managing multiple data sources
pub struct IngestionEngine {
    /// Historical buffer for data points
    buffer: CircularBuffer<DataPoint>,
    /// Channel sender for new data
    tx: mpsc::Sender<DataPoint>,
    /// Channel receiver for new data
    rx: Option<mpsc::Receiver<DataPoint>>,
    /// Performance statistics
    stats: Arc<RwLock<IngestionStats>>,
    /// Start time for rate calculation
    start_time: Instant,
    /// Retry policy for transient failures
    retry_policy: RetryPolicy,
    /// Circuit breakers per source
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
}

impl IngestionEngine {
    /// Create a new ingestion engine
    ///
    /// # Arguments
    /// * `channel_size` - Size of the async channel buffer
    /// * `history_size` - Size of the historical circular buffer
    pub fn new(channel_size: usize, history_size: usize) -> Self {
        Self::with_retry_policy(channel_size, history_size, RetryPolicy::default())
    }

    /// Create a new ingestion engine with custom retry policy
    ///
    /// # Arguments
    /// * `channel_size` - Size of the async channel buffer
    /// * `history_size` - Size of the historical circular buffer
    /// * `retry_policy` - Retry policy for handling failures
    pub fn with_retry_policy(
        channel_size: usize,
        history_size: usize,
        retry_policy: RetryPolicy,
    ) -> Self {
        let (tx, rx) = mpsc::channel(channel_size);

        Self {
            buffer: CircularBuffer::new(history_size),
            tx,
            rx: Some(rx),
            stats: Arc::new(RwLock::new(IngestionStats::default())),
            start_time: Instant::now(),
            retry_policy,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start ingesting from a data source
    ///
    /// Spawns an async task to continuously read from the source
    pub async fn start_source(&mut self, mut source: Box<dyn DataSource>) -> Result<()> {
        let source_info = source.get_source_info();
        let source_name = source_info.name.clone();

        // Initialize circuit breaker for this source
        {
            let mut breakers = self.circuit_breakers.write().await;
            breakers.insert(source_name.clone(), CircuitBreaker::default());
        }

        // Connect to source with retry
        let mut retry_count = 0;
        loop {
            match source.connect().await {
                Ok(_) => {
                    log::info!("Successfully connected to: {}", source_name);
                    break;
                }
                Err(e) => {
                    if retry_count >= self.retry_policy.max_attempts {
                        log::error!(
                            "Failed to connect to {} after {} attempts: {}",
                            source_name,
                            retry_count,
                            e
                        );
                        return Err(IngestionError::ConnectionFailed {
                            source: source_name,
                            reason: e.to_string(),
                            retryable: false,
                        }
                        .into());
                    }

                    let backoff = self.retry_policy.backoff_delay(retry_count);
                    log::warn!(
                        "Failed to connect to {} (attempt {}/{}): {}. Retrying in {}ms",
                        source_name,
                        retry_count + 1,
                        self.retry_policy.max_attempts,
                        e,
                        backoff
                    );

                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                    retry_count += 1;
                }
            }
        }

        log::info!("Starting ingestion from: {}", source_name);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.active_sources += 1;
        }

        // Spawn ingestion task
        let tx = self.tx.clone();
        let stats = Arc::clone(&self.stats);
        let circuit_breakers = Arc::clone(&self.circuit_breakers);
        let retry_policy = self.retry_policy.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::ingest_from_source_with_recovery(
                source,
                tx,
                stats,
                circuit_breakers,
                retry_policy,
            )
            .await
            {
                log::error!("Ingestion task failed: {}", e);
            }
        });

        Ok(())
    }

    /// Internal task to ingest from a single source with retry and circuit breaker
    async fn ingest_from_source_with_recovery(
        mut source: Box<dyn DataSource>,
        tx: mpsc::Sender<DataPoint>,
        stats: Arc<RwLock<IngestionStats>>,
        circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
        retry_policy: RetryPolicy,
    ) -> Result<()> {
        let source_info = source.get_source_info();
        let source_name = source_info.name.clone();

        loop {
            // Check circuit breaker
            {
                let breakers = circuit_breakers.read().await;
                if let Some(cb) = breakers.get(&source_name) {
                    if !cb.is_closed() {
                        log::warn!(
                            "Circuit breaker open for {}, waiting before retry...",
                            source_name
                        );
                        tokio::time::sleep(Duration::from_millis(cb.timeout_ms)).await;
                        continue;
                    }
                }
            }

            // Try to read batch with retries
            let mut retry_count = 0;
            let result = loop {
                match source.read_batch().await {
                    Ok(batch) => break Ok(batch),
                    Err(e) => {
                        if retry_count >= retry_policy.max_attempts {
                            break Err(e);
                        }

                        let backoff = retry_policy.backoff_delay(retry_count);
                        log::warn!(
                            "Error reading from {} (attempt {}/{}): {}. Retrying in {}ms",
                            source_name,
                            retry_count + 1,
                            retry_policy.max_attempts,
                            e,
                            backoff
                        );

                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        retry_count += 1;
                    }
                }
            };

            match result {
                Ok(batch) => {
                    let batch_size = batch.len();

                    if batch_size == 0 {
                        // No data, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }

                    // Send each point through the channel
                    for point in batch {
                        if tx.send(point).await.is_err() {
                            log::warn!("Ingestion channel closed for {}", source_name);
                            return Err(IngestionError::ChannelClosed {
                                source: source_name.clone(),
                            }
                            .into());
                        }
                    }

                    // Success - update circuit breaker and stats
                    {
                        let mut breakers = circuit_breakers.write().await;
                        if let Some(cb) = breakers.get_mut(&source_name) {
                            cb.record_success();
                        }
                    }

                    {
                        let mut s = stats.write().await;
                        s.total_points += batch_size;
                        s.last_update = Instant::now();

                        if retry_count > 0 {
                            s.retry_success_count += 1;
                        }
                    }
                }
                Err(e) => {
                    log::error!("Error reading from {} after retries: {}", source_name, e);

                    // Record failure in circuit breaker
                    {
                        let mut breakers = circuit_breakers.write().await;
                        if let Some(cb) = breakers.get_mut(&source_name) {
                            cb.record_failure();

                            // Check if circuit breaker opened
                            if !cb.is_closed() {
                                log::error!(
                                    "Circuit breaker opened for {} after {} errors (threshold: {})",
                                    source_name,
                                    cb.error_count(),
                                    cb.error_threshold
                                );
                            }
                        }
                    }

                    // Update error stats
                    {
                        let mut s = stats.write().await;
                        s.error_count += 1;
                        s.retry_failed_count += 1;
                    }

                    // Try to reconnect
                    log::info!("Attempting to reconnect to {}...", source_name);
                    if let Err(reconnect_err) = source.connect().await {
                        log::error!("Failed to reconnect to {}: {}", source_name, reconnect_err);
                    }

                    // Wait before next attempt
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }

    /// Internal task to ingest from a single source (legacy without retry)
    async fn ingest_from_source(
        mut source: Box<dyn DataSource>,
        tx: mpsc::Sender<DataPoint>,
        stats: Arc<RwLock<IngestionStats>>,
    ) -> Result<()> {
        let source_info = source.get_source_info();

        loop {
            match source.read_batch().await {
                Ok(batch) => {
                    let batch_size = batch.len();

                    if batch_size == 0 {
                        // No data, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }

                    // Send each point through the channel
                    for point in batch {
                        if tx.send(point).await.is_err() {
                            log::warn!("Ingestion channel closed for {}", source_info.name);
                            return Ok(());
                        }
                    }

                    // Update statistics
                    let mut s = stats.write().await;
                    s.total_points += batch_size;
                    s.last_update = Instant::now();
                }
                Err(e) => {
                    log::error!("Error reading from {}: {}", source_info.name, e);

                    // Update error count
                    let mut s = stats.write().await;
                    s.error_count += 1;

                    // Wait before retrying
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Get a batch of data points
    ///
    /// Blocks until `size` points are available or `timeout` is reached
    pub async fn get_batch(&mut self, size: usize, timeout: Duration) -> Result<Vec<DataPoint>> {
        let rx = self
            .rx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Receiver already taken"))?;

        let mut batch = Vec::with_capacity(size);
        let deadline = tokio::time::Instant::now() + timeout;

        while batch.len() < size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(point)) => {
                    // Add to buffer
                    self.buffer.push(point.clone()).await;
                    batch.push(point);
                }
                Ok(None) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout
                    break;
                }
            }
        }

        Ok(batch)
    }

    /// Get a single data point (blocking with timeout)
    pub async fn get_point(&mut self, timeout: Duration) -> Result<DataPoint> {
        let rx = self
            .rx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Receiver already taken"))?;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(point)) => {
                self.buffer.push(point.clone()).await;
                Ok(point)
            }
            Ok(None) => Err(anyhow::anyhow!("Channel closed")),
            Err(_) => Err(anyhow::anyhow!("Timeout waiting for data")),
        }
    }

    /// Get recent historical data
    pub async fn get_history(&self, n: usize) -> Vec<DataPoint> {
        self.buffer.get_recent(n).await
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> IngestionStats {
        let mut stats = self.stats.read().await.clone();

        // Calculate average rate
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            stats.average_rate_hz = stats.total_points as f64 / elapsed;
        }

        // Update circuit breaker states
        let breakers = self.circuit_breakers.read().await;
        for (source, cb) in breakers.iter() {
            let state_str = match cb.state() {
                super::error::CircuitBreakerState::Closed => "closed",
                super::error::CircuitBreakerState::Open => "open",
                super::error::CircuitBreakerState::HalfOpen => "half-open",
            };
            stats
                .circuit_breaker_states
                .insert(source.clone(), state_str.to_string());
        }

        stats
    }

    /// Get circuit breaker status for a source
    pub async fn get_circuit_breaker_status(&self, source_name: &str) -> Option<String> {
        let breakers = self.circuit_breakers.read().await;
        breakers.get(source_name).map(|cb| {
            format!(
                "state={:?}, errors={}/{}",
                cb.state(),
                cb.error_count(),
                cb.error_threshold
            )
        })
    }

    /// Get comprehensive health report
    pub async fn get_health_report(&self) -> super::health::HealthReport {
        let stats = self.get_stats().await;
        let buffer_size = self.buffer_size().await;
        let buffer_capacity = self.buffer.capacity();

        super::health::HealthReport::from_stats(stats, buffer_size, buffer_capacity)
    }

    /// Get buffer size
    pub async fn buffer_size(&self) -> usize {
        self.buffer.len().await
    }

    /// Clear historical buffer
    pub async fn clear_buffer(&self) {
        self.buffer.clear().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::types::{DataPoint, DataSource, SourceInfo};
    use futures::future::BoxFuture;

    struct MockSource {
        counter: usize,
        batch_size: usize,
    }

    impl DataSource for MockSource {
        fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
            Box::pin(async move {
                let mut batch = Vec::new();
                for _ in 0..self.batch_size {
                    batch.push(DataPoint::new(
                        chrono::Utc::now().timestamp_millis(),
                        vec![self.counter as f64],
                    ));
                    self.counter += 1;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(batch)
            })
        }

        fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn get_source_info(&self) -> SourceInfo {
            SourceInfo {
                name: "MockSource".to_string(),
                data_type: "test".to_string(),
                sampling_rate_hz: 100.0,
                dimensions: 1,
            }
        }
    }

    #[tokio::test]
    async fn test_ingestion_engine_basic() {
        let mut engine = IngestionEngine::new(100, 1000);

        let source = Box::new(MockSource {
            counter: 0,
            batch_size: 5,
        });

        engine.start_source(source).await.unwrap();

        // Wait for some data
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Get a batch
        let batch = engine
            .get_batch(10, Duration::from_millis(500))
            .await
            .unwrap();

        assert!(!batch.is_empty());
        assert!(batch.len() <= 10);
    }

    #[tokio::test]
    async fn test_ingestion_stats() {
        let mut engine = IngestionEngine::new(100, 1000);

        let source = Box::new(MockSource {
            counter: 0,
            batch_size: 10,
        });

        engine.start_source(source).await.unwrap();

        // Wait for data
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = engine.get_stats().await;
        assert!(stats.total_points > 0);
        assert_eq!(stats.active_sources, 1);
    }
}
