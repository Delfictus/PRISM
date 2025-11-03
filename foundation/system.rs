//! High-level runtime that bridges ingestion with the neuromorphic-quantum platform.
//!
//! The `PrismSystem` orchestrates data ingestion, neuromorphic processing, and quantum
//! optimisation, providing an end-to-end execution environment suitable for integration
//! tests and production harnesses.

use crate::foundation::ingestion::{
    DataSource, EngineConfig, IngestionConfig, IngestionEngine, IngestionStats, RetryPolicy,
};
use crate::foundation::platform::NeuromorphicQuantumPlatform;
use crate::foundation::types::{PlatformInput, PlatformOutput, ProcessingConfig};
use anyhow::{anyhow, Result};
use std::time::Duration;

/// Combined runtime for the full PRISM platform.
pub struct PrismSystem {
    platform: NeuromorphicQuantumPlatform,
    ingestion: IngestionEngine,
    processing_config: ProcessingConfig,
    engine_limits: EngineConfig,
    retry_policy: RetryPolicy,
    active_sources: usize,
}

impl PrismSystem {
    /// Create a new runtime with the provided processing and ingestion configuration.
    pub async fn new(
        processing_config: ProcessingConfig,
        ingestion_config: IngestionConfig,
    ) -> Result<Self> {
        let retry_policy: RetryPolicy = ingestion_config.retry.clone().into();
        let engine = IngestionEngine::with_retry_policy(
            ingestion_config.engine.channel_size,
            ingestion_config.engine.history_size,
            retry_policy.clone(),
        );

        let platform = NeuromorphicQuantumPlatform::new(processing_config.clone()).await?;

        Ok(Self {
            platform,
            ingestion: engine,
            processing_config,
            engine_limits: ingestion_config.engine,
            retry_policy,
            active_sources: 0,
        })
    }

    /// Add a new data source using the resilient ingestion pipeline.
    pub async fn add_source(&mut self, source: Box<dyn DataSource>) -> Result<()> {
        if self.active_sources >= self.engine_limits.max_sources {
            return Err(anyhow!(
                "cannot register additional source – max_sources={} reached",
                self.engine_limits.max_sources
            ));
        }

        self.ingestion.start_source(source).await?;
        self.active_sources += 1;
        Ok(())
    }

    /// Add a source using the low-overhead legacy ingestion path.
    ///
    /// This is particularly useful for deterministic synthetic fixtures in tests.
    pub async fn add_legacy_source(&mut self, source: Box<dyn DataSource>) -> Result<()> {
        if self.active_sources >= self.engine_limits.max_sources {
            return Err(anyhow!(
                "cannot register additional source – max_sources={} reached",
                self.engine_limits.max_sources
            ));
        }

        self.ingestion.start_legacy_source(source).await?;
        self.active_sources += 1;
        Ok(())
    }

    /// Process the next available data point within the given timeout.
    ///
    /// Returns `Ok(None)` if no data was available before the timeout elapsed.
    pub async fn process_next(&mut self, timeout: Duration) -> Result<Option<PlatformOutput>> {
        match self.ingestion.get_point(timeout).await {
            Ok(point) => {
                let input = PlatformInput::from_data_point(point, self.processing_config.clone());
                let output = self.platform.process(input).await?;
                Ok(Some(output))
            }
            Err(err) => {
                log::debug!("No ingestion data available within timeout: {}", err);
                Ok(None)
            }
        }
    }

    /// Access current ingestion statistics.
    pub async fn ingestion_stats(&self) -> IngestionStats {
        self.ingestion.get_stats().await
    }

    /// Access the underlying retry policy (useful for telemetry).
    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.retry_policy
    }

    /// Get a shared reference to the neuromorphic-quantum platform.
    pub fn platform(&self) -> &NeuromorphicQuantumPlatform {
        &self.platform
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::SyntheticDataSource;
    use tokio::time::sleep;

    #[tokio::test]
    async fn prism_system_processes_stream() {
        let processing_config = ProcessingConfig::default();
        let ingestion_config = IngestionConfig::default();

        let mut system = PrismSystem::new(processing_config, ingestion_config)
            .await
            .expect("system should initialise");

        let source = Box::new(SyntheticDataSource::sine_wave(4, 1.0));
        system
            .add_legacy_source(source)
            .await
            .expect("legacy source should start");

        // Allow ingestion to prime the channel.
        sleep(Duration::from_millis(100)).await;

        let mut attempts = 0;
        let mut output = None;
        while attempts < 5 {
            if let Some(result) = system
                .process_next(Duration::from_millis(200))
                .await
                .expect("processing should succeed")
            {
                output = Some(result);
                break;
            }
            attempts += 1;
        }

        let output = output.expect("system should produce an output");
        assert!(output.prediction.confidence >= 0.0);
    }
}
