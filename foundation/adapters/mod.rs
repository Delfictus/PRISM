//! Data source adapters for various input types
//!
//! Provides implementations of the DataSource trait for:
//! - Market data (REST APIs like Alpaca)
//! - Sensor data (optical arrays, IoT sensors)
//! - Synthetic data (for testing and demonstrations)

pub mod market_data;
pub mod sensor_data;
pub mod synthetic;

pub use market_data::{AlpacaMarketDataSource, RestApiMarketDataSource};
pub use sensor_data::{IoTSensorSource, OpticalSensorArray};
pub use synthetic::{HighFrequencySource, SyntheticDataSource};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::types::{DataPoint, DataSource};
    use futures::future::BoxFuture;
    use std::collections::HashMap;

    /// Test source that fails intermittently for testing error handling
    pub struct FlakeySource {
        pub counter: usize,
        pub fail_every_n: usize,
        pub name: String,
    }

    impl FlakeySource {
        pub fn new(fail_every_n: usize) -> Self {
            Self {
                counter: 0,
                fail_every_n,
                name: "FlakeySource".to_string(),
            }
        }
    }

    impl DataSource for FlakeySource {
        fn connect(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn read_batch(&mut self) -> BoxFuture<'_, anyhow::Result<Vec<DataPoint>>> {
            Box::pin(async move {
                self.counter += 1;

                if self.counter % self.fail_every_n == 0 {
                    return Err(anyhow::anyhow!("Simulated transient failure"));
                }

                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), self.name.clone());
                metadata.insert("counter".to_string(), self.counter.to_string());

                Ok(vec![DataPoint {
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    values: vec![self.counter as f64],
                    metadata,
                }])
            })
        }

        fn disconnect(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn get_source_info(&self) -> crate::ingestion::types::SourceInfo {
            crate::ingestion::types::SourceInfo {
                name: self.name.clone(),
                data_type: "test".to_string(),
                sampling_rate_hz: 10.0,
                dimensions: 1,
            }
        }
    }

    #[tokio::test]
    async fn test_synthetic_source_basic() {
        let mut source = SyntheticDataSource::sine_wave(5, 1.0);
        source.connect().await.unwrap();

        let batch = source.read_batch().await.unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch[0].values.len(), 5);
    }

    #[tokio::test]
    async fn test_high_frequency_source() {
        let mut source = HighFrequencySource::new(3);
        source.connect().await.unwrap();

        let batch = source.read_batch().await.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].values.len(), 3);
    }

    #[tokio::test]
    async fn test_flakey_source() {
        let mut source = FlakeySource::new(3);
        source.connect().await.unwrap();

        // First two should succeed
        assert!(source.read_batch().await.is_ok());
        assert!(source.read_batch().await.is_ok());

        // Third should fail
        assert!(source.read_batch().await.is_err());

        // Fourth should succeed
        assert!(source.read_batch().await.is_ok());
    }
}
