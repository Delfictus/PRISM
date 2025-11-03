//! Synthetic Data Source for Testing and Demonstrations
//!
//! Generates realistic simulated data for testing ingestion pipelines

use super::super::ingestion::types::{DataPoint, DataSource, SourceInfo};
use anyhow::Result;
use chrono::Utc;
use futures::future::BoxFuture;
use rand::Rng;
use std::collections::HashMap;

/// Synthetic data source that generates realistic test data
pub struct SyntheticDataSource {
    name: String,
    data_type: String,
    dimensions: usize,
    sampling_rate_hz: f64,
    batch_size: usize,
    counter: usize,
    generator: DataGenerator,
}

/// Data generation patterns
#[derive(Debug, Clone)]
pub enum DataGenerator {
    /// Sine wave with optional noise
    SineWave {
        frequency: f64,
        amplitude: f64,
        noise: f64,
    },
    /// Random walk
    RandomWalk { step_size: f64, start: f64 },
    /// Gaussian noise
    Gaussian { mean: f64, std_dev: f64 },
    /// Constant value
    Constant { value: f64 },
    /// Linear trend
    Linear { slope: f64, intercept: f64 },
}

impl SyntheticDataSource {
    /// Create a new synthetic data source
    pub fn new(
        name: String,
        data_type: String,
        dimensions: usize,
        sampling_rate_hz: f64,
        batch_size: usize,
        generator: DataGenerator,
    ) -> Self {
        Self {
            name,
            data_type,
            dimensions,
            sampling_rate_hz,
            batch_size,
            counter: 0,
            generator,
        }
    }

    /// Create a sine wave source
    pub fn sine_wave(dimensions: usize, frequency: f64) -> Self {
        Self::new(
            "Synthetic Sine Wave".to_string(),
            "synthetic".to_string(),
            dimensions,
            100.0,
            10,
            DataGenerator::SineWave {
                frequency,
                amplitude: 1.0,
                noise: 0.1,
            },
        )
    }

    /// Create a random walk source
    pub fn random_walk(dimensions: usize) -> Self {
        Self::new(
            "Synthetic Random Walk".to_string(),
            "synthetic".to_string(),
            dimensions,
            100.0,
            10,
            DataGenerator::RandomWalk {
                step_size: 0.1,
                start: 0.0,
            },
        )
    }

    /// Create a Gaussian noise source
    pub fn gaussian(dimensions: usize) -> Self {
        Self::new(
            "Synthetic Gaussian".to_string(),
            "synthetic".to_string(),
            dimensions,
            100.0,
            10,
            DataGenerator::Gaussian {
                mean: 0.0,
                std_dev: 1.0,
            },
        )
    }

    /// Generate a single value based on the generator type
    fn generate_value(&self, time_step: usize, dimension: usize) -> f64 {
        let mut rng = rand::thread_rng();

        match &self.generator {
            DataGenerator::SineWave {
                frequency,
                amplitude,
                noise,
            } => {
                let t = time_step as f64 / self.sampling_rate_hz;
                let phase_offset = dimension as f64 * std::f64::consts::PI / 4.0;
                let sine =
                    amplitude * (2.0 * std::f64::consts::PI * frequency * t + phase_offset).sin();
                let noise_val = rng.gen::<f64>() * noise - noise / 2.0;
                sine + noise_val
            }
            DataGenerator::RandomWalk { step_size, start } => {
                let steps = time_step as f64;
                let random_step = (rng.gen::<f64>() - 0.5) * step_size;
                start + steps * random_step
            }
            DataGenerator::Gaussian { mean, std_dev } => {
                // Box-Muller transform for Gaussian
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let z0 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std_dev * z0
            }
            DataGenerator::Constant { value } => *value,
            DataGenerator::Linear { slope, intercept } => slope * time_step as f64 + intercept,
        }
    }
}

impl DataSource for SyntheticDataSource {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            log::info!("Connected to synthetic data source: {}", self.name);
            self.counter = 0;
            Ok(())
        })
    }

    fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
        Box::pin(async move {
            let mut batch = Vec::with_capacity(self.batch_size);

            for _ in 0..self.batch_size {
                let timestamp = Utc::now().timestamp_millis();

                let mut values = Vec::with_capacity(self.dimensions);
                for dim in 0..self.dimensions {
                    values.push(self.generate_value(self.counter, dim));
                }

                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), self.name.clone());
                metadata.insert("type".to_string(), self.data_type.clone());
                metadata.insert("sample".to_string(), self.counter.to_string());

                batch.push(DataPoint {
                    timestamp,
                    values,
                    metadata,
                });

                self.counter += 1;
            }

            let delay_ms = (1000.0 / self.sampling_rate_hz * self.batch_size as f64) as u64;
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

            Ok(batch)
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            log::info!("Disconnected from synthetic data source");
            Ok(())
        })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name.clone(),
            data_type: self.data_type.clone(),
            sampling_rate_hz: self.sampling_rate_hz,
            dimensions: self.dimensions,
        }
    }

    fn is_connected(&self) -> bool {
        true // Synthetic source is always "connected"
    }
}

/// High-frequency synthetic source (for latency testing)
pub struct HighFrequencySource {
    dimensions: usize,
    counter: usize,
}

impl HighFrequencySource {
    /// Create a new high-frequency source
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            counter: 0,
        }
    }
}

impl DataSource for HighFrequencySource {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.counter = 0;
            Ok(())
        })
    }

    fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
        Box::pin(async move {
            let timestamp = Utc::now().timestamp_millis();

            let values: Vec<f64> = (0..self.dimensions)
                .map(|i| (self.counter + i) as f64)
                .collect();

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), "high_frequency".to_string());
            metadata.insert("counter".to_string(), self.counter.to_string());

            self.counter += 1;

            Ok(vec![DataPoint {
                timestamp,
                values,
                metadata,
            }])
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { Ok(()) })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: "High-Frequency Test Source".to_string(),
            data_type: "high_frequency_test".to_string(),
            sampling_rate_hz: 1000.0,
            dimensions: self.dimensions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synthetic_sine_wave() {
        let mut source = SyntheticDataSource::sine_wave(3, 1.0);
        source.connect().await.unwrap();

        let batch = source.read_batch().await.unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch[0].values.len(), 3);
    }

    #[tokio::test]
    async fn test_synthetic_random_walk() {
        let mut source = SyntheticDataSource::random_walk(5);
        source.connect().await.unwrap();

        let batch = source.read_batch().await.unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch[0].values.len(), 5);
    }

    #[tokio::test]
    async fn test_high_frequency_source() {
        let mut source = HighFrequencySource::new(10);
        source.connect().await.unwrap();

        let batch = source.read_batch().await.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].values.len(), 10);
    }

    #[test]
    fn test_data_generators() {
        let source = SyntheticDataSource::sine_wave(1, 1.0);
        let val = source.generate_value(0, 0);
        assert!(val.abs() <= 1.2); // Within amplitude + noise

        let source = SyntheticDataSource::new(
            "test".to_string(),
            "test".to_string(),
            1,
            100.0,
            10,
            DataGenerator::Constant { value: 5.0 },
        );
        let val = source.generate_value(0, 0);
        assert_eq!(val, 5.0);
    }
}
