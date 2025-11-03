//! Sensor Data Adapter for IoT and Scientific Sensors
//!
//! Provides real-time sensor data ingestion for telescopes, IoT devices, etc.

use super::super::ingestion::types::{DataPoint, DataSource, SourceInfo};
use anyhow::Result;
use chrono::Utc;
use futures::future::BoxFuture;
use std::collections::HashMap;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

/// Optical Sensor Array (for DARPA Narcissus-style applications)
///
/// Connects to a network of optical sensors for photon flux measurements
pub struct OpticalSensorArray {
    name: String,
    sensor_addresses: Vec<String>,
    num_sensors: usize,
    client: Option<TcpStream>,
    sampling_rate_hz: f64,
}

impl OpticalSensorArray {
    /// Create a new optical sensor array
    pub fn new(
        name: String,
        sensor_addresses: Vec<String>,
        num_sensors: usize,
        sampling_rate_hz: f64,
    ) -> Self {
        Self {
            name,
            sensor_addresses,
            num_sensors,
            client: None,
            sampling_rate_hz,
        }
    }

    /// Parse binary sensor packet
    ///
    /// Each sensor transmits 8-byte double precision photon flux values
    fn parse_sensor_packet(data: &[u8]) -> Result<Vec<f64>> {
        let mut readings = Vec::new();

        for chunk in data.chunks(8) {
            if chunk.len() == 8 {
                let bytes: [u8; 8] = chunk.try_into()?;
                let value = f64::from_le_bytes(bytes);
                readings.push(value);
            }
        }

        Ok(readings)
    }
}

impl DataSource for OpticalSensorArray {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            if self.sensor_addresses.is_empty() {
                return Err(anyhow::anyhow!("No sensor addresses configured"));
            }

            let stream = TcpStream::connect(&self.sensor_addresses[0]).await?;
            self.client = Some(stream);

            log::info!(
                "Connected to optical sensor array: {} sensors at {}",
                self.num_sensors,
                self.sensor_addresses[0]
            );

            Ok(())
        })
    }

    fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
        Box::pin(async move {
            let stream = self
                .client
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Not connected to sensor array"))?;

            let mut buffer = vec![0u8; 16384];
            let n = stream.read(&mut buffer).await?;

            if n == 0 {
                return Err(anyhow::anyhow!("Connection closed by sensor array"));
            }

            let readings = Self::parse_sensor_packet(&buffer[..n])?;

            if readings.is_empty() {
                return Ok(Vec::new());
            }

            let timestamp = Utc::now().timestamp_millis();

            let mut metadata = HashMap::new();
            metadata.insert("sensor_type".to_string(), "optical_aperture".to_string());
            metadata.insert("num_sensors".to_string(), self.num_sensors.to_string());
            metadata.insert("source".to_string(), self.name.clone());

            let point = DataPoint {
                timestamp,
                values: readings,
                metadata,
            };

            Ok(vec![point])
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = None;
            log::info!("Disconnected from optical sensor array");
            Ok(())
        })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name.clone(),
            data_type: "photon_flux".to_string(),
            sampling_rate_hz: self.sampling_rate_hz,
            dimensions: self.num_sensors,
        }
    }

    fn is_connected(&self) -> bool {
        self.client.is_some()
    }
}

/// Generic IoT Sensor Source
///
/// Connects to IoT sensors via HTTP/REST endpoints
pub struct IoTSensorSource {
    name: String,
    endpoint_url: String,
    sensor_ids: Vec<String>,
    client: Option<reqwest::Client>,
    auth_token: Option<String>,
}

impl IoTSensorSource {
    /// Create a new IoT sensor source
    pub fn new(
        name: String,
        endpoint_url: String,
        sensor_ids: Vec<String>,
        auth_token: Option<String>,
    ) -> Self {
        Self {
            name,
            endpoint_url,
            sensor_ids,
            client: None,
            auth_token,
        }
    }
}

impl DataSource for IoTSensorSource {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = Some(reqwest::Client::new());
            log::info!("Connected to IoT sensor endpoint: {}", self.endpoint_url);
            Ok(())
        })
    }

    fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
        Box::pin(async move {
            let client = self
                .client
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Not connected"))?;

            let mut points = Vec::new();

            for sensor_id in &self.sensor_ids {
                let url = format!("{}/sensors/{}/latest", self.endpoint_url, sensor_id);

                let mut request = client.get(&url);

                if let Some(token) = &self.auth_token {
                    request = request.header("Authorization", format!("Bearer {}", token));
                }

                match request.send().await {
                    Ok(response) => {
                        if let Ok(data) = response.json::<serde_json::Value>().await {
                            if let Some(reading) = data.get("value").and_then(|v| v.as_f64()) {
                                let timestamp = data
                                    .get("timestamp")
                                    .and_then(|t| t.as_i64())
                                    .unwrap_or_else(|| Utc::now().timestamp_millis());

                                let mut metadata = HashMap::new();
                                metadata.insert("sensor_id".to_string(), sensor_id.clone());
                                metadata.insert("source".to_string(), self.name.clone());

                                if let Some(unit) = data.get("unit").and_then(|u| u.as_str()) {
                                    metadata.insert("unit".to_string(), unit.to_string());
                                }

                                points.push(DataPoint {
                                    timestamp,
                                    values: vec![reading],
                                    metadata,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to fetch data for sensor {}: {}", sensor_id, e);
                    }
                }
            }

            Ok(points)
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = None;
            log::info!("Disconnected from IoT sensor endpoint");
            Ok(())
        })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name.clone(),
            data_type: "iot_sensor_readings".to_string(),
            sampling_rate_hz: 1.0,
            dimensions: self.sensor_ids.len(),
        }
    }

    fn is_connected(&self) -> bool {
        self.client.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_sensor_array_creation() {
        let source = OpticalSensorArray::new(
            "Test Array".to_string(),
            vec!["localhost:9000".to_string()],
            900,
            100.0,
        );

        let info = source.get_source_info();
        assert_eq!(info.dimensions, 900);
        assert_eq!(info.sampling_rate_hz, 100.0);
        assert!(!source.is_connected());
    }

    #[test]
    fn test_parse_sensor_packet() {
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // 2.0
        ];

        let readings = OpticalSensorArray::parse_sensor_packet(&data).unwrap();
        assert_eq!(readings.len(), 2);
        assert_eq!(readings[0], 1.0);
        assert_eq!(readings[1], 2.0);
    }

    #[test]
    fn test_iot_sensor_source_creation() {
        let source = IoTSensorSource::new(
            "Test IoT".to_string(),
            "http://localhost:8080".to_string(),
            vec!["sensor1".to_string(), "sensor2".to_string()],
            None,
        );

        let info = source.get_source_info();
        assert_eq!(info.dimensions, 2);
        assert!(!source.is_connected());
    }
}
