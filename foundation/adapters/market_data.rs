//! Market Data Adapter for Financial Applications
//!
//! Provides real-time market data ingestion from financial APIs

use super::super::ingestion::types::{DataPoint, DataSource, SourceInfo};
use anyhow::Result;
use chrono::Utc;
use futures::future::BoxFuture;
use std::collections::HashMap;

/// Alpaca Market Data Source
///
/// Connects to Alpaca Markets API for real-time stock trade data
pub struct AlpacaMarketDataSource {
    api_key: String,
    api_secret: String,
    symbols: Vec<String>,
    client: Option<reqwest::Client>,
    base_url: String,
}

impl AlpacaMarketDataSource {
    /// Create a new Alpaca market data source
    pub fn new(api_key: String, api_secret: String, symbols: Vec<String>) -> Self {
        Self {
            api_key,
            api_secret,
            symbols,
            client: None,
            base_url: "https://data.alpaca.markets/v2/stocks".to_string(),
        }
    }

    /// Parse trade data from API response
    fn parse_trade_data(&self, symbol: &str, data: &serde_json::Value) -> Option<DataPoint> {
        let trade = data.get("trade")?;

        let price = trade["p"].as_f64()?;
        let volume = trade["s"].as_f64()?;

        let timestamp = trade["t"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.timestamp_millis())
            .unwrap_or_else(|| Utc::now().timestamp_millis());

        let mut metadata = HashMap::new();
        metadata.insert("symbol".to_string(), symbol.to_string());
        metadata.insert("source".to_string(), "alpaca".to_string());
        metadata.insert("type".to_string(), "trade".to_string());

        Some(DataPoint {
            timestamp,
            values: vec![price, volume],
            metadata,
        })
    }
}

impl DataSource for AlpacaMarketDataSource {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = Some(reqwest::Client::new());
            log::info!("Connected to Alpaca Market Data API");
            Ok(())
        })
    }

    fn read_batch(&mut self) -> BoxFuture<'_, Result<Vec<DataPoint>>> {
        Box::pin(async move {
            let client = self
                .client
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Not connected to Alpaca API"))?;

            let mut points = Vec::new();

            for symbol in &self.symbols {
                let url = format!("{}/{}/trades/latest", self.base_url, symbol);

                match client
                    .get(&url)
                    .header("APCA-API-KEY-ID", &self.api_key)
                    .header("APCA-API-SECRET-KEY", &self.api_secret)
                    .send()
                    .await
                {
                    Ok(response) => {
                        if let Ok(data) = response.json::<serde_json::Value>().await {
                            if let Some(point) = self.parse_trade_data(symbol, &data) {
                                points.push(point);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to fetch data for {}: {}", symbol, e);
                    }
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            Ok(points)
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = None;
            log::info!("Disconnected from Alpaca Market Data API");
            Ok(())
        })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: "Alpaca Market Data".to_string(),
            data_type: "financial_trades".to_string(),
            sampling_rate_hz: 10.0,
            dimensions: 2, // price, volume
        }
    }

    fn is_connected(&self) -> bool {
        self.client.is_some()
    }
}

/// Generic REST API Market Data Source
///
/// Configurable source for any REST API returning market data
pub struct RestApiMarketDataSource {
    name: String,
    base_url: String,
    symbols: Vec<String>,
    headers: HashMap<String, String>,
    client: Option<reqwest::Client>,
    parse_fn: fn(&str, &serde_json::Value) -> Option<DataPoint>,
}

impl RestApiMarketDataSource {
    /// Create a new REST API market data source
    pub fn new(
        name: String,
        base_url: String,
        symbols: Vec<String>,
        headers: HashMap<String, String>,
        parse_fn: fn(&str, &serde_json::Value) -> Option<DataPoint>,
    ) -> Self {
        Self {
            name,
            base_url,
            symbols,
            headers,
            client: None,
            parse_fn,
        }
    }
}

impl DataSource for RestApiMarketDataSource {
    fn connect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = Some(reqwest::Client::new());
            log::info!("Connected to {} API", self.name);
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

            for symbol in &self.symbols {
                let url = format!("{}/{}", self.base_url, symbol);

                let mut request = client.get(&url);
                for (key, value) in &self.headers {
                    request = request.header(key, value);
                }

                match request.send().await {
                    Ok(response) => {
                        if let Ok(data) = response.json::<serde_json::Value>().await {
                            if let Some(point) = (self.parse_fn)(symbol, &data) {
                                points.push(point);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to fetch data for {}: {}", symbol, e);
                    }
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            Ok(points)
        })
    }

    fn disconnect(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.client = None;
            log::info!("Disconnected from {} API", self.name);
            Ok(())
        })
    }

    fn get_source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name.clone(),
            data_type: "financial_data".to_string(),
            sampling_rate_hz: 10.0,
            dimensions: 2,
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
    fn test_alpaca_source_creation() {
        let source = AlpacaMarketDataSource::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            vec!["AAPL".to_string(), "GOOGL".to_string()],
        );

        let info = source.get_source_info();
        assert_eq!(info.name, "Alpaca Market Data");
        assert_eq!(info.dimensions, 2);
        assert!(!source.is_connected());
    }
}
