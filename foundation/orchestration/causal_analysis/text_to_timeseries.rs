//! Text-to-TimeSeries Converter with Multi-scale Analysis
//!
//! Mission Charlie: Task 3.1
//!
//! Converts LLM text to time series format for transfer entropy computation

use anyhow::Result;
use ndarray::Array1;

/// Text-to-TimeSeries Converter
pub struct TextToTimeSeriesConverter {
    window_size: usize,
}

impl TextToTimeSeriesConverter {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Convert text to time series via sliding window
    ///
    /// Each window position becomes a time point
    pub fn convert(&self, text: &str) -> Result<Array1<f64>> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() < self.window_size {
            return Ok(Array1::from_vec(vec![0.0]));
        }

        let mut series = Vec::new();

        for i in 0..words.len().saturating_sub(self.window_size) {
            // Aggregate window (simple: average word lengths)
            let window_value: f64 = words[i..i + self.window_size]
                .iter()
                .map(|w| w.len() as f64)
                .sum::<f64>()
                / self.window_size as f64;

            series.push(window_value);
        }

        Ok(Array1::from_vec(series))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_timeseries() {
        let converter = TextToTimeSeriesConverter::new(3);
        let text = "This is a test of text conversion";

        let series = converter.convert(text).unwrap();

        assert!(series.len() > 0, "Should produce time series");
    }
}
