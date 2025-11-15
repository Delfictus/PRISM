//! Google Gemini Client - Production Grade
//!
//! Mission Charlie: Task 1.3

use anyhow::{bail, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::time::{sleep, timeout, Duration, Instant};

use super::{LLMResponse, Usage};

/// Google Gemini client
pub struct GeminiClient {
    api_key: String,
    http_client: Client,
    base_url: String,
    cache: Arc<DashMap<String, CachedResponse>>,
    token_counter: Arc<Mutex<TokenCounter>>,
}

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: UsageMetadata,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: ContentResponse,
}

#[derive(Debug, Deserialize)]
struct ContentResponse {
    parts: Vec<PartResponse>,
}

#[derive(Debug, Deserialize)]
struct PartResponse {
    text: String,
}

#[derive(Debug, Deserialize, Clone)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: usize,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: usize,
    #[serde(rename = "totalTokenCount")]
    total_token_count: usize,
}

#[derive(Clone)]
struct CachedResponse {
    response: LLMResponse,
    timestamp: SystemTime,
}

struct TokenCounter {
    total_tokens: usize,
    total_cost: f64,
}

impl TokenCounter {
    fn new() -> Self {
        Self {
            total_tokens: 0,
            total_cost: 0.0,
        }
    }
}

impl GeminiClient {
    pub fn new(api_key: String) -> Result<Self> {
        Ok(Self {
            api_key,
            http_client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            cache: Arc::new(DashMap::new()),
            token_counter: Arc::new(Mutex::new(TokenCounter::new())),
        })
    }

    pub async fn generate(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let start = Instant::now();

        let cache_key = format!("{}-{}", prompt, temperature);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(LLMResponse {
                cached: true,
                latency: start.elapsed(),
                ..cached.response.clone()
            });
        }

        let mut response = self.make_request(prompt, temperature).await?;
        response.latency = start.elapsed();

        self.cache.insert(
            cache_key,
            CachedResponse {
                response: response.clone(),
                timestamp: SystemTime::now(),
            },
        );

        let cost = self.calculate_cost(&response.usage);
        self.token_counter.lock().total_tokens += response.usage.total_tokens;
        self.token_counter.lock().total_cost += cost;

        Ok(response)
    }

    async fn make_request(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let request = GeminiRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: GenerationConfig {
                temperature,
                max_output_tokens: 1000,
            },
        };

        let url = format!(
            "{}/models/gemini-2.0-flash-exp:generateContent?key={}",
            self.base_url, self.api_key
        );

        let response = timeout(
            Duration::from_secs(30),
            self.http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send(),
        )
        .await??;

        if !response.status().is_success() {
            bail!("Gemini API error: {}", response.status());
        }

        let api_response: GeminiResponse = response.json().await?;

        let text = api_response.candidates[0]
            .content
            .parts
            .iter()
            .map(|p| p.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(LLMResponse {
            model: "gemini-2.0-flash".to_string(),
            text,
            usage: Usage {
                prompt_tokens: api_response.usage_metadata.prompt_token_count,
                completion_tokens: api_response.usage_metadata.candidates_token_count,
                total_tokens: api_response.usage_metadata.total_token_count,
            },
            latency: Duration::from_secs(0),
            cached: false,
        })
    }

    fn calculate_cost(&self, usage: &Usage) -> f64 {
        const COST_PER_1K: f64 = 0.0001; // Very cheap
        (usage.total_tokens as f64 / 1000.0) * COST_PER_1K
    }

    pub fn get_total_cost(&self) -> f64 {
        self.token_counter.lock().total_cost
    }

    pub fn get_total_tokens(&self) -> usize {
        self.token_counter.lock().total_tokens
    }
}
