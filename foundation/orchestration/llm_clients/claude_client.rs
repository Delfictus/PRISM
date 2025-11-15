//! Anthropic Claude Client - Production Grade
//!
//! Mission Charlie: Task 1.2

use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::time::{sleep, timeout, Duration, Instant};

use super::{LLMResponse, Usage};

/// Anthropic Claude client
pub struct ClaudeClient {
    api_key: String,
    http_client: Client,
    base_url: String,
    cache: Arc<DashMap<String, CachedResponse>>,
    token_counter: Arc<Mutex<TokenCounter>>,
    max_retries: usize,
    retry_delay_ms: u64,
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    messages: Vec<ClaudeMessage>,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    id: String,
    content: Vec<ContentBlock>,
    usage: ClaudeUsage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ClaudeUsage {
    input_tokens: usize,
    output_tokens: usize,
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

    fn add_usage(&mut self, tokens: usize, cost: f64) {
        self.total_tokens += tokens;
        self.total_cost += cost;
    }
}

impl ClaudeClient {
    pub fn new(api_key: String) -> Result<Self> {
        Ok(Self {
            api_key,
            http_client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            base_url: "https://api.anthropic.com/v1".to_string(),
            cache: Arc::new(DashMap::new()),
            token_counter: Arc::new(Mutex::new(TokenCounter::new())),
            max_retries: 3,
            retry_delay_ms: 1000,
        })
    }

    pub async fn generate(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let start = Instant::now();

        // Check cache
        let cache_key = format!("{}-{}", prompt, temperature);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(LLMResponse {
                cached: true,
                latency: start.elapsed(),
                ..cached.response.clone()
            });
        }

        // Retry loop
        for attempt in 0..self.max_retries {
            match self.make_request(prompt, temperature).await {
                Ok(mut response) => {
                    response.latency = start.elapsed();

                    // Cache response
                    self.cache.insert(
                        cache_key,
                        CachedResponse {
                            response: response.clone(),
                            timestamp: SystemTime::now(),
                        },
                    );

                    // Track cost
                    let cost = self.calculate_cost(&response.usage);
                    self.token_counter
                        .lock()
                        .add_usage(response.usage.total_tokens, cost);

                    return Ok(response);
                }
                Err(e) => {
                    if attempt < self.max_retries - 1 {
                        let delay = self.retry_delay_ms * 2_u64.pow(attempt as u32);
                        sleep(Duration::from_millis(delay)).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        bail!("All retries failed")
    }

    async fn make_request(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let request = ClaudeRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: 1000,
            temperature,
        };

        let response = timeout(
            Duration::from_secs(30),
            self.http_client
                .post(&format!("{}/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&request)
                .send(),
        )
        .await??;

        if !response.status().is_success() {
            bail!("Claude API error: {}", response.status());
        }

        let api_response: ClaudeResponse = response.json().await?;

        let text = api_response
            .content
            .iter()
            .filter(|c| c.content_type == "text")
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(LLMResponse {
            model: "claude-3.5-sonnet".to_string(),
            text,
            usage: Usage {
                prompt_tokens: api_response.usage.input_tokens,
                completion_tokens: api_response.usage.output_tokens,
                total_tokens: api_response.usage.input_tokens + api_response.usage.output_tokens,
            },
            latency: Duration::from_secs(0),
            cached: false,
        })
    }

    fn calculate_cost(&self, usage: &Usage) -> f64 {
        // Claude 3.5 Sonnet pricing
        const PROMPT_COST_PER_1K: f64 = 0.003;
        const COMPLETION_COST_PER_1K: f64 = 0.015;

        (usage.prompt_tokens as f64 / 1000.0) * PROMPT_COST_PER_1K
            + (usage.completion_tokens as f64 / 1000.0) * COMPLETION_COST_PER_1K
    }

    pub fn get_total_cost(&self) -> f64 {
        self.token_counter.lock().total_cost
    }

    pub fn get_total_tokens(&self) -> usize {
        self.token_counter.lock().total_tokens
    }
}
