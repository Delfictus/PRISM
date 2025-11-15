//! xAI Grok-4 Client - Production Grade
//!
//! Mission Charlie: Task 1.4 (replaces Local Llama)

use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::time::{sleep, timeout, Duration, Instant};

use super::{LLMResponse, Usage};

/// xAI Grok-4 client
pub struct GrokClient {
    api_key: String,
    http_client: Client,
    base_url: String,
    cache: Arc<DashMap<String, CachedResponse>>,
    token_counter: Arc<Mutex<TokenCounter>>,
    max_retries: usize,
}

#[derive(Debug, Serialize)]
struct GrokRequest {
    model: String,
    messages: Vec<GrokMessage>,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GrokMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct GrokResponse {
    choices: Vec<GrokChoice>,
    usage: GrokUsage,
}

#[derive(Debug, Deserialize)]
struct GrokChoice {
    message: GrokMessage,
}

#[derive(Debug, Deserialize, Clone)]
struct GrokUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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

impl GrokClient {
    pub fn new(api_key: String) -> Result<Self> {
        Ok(Self {
            api_key,
            http_client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            base_url: "https://api.x.ai/v1".to_string(),
            cache: Arc::new(DashMap::new()),
            token_counter: Arc::new(Mutex::new(TokenCounter::new())),
            max_retries: 3,
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

                    return Ok(response);
                }
                Err(e) => {
                    if attempt < self.max_retries - 1 {
                        sleep(Duration::from_millis(1000 * 2_u64.pow(attempt as u32))).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        bail!("Retries exhausted")
    }

    async fn make_request(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let request = GrokRequest {
            model: "grok-4-latest".to_string(),
            messages: vec![
                GrokMessage {
                    role: "system".to_string(),
                    content: "You are a technical intelligence analyst.".to_string(),
                },
                GrokMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            temperature,
            stream: false,
        };

        let response = timeout(
            Duration::from_secs(30),
            self.http_client
                .post(&format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send(),
        )
        .await??;

        if !response.status().is_success() {
            bail!("Grok API error: {}", response.status());
        }

        let api_response: GrokResponse = response.json().await?;

        Ok(LLMResponse {
            model: "grok-4".to_string(),
            text: api_response.choices[0].message.content.clone(),
            usage: Usage {
                prompt_tokens: api_response.usage.prompt_tokens,
                completion_tokens: api_response.usage.completion_tokens,
                total_tokens: api_response.usage.total_tokens,
            },
            latency: Duration::from_secs(0),
            cached: false,
        })
    }

    fn calculate_cost(&self, usage: &Usage) -> f64 {
        // Grok pricing (estimate based on similar models)
        const COST_PER_1K: f64 = 0.01;
        (usage.total_tokens as f64 / 1000.0) * COST_PER_1K
    }

    pub fn get_total_cost(&self) -> f64 {
        self.token_counter.lock().total_cost
    }

    pub fn get_total_tokens(&self) -> usize {
        self.token_counter.lock().total_tokens
    }
}
