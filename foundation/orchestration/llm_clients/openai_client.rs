//! OpenAI GPT-4 Client - Production Grade
//!
//! Mission Charlie: Task 1.1
//!
//! Features:
//! - Retry logic with exponential backoff
//! - Rate limiting (60 req/min for GPT-4)
//! - Response caching (1-hour TTL)
//! - Token counting (cost tracking)
//! - Comprehensive error handling
//! - Async/await (non-blocking)

use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::time::{sleep, timeout, Duration, Instant};

/// OpenAI GPT-4 client (production-grade)
pub struct OpenAIClient {
    api_key: String,
    http_client: Client,
    base_url: String,

    /// Rate limiter (60 requests/minute)
    rate_limiter: Arc<RateLimiter>,

    /// Response cache (semantic similarity based - will be enhanced in Task 1.7)
    cache: Arc<DashMap<String, CachedResponse>>,

    /// Cost tracker
    token_counter: Arc<Mutex<TokenCounter>>,

    /// Retry configuration
    max_retries: usize,
    retry_delay_ms: u64,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Clone)]
struct OpenAIResponse {
    id: String,
    #[allow(dead_code)]
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize, Clone)]
struct Choice {
    #[allow(dead_code)]
    index: usize,
    message: Message,
    #[allow(dead_code)]
    finish_reason: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
struct CachedResponse {
    response: LLMResponse,
    timestamp: SystemTime,
    ttl: Duration,
}

impl CachedResponse {
    fn is_fresh(&self) -> bool {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or(Duration::from_secs(0))
            < self.ttl
    }
}

/// LLM Response (unified format)
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub model: String,
    pub text: String,
    pub usage: Usage,
    pub latency: Duration,
    pub cached: bool,
}

/// Rate limiter (token bucket algorithm)
pub struct RateLimiter {
    max_rate: f64, // requests per second
    last_request: Mutex<Instant>,
}

impl RateLimiter {
    pub fn new(requests_per_minute: f64) -> Self {
        Self {
            max_rate: requests_per_minute / 60.0,
            last_request: Mutex::new(Instant::now()),
        }
    }

    pub async fn wait_if_needed(&self) -> Result<()> {
        let mut last = self.last_request.lock();
        let elapsed = last.elapsed();
        let min_interval = Duration::from_secs_f64(1.0 / self.max_rate);

        if elapsed < min_interval {
            let wait_time = min_interval - elapsed;
            drop(last); // Release lock before sleeping
            sleep(wait_time).await;
        }

        *self.last_request.lock() = Instant::now();
        Ok(())
    }
}

/// Token counter (cost tracking)
pub struct TokenCounter {
    total_tokens: usize,
    total_cost_usd: f64,
}

impl TokenCounter {
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            total_cost_usd: 0.0,
        }
    }

    pub fn add_usage(&mut self, usage: &Usage, cost: f64) {
        self.total_tokens += usage.total_tokens;
        self.total_cost_usd += cost;
    }

    pub fn get_total_cost(&self) -> f64 {
        self.total_cost_usd
    }

    pub fn get_total_tokens(&self) -> usize {
        self.total_tokens
    }
}

impl OpenAIClient {
    /// Create new OpenAI client
    pub fn new(api_key: String) -> Result<Self> {
        Ok(Self {
            api_key,
            http_client: Client::builder()
                .timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(10)
                .build()
                .context("Failed to build HTTP client")?,
            base_url: "https://api.openai.com/v1".to_string(),
            rate_limiter: Arc::new(RateLimiter::new(60.0)), // 60 req/min
            cache: Arc::new(DashMap::new()),
            token_counter: Arc::new(Mutex::new(TokenCounter::new())),
            max_retries: 3,
            retry_delay_ms: 1000,
        })
    }

    /// Query GPT-4 with full production features
    ///
    /// Features:
    /// - Caching (1-hour TTL)
    /// - Rate limiting (60 req/min)
    /// - Retry logic (3 attempts, exponential backoff)
    /// - Cost tracking
    /// - Comprehensive error handling
    pub async fn generate(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let start = Instant::now();

        // 1. Check cache first
        let cache_key = self.compute_cache_key(prompt, temperature);

        if let Some(cached) = self.cache.get(&cache_key) {
            if cached.is_fresh() {
                // Cache hit
                return Ok(LLMResponse {
                    cached: true,
                    latency: start.elapsed(),
                    ..cached.response.clone()
                });
            } else {
                // Expired - remove
                self.cache.remove(&cache_key);
            }
        }

        // 2. Rate limiting (wait if necessary)
        self.rate_limiter.wait_if_needed().await?;

        // 3. Retry loop with exponential backoff
        let mut last_error = None;

        for attempt in 0..self.max_retries {
            match self.make_api_request(prompt, temperature).await {
                Ok(mut response) => {
                    // Cache successful response
                    response.latency = start.elapsed();
                    response.cached = false;

                    self.cache.insert(
                        cache_key,
                        CachedResponse {
                            response: response.clone(),
                            timestamp: SystemTime::now(),
                            ttl: Duration::from_secs(3600), // 1 hour
                        },
                    );

                    // Track cost
                    let cost = self.calculate_cost(&response.usage);
                    self.token_counter.lock().add_usage(&response.usage, cost);

                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.max_retries - 1 {
                        // Exponential backoff: 1s, 2s, 4s
                        let delay = self.retry_delay_ms * 2_u64.pow(attempt as u32);
                        sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "All {} retries failed: {:?}",
            self.max_retries,
            last_error
        ))
    }

    async fn make_api_request(&self, prompt: &str, temperature: f32) -> Result<LLMResponse> {
        let request = OpenAIRequest {
            model: "gpt-4-turbo-preview".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: "You are a geopolitical intelligence analyst specializing in missile threat assessment.".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }
            ],
            temperature,
            max_tokens: 1000,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
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
        .await
        .context("Request timeout")??;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            bail!("OpenAI API error {}: {}", status, error_body);
        }

        let api_response: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        if api_response.choices.is_empty() {
            bail!("OpenAI returned no choices");
        }

        Ok(LLMResponse {
            model: "gpt-4".to_string(),
            text: api_response.choices[0].message.content.clone(),
            usage: api_response.usage,
            latency: Duration::from_secs(0), // Will be set by caller
            cached: false,
        })
    }

    fn compute_cache_key(&self, prompt: &str, temperature: f32) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        temperature.to_bits().hash(&mut hasher);

        format!("openai-gpt4-{}", hasher.finish())
    }

    fn calculate_cost(&self, usage: &Usage) -> f64 {
        // GPT-4-turbo pricing (January 2025)
        const PROMPT_COST_PER_1K: f64 = 0.01;
        const COMPLETION_COST_PER_1K: f64 = 0.03;

        let prompt_cost = (usage.prompt_tokens as f64 / 1000.0) * PROMPT_COST_PER_1K;
        let completion_cost = (usage.completion_tokens as f64 / 1000.0) * COMPLETION_COST_PER_1K;

        prompt_cost + completion_cost
    }

    /// Get total cost (for monitoring)
    pub fn get_total_cost(&self) -> f64 {
        self.token_counter.lock().get_total_cost()
    }

    /// Get total tokens processed
    pub fn get_total_tokens(&self) -> usize {
        self.token_counter.lock().get_total_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_api_key() -> String {
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "test-key".to_string())
    }

    #[tokio::test]
    async fn test_openai_client_creation() {
        let client = OpenAIClient::new(get_test_api_key());
        assert!(client.is_ok());
    }

    #[test]
    fn test_cache_key_generation() {
        let client = OpenAIClient::new("test-key".to_string()).unwrap();

        let key1 = client.compute_cache_key("test prompt", 0.7);
        let key2 = client.compute_cache_key("test prompt", 0.7);
        let key3 = client.compute_cache_key("different prompt", 0.7);

        // Same prompt should generate same key
        assert_eq!(key1, key2);

        // Different prompt should generate different key
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cost_calculation() {
        let client = OpenAIClient::new("test-key".to_string()).unwrap();

        let usage = Usage {
            prompt_tokens: 500,
            completion_tokens: 200,
            total_tokens: 700,
        };

        let cost = client.calculate_cost(&usage);

        // Expected: (500/1000)*0.01 + (200/1000)*0.03 = 0.005 + 0.006 = 0.011
        assert!(
            (cost - 0.011).abs() < 0.001,
            "Cost calculation incorrect: {}",
            cost
        );
    }

    #[test]
    fn test_cached_response_freshness() {
        let response = LLMResponse {
            model: "gpt-4".to_string(),
            text: "Test response".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
            latency: Duration::from_secs(1),
            cached: false,
        };

        let cached = CachedResponse {
            response,
            timestamp: SystemTime::now(),
            ttl: Duration::from_secs(3600),
        };

        assert!(cached.is_fresh());

        // Create expired cache entry
        let expired = CachedResponse {
            response: cached.response.clone(),
            timestamp: SystemTime::now() - Duration::from_secs(7200), // 2 hours ago
            ttl: Duration::from_secs(3600),                           // 1 hour TTL
        };

        assert!(!expired.is_fresh());
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(120.0); // 120 req/min = 2 req/s

        let start = Instant::now();

        // Make 3 requests (should take ~1 second due to rate limiting)
        for _ in 0..3 {
            limiter.wait_if_needed().await.unwrap();
        }

        let elapsed = start.elapsed();

        // Should take at least 1 second (2 requests/second)
        assert!(
            elapsed >= Duration::from_millis(800),
            "Rate limiting not working"
        );
    }
}
