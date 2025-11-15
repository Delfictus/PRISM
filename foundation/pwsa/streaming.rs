//! Async Streaming Telemetry Ingestion for PWSA
//!
//! **Week 2 Enhancement:** Real-time streaming architecture with Tokio
//!
//! Supports continuous telemetry streams from all three layers:
//! - Transport Layer: 154 satellites × 4 links × 10Hz = 6,160 messages/second
//! - Tracking Layer: 35 satellites × 10Hz = 350 messages/second
//! - Ground Layer: 5 stations × 1Hz = 5 messages/second
//!
//! Total ingestion rate: ~6,500 messages/second

use super::satellite_adapters::*;
use anyhow::Result;
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration, Instant};

/// Streaming PWSA fusion platform with async telemetry ingestion
///
/// Processes continuous telemetry streams in real-time while maintaining
/// <1ms fusion latency.
pub struct StreamingPwsaFusionPlatform {
    /// Core fusion platform
    fusion_core: PwsaFusionPlatform,

    /// Input channels
    transport_rx: mpsc::Receiver<OctTelemetry>,
    tracking_rx: mpsc::Receiver<IrSensorFrame>,
    ground_rx: mpsc::Receiver<GroundStationData>,

    /// Output channel
    output_tx: mpsc::Sender<MissionAwareness>,

    /// Buffering for synchronized fusion
    transport_buffer: Option<OctTelemetry>,
    tracking_buffer: Option<IrSensorFrame>,
    ground_buffer: Option<GroundStationData>,

    /// Backpressure controller
    rate_limiter: RateLimiter,

    /// Statistics
    fusions_completed: u64,
    total_latency_us: u128,
}

impl StreamingPwsaFusionPlatform {
    /// Create new streaming platform
    pub fn new(
        transport_rx: mpsc::Receiver<OctTelemetry>,
        tracking_rx: mpsc::Receiver<IrSensorFrame>,
        ground_rx: mpsc::Receiver<GroundStationData>,
        output_tx: mpsc::Sender<MissionAwareness>,
        max_rate_hz: f64,
    ) -> Result<Self> {
        Ok(Self {
            fusion_core: PwsaFusionPlatform::new_tranche1()?,
            transport_rx,
            tracking_rx,
            ground_rx,
            output_tx,
            transport_buffer: None,
            tracking_buffer: None,
            ground_buffer: None,
            rate_limiter: RateLimiter::new(max_rate_hz),
            fusions_completed: 0,
            total_latency_us: 0,
        })
    }

    /// Run the streaming fusion loop
    ///
    /// Continuously processes incoming telemetry and outputs mission awareness.
    /// Blocks until all input channels are closed.
    pub async fn run(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                Some(telem) = self.transport_rx.recv() => {
                    self.transport_buffer = Some(telem);
                    self.try_fusion().await?;
                }
                Some(frame) = self.tracking_rx.recv() => {
                    self.tracking_buffer = Some(frame);
                    self.try_fusion().await?;
                }
                Some(data) = self.ground_rx.recv() => {
                    self.ground_buffer = Some(data);
                    self.try_fusion().await?;
                }
                else => {
                    // All channels closed
                    break;
                }
            }
        }

        Ok(())
    }

    /// Attempt fusion if all three layers have data
    async fn try_fusion(&mut self) -> Result<()> {
        if self.transport_buffer.is_some()
            && self.tracking_buffer.is_some()
            && self.ground_buffer.is_some()
        {
            // Check backpressure
            self.rate_limiter.check_and_wait().await?;

            let start = Instant::now();

            // Perform fusion
            let awareness = self.fusion_core.fuse_mission_data(
                self.transport_buffer.as_ref().unwrap(),
                self.tracking_buffer.as_ref().unwrap(),
                self.ground_buffer.as_ref().unwrap(),
            )?;

            let latency = start.elapsed();

            // Update statistics
            self.fusions_completed += 1;
            self.total_latency_us += latency.as_micros();

            // Send output
            self.output_tx
                .send(awareness)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send output: {:?}", e))?;

            // Clear buffers for next fusion
            self.transport_buffer = None;
            self.tracking_buffer = None;
            self.ground_buffer = None;
        }

        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> StreamingStats {
        StreamingStats {
            fusions_completed: self.fusions_completed,
            average_latency_us: if self.fusions_completed > 0 {
                (self.total_latency_us / self.fusions_completed as u128) as u64
            } else {
                0
            },
            total_latency_us: self.total_latency_us,
        }
    }
}

/// Performance statistics for streaming platform
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub fusions_completed: u64,
    pub average_latency_us: u64,
    pub total_latency_us: u128,
}

/// Rate limiter for backpressure control
///
/// Prevents system overload by enforcing maximum fusion rate.
/// Uses token bucket algorithm for smooth rate limiting.
pub struct RateLimiter {
    max_rate_hz: f64,
    window: VecDeque<Instant>,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_rate_hz: f64) -> Self {
        Self {
            max_rate_hz,
            window: VecDeque::new(),
            window_duration: Duration::from_secs(1),
        }
    }

    /// Check rate and wait if necessary
    ///
    /// Implements sliding window rate limiting.
    /// Sleeps if rate would be exceeded.
    pub async fn check_and_wait(&mut self) -> Result<()> {
        let now = Instant::now();

        // Remove timestamps outside window
        while let Some(&ts) = self.window.front() {
            if now.duration_since(ts) > self.window_duration {
                self.window.pop_front();
            } else {
                break;
            }
        }

        // Check if rate would be exceeded
        let max_in_window = (self.max_rate_hz * self.window_duration.as_secs_f64()) as usize;
        if self.window.len() >= max_in_window {
            // Calculate sleep time
            if let Some(&oldest) = self.window.front() {
                let time_until_free = self
                    .window_duration
                    .saturating_sub(now.duration_since(oldest));

                if !time_until_free.is_zero() {
                    tokio::time::sleep(time_until_free).await;
                }
            }
        }

        self.window.push_back(now);
        Ok(())
    }

    /// Get current rate (fusions/second)
    pub fn current_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }

        let now = Instant::now();
        let window_start = self.window.front().unwrap();
        let elapsed = now.duration_since(*window_start).as_secs_f64();

        if elapsed > 0.0 {
            self.window.len() as f64 / elapsed
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10.0); // 10 Hz

        let start = Instant::now();

        // Try to execute 20 times (should take ~2 seconds at 10 Hz)
        for _ in 0..20 {
            limiter.check_and_wait().await.unwrap();
        }

        let elapsed = start.elapsed();

        // Should take approximately 2 seconds
        assert!(elapsed >= Duration::from_millis(1800)); // Allow some variance
        assert!(elapsed <= Duration::from_millis(2500));
    }
}
