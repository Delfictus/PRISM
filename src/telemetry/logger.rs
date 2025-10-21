use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::json;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct TelemetryLogger {
    component: String,
    state: Arc<Mutex<LoggerState>>,
    fsync_interval: Duration,
    expected_stages: Arc<Vec<String>>,
    alert_webhook: Option<String>,
}

struct LoggerState {
    writer: BufWriter<std::fs::File>,
    last_fsync: Instant,
    seen_stages: HashSet<String>,
    last_missing_alert: Option<Vec<String>>,
}

impl TelemetryLogger {
    pub fn new(component: &str) -> Self {
        Self::with_path(component, "telemetry/device_guard.jsonl")
    }

    pub fn with_path(component: &str, path: &str) -> Self {
        let fsync_interval = read_fsync_interval();
        let expected_stages = Arc::new(read_expected_stages());
        let alert_webhook = std::env::var("TELEMETRY_ALERT_WEBHOOK").ok();

        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("telemetry file");
        let state = LoggerState {
            writer: BufWriter::new(file),
            last_fsync: Instant::now(),
            seen_stages: HashSet::new(),
            last_missing_alert: None,
        };
        Self {
            component: component.to_string(),
            state: Arc::new(Mutex::new(state)),
            fsync_interval,
            expected_stages,
            alert_webhook,
        }
    }

    pub fn log<T: Serialize>(&self, event: T) {
        let entry = json!({
            "timestamp_us": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros())
                .unwrap_or_default(),
            "component": self.component,
            "event": event,
        });

        if let Ok(line) = serde_json::to_string(&entry) {
            self.write_line(&line, None);
        }
    }

    pub fn log_telemetry_entry(&self, entry: &crate::telemetry::TelemetryEntry) {
        match serde_json::to_string(entry) {
            Ok(line) => {
                let stage = extract_stage(&entry.event);
                self.write_line(&line, stage);
            }
            Err(err) => {
                eprintln!("Telemetry serialization failed: {}", err);
            }
        }
    }

    fn write_line(&self, line: &str, stage: Option<&'static str>) {
        let mut guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                eprintln!("Telemetry logger lock poisoned");
                poisoned.into_inner()
            }
        };

        if let Err(err) = writeln!(guard.writer, "{}", line) {
            eprintln!("Telemetry write failed: {}", err);
            return;
        }

        if let Some(stage_name) = stage {
            guard.seen_stages.insert(stage_name.to_string());
        }

        if let Err(err) = guard.writer.flush() {
            eprintln!("Telemetry flush failed: {}", err);
        }

        if guard.last_fsync.elapsed() >= self.fsync_interval {
            if let Err(err) = guard.writer.get_ref().sync_all() {
                eprintln!("Telemetry fsync failed: {}", err);
            }
            guard.last_fsync = Instant::now();
        }

        drop(guard);
        self.check_missing_stages();
    }

    fn check_missing_stages(&self) {
        let mut guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let missing: Vec<String> = self
            .expected_stages
            .iter()
            .filter(|stage| !guard.seen_stages.contains(*stage))
            .cloned()
            .collect();

        if missing.is_empty() {
            guard.last_missing_alert = None;
            return;
        }

        let should_alert = match &guard.last_missing_alert {
            Some(previous) if previous == &missing => false,
            _ => true,
        };

        if should_alert {
            eprintln!(
                "⚠️  TELEMETRY ALERT: Missing critical stages: {}",
                missing.join(", ")
            );

            if let Some(webhook) = &self.alert_webhook {
                if let Err(err) = send_alert(webhook, &missing) {
                    eprintln!("Telemetry alert webhook failed: {}", err);
                }
            }

            guard.last_missing_alert = Some(missing);
        }
    }
}

fn read_fsync_interval() -> Duration {
    std::env::var("TELEMETRY_FSYNC_INTERVAL")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(5))
}

fn read_expected_stages() -> Vec<String> {
    if let Ok(raw) = std::env::var("TELEMETRY_EXPECTED_STAGES") {
        raw.split(',')
            .map(|stage| stage.trim().to_string())
            .filter(|stage| !stage.is_empty())
            .collect()
    } else {
        vec![
            "adapter_started".to_string(),
            "processing_started".to_string(),
            "consensus_proposed".to_string(),
            "consensus_reached".to_string(),
            "processing_completed".to_string(),
            "meta_variant_emitted".to_string(),
        ]
    }
}

fn extract_stage(event: &crate::telemetry::EventData) -> Option<&'static str> {
    use crate::telemetry::EventData::*;
    match event {
        AdapterStarted { .. } => Some("adapter_started"),
        AdapterProcessed { .. } => Some("processing_completed"),
        AdapterFailed { .. } => Some("processing_failed"),
        ProcessingStarted { .. } => Some("processing_started"),
        ProcessingCompleted { .. } => Some("processing_completed"),
        ProcessingFailed { .. } => Some("processing_failed"),
        ConsensusProposed { .. } => Some("consensus_proposed"),
        ConsensusReached { .. } => Some("consensus_reached"),
        ConsensusConflict { .. } => Some("consensus_conflict"),
        Custom { payload } => {
            if payload.get("meta_variant").is_some() {
                Some("meta_variant_emitted")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn send_alert(webhook: &str, missing: &[String]) -> Result<(), String> {
    let client = Client::new();
    let payload = json!({
        "severity": "warning",
        "missing_stages": missing,
        "timestamp": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default(),
    });

    client
        .post(webhook)
        .json(&payload)
        .send()
        .and_then(|response| response.error_for_status())
        .map(|_| ())
        .map_err(|err| err.to_string())
}
