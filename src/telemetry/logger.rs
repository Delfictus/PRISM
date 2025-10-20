use serde::Serialize;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct TelemetryLogger {
    component: String,
    writer: Arc<Mutex<std::fs::File>>,
}

impl TelemetryLogger {
    pub fn new(component: &str) -> Self {
        Self::with_path(component, "telemetry/device_guard.jsonl")
    }

    pub fn with_path(component: &str, path: &str) -> Self {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("telemetry file");
        Self {
            component: component.to_string(),
            writer: Arc::new(Mutex::new(file)),
        }
    }

    pub fn log<T: Serialize>(&self, event: T) {
        if let Ok(mut writer) = self.writer.lock() {
            let entry = json!({
                "timestamp_us": SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_micros())
                    .unwrap_or_default(),
                "component": self.component,
                "event": event,
            });
            if let Err(err) = writeln!(writer, "{}", entry.to_string()) {
                eprintln!("Telemetry write failed: {}", err);
            }
        }
    }
}
