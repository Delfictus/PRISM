use super::contract::TelemetryEntry;
use super::logger::TelemetryLogger;
use std::sync::Arc;

#[derive(Clone)]
pub struct TelemetrySink {
    logger: Arc<TelemetryLogger>,
}

impl TelemetrySink {
    pub fn new(component: &str) -> Self {
        let logger = TelemetryLogger::with_path(component, "telemetry/contract.jsonl");
        Self {
            logger: Arc::new(logger),
        }
    }

    pub fn log(&self, entry: &TelemetryEntry) {
        self.logger.log_telemetry_entry(entry);
    }
}
