use super::contract::{ComponentId, EventData, EventLevel, Metrics, TelemetryEntry};
use super::sink::TelemetrySink;
use uuid::Uuid;

pub trait TelemetryProvider {
    fn component_id(&self) -> ComponentId;
    fn telemetry_sink(&self) -> &TelemetrySink;

    fn correlation_id(&self) -> Option<Uuid> {
        None
    }

    fn capture_metrics(&self) -> Option<Metrics> {
        None
    }

    fn log_event(&self, level: EventLevel, event: EventData) -> TelemetryEntry {
        let entry = TelemetryEntry::new(
            self.component_id(),
            level,
            event,
            self.correlation_id(),
            self.capture_metrics(),
        );
        self.telemetry_sink().log(&entry);
        entry
    }
}
