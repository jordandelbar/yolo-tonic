use opentelemetry::{
    global,
    metrics::{Counter, Histogram, MeterProvider},
    KeyValue,
};
use prometheus::Registry;

#[derive(Clone)]
pub struct Metrics {
    request_counter: Counter<u64>,
    prediction_duration: Histogram<u64>,
    pub registry: Registry,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap();

        let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
            .with_reader(exporter)
            .build();

        let meter = provider.meter("webcam_capture");
        global::set_meter_provider(provider);

        let request_counter = meter
            .u64_counter("requests_total")
            .with_description("Total number of requests")
            .build();

        let prediction_duration = meter
            .u64_histogram("prediction_duration_ms")
            .with_description("Duration of prediction operations in milliseconds")
            .build();

        Metrics {
            request_counter,
            prediction_duration,
            registry,
        }
    }

    pub fn record_request(&self, route: &str) {
        let attributes = vec![KeyValue::new("route", route.to_string())];
        self.request_counter.add(1, &attributes);
    }

    pub fn record_prediction_duration(&self, duration_ms: u64, route: &str) {
        let attributes = vec![KeyValue::new("route", route.to_string())];
        self.prediction_duration.record(duration_ms, &attributes);
    }
}
