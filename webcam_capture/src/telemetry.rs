use opentelemetry::{
    global,
    metrics::{Counter, Gauge, Histogram, MeterProvider},
    KeyValue,
};
use prometheus::Registry;
use std::collections::HashSet;

pub struct Metrics {
    request_counter: Counter<u64>,
    prediction_duration: Histogram<u64>,
    camera_fps: Gauge<f64>,
    prediction_fps: Gauge<f64>,
    pub registry: Registry,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        // TODO: deprecated crate to be replaced with an OLTP exporter
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

        let boundaries = generate_boundaries((15, 30, 60, 500, 1000));

        let prediction_duration = meter
            .u64_histogram("prediction_duration_ms")
            .with_boundaries(boundaries)
            .with_description("Duration of prediction operations in milliseconds")
            .build();

        let camera_fps = meter
            .f64_gauge("camera_fps")
            .with_description("FPS of camera")
            .build();

        let prediction_fps = meter
            .f64_gauge("prediction_fps")
            .with_description("FPS of prediction operations")
            .build();

        Metrics {
            request_counter,
            prediction_duration,
            camera_fps,
            prediction_fps,
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

    pub fn record_camera_fps(&self, fps: f64, route: &str) {
        let attributes = vec![KeyValue::new("route", route.to_string())];
        self.camera_fps.record(fps, &attributes);
    }

    pub fn record_prediction_fps(&self, fps: f64, route: &str) {
        let attributes = vec![KeyValue::new("route", route.to_string())];
        self.prediction_fps.record(fps, &attributes);
    }
}

fn generate_boundaries(parts: (i32, i32, i32, i32, i32)) -> Vec<f64> {
    let first_step: usize = 10;
    let middle_step: usize = 2;
    let end_step: usize = 20;
    let tail_step: usize = 100;
    let first_part = (parts.0..=parts.1).step_by(first_step);
    let middle_part = (parts.1..=parts.2).step_by(middle_step);
    let end_part = (parts.2..=parts.3).step_by(end_step);
    let tail_part = (parts.3..=parts.4).step_by(tail_step);

    let mut seen = HashSet::new();
    first_part
        .chain(middle_part)
        .chain(end_part)
        .chain(tail_part)
        .filter(|&x| seen.insert(x))
        .map(|x| x as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_boundaries() {
        let parts = (2, 22, 26, 46, 146);
        let get = generate_boundaries(parts);
        let expected = vec![2.0, 12.0, 22.0, 24.0, 26.0, 46.0, 146.0];

        assert_eq!(get, expected);
    }
}
