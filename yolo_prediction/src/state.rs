use crate::config::{LabelsConfig, Validatable};
use std::{
    fs::File,
    io::{self, BufRead},
    path::PathBuf,
};
use yolo_proto::ColorLabel;

pub trait State: Send + Sync + 'static {
    fn new(labels_file: &LabelsConfig) -> Result<Self, String>
    where
        Self: Sized;
    fn get_labels(&self) -> &Vec<ColorLabel>;
}

#[derive(Debug)]
pub struct ServiceState {
    class_labels: Vec<ColorLabel>,
}

impl State for ServiceState {
    fn new(labels_cfg: &LabelsConfig) -> Result<ServiceState, String> {
        match load_yolov8_labels(&labels_cfg.get_path()) {
            Ok(labels) => Ok(ServiceState {
                class_labels: labels,
            }),
            Err(e) => Err(format!("Failed to load labels: {}", e)),
        }
    }

    fn get_labels(&self) -> &Vec<ColorLabel> {
        &self.class_labels
    }
}

pub fn load_yolov8_labels(filepath: &PathBuf) -> io::Result<Vec<ColorLabel>> {
    let file = File::open(filepath)?;
    let reader = io::BufReader::new(file);
    let mut color_labels = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let parts: Vec<&str> = line.split(",").collect();

        if parts.len() == 4 {
            let label = parts[0].trim().to_string();
            let red: u32 = parts[1]
                .trim()
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid red value"))?;
            let green: u32 = parts[2]
                .trim()
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid green value"))?;
            let blue: u32 = parts[3]
                .trim()
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid blue value"))?;

            color_labels.push(ColorLabel {
                label,
                red,
                green,
                blue,
            });
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid line format: {}", line),
            ));
        }
    }

    Ok(color_labels)
}
