#[derive(Debug)]
pub struct BoundingBoxWithLabels {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_label: String,
    pub red: u32,
    pub green: u32,
    pub blue: u32,
    pub confidence: f32,
}
