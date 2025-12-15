use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let descriptor_path = out_dir.join("yolo.bin");

    tonic_prost_build::configure()
        .build_server(true)
        .file_descriptor_set_path(&descriptor_path)
        .compile_protos(&["proto/yolo_service.proto"], &["proto"])?;
    Ok(())
}
