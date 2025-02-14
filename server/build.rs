use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    tonic_build::compile_protos("proto/yolo_service.proto")?;

    Ok(())
}
