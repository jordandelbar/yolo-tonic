use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub labels: LabelsConfig,
    #[serde(deserialize_with = "deserialize_log_level")]
    pub log_level: LogLevel,
}

fn deserialize_log_level<'de, D>(deserializer: D) -> Result<LogLevel, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.try_into().map_err(serde::de::Error::custom)
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl ServerConfig {
    pub fn get_address(self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub onnx_file: String,
    #[serde(default = "default_model_instances")]
    pub num_instances: usize,
    pub model_dir: PathBuf,
    #[serde(default = "default_min_probability")]
    pub min_probability: f32,
}

fn default_model_instances() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(5)
}

fn default_min_probability() -> f32 {
    0.50
}

#[derive(Debug, Deserialize, Clone)]
pub struct LabelsConfig {
    pub labels_file: String,
    pub labels_dir: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub enum Environment {
    Local,
    Production,
}

impl Environment {
    fn as_str(&self) -> &'static str {
        match self {
            Environment::Local => "local",
            Environment::Production => "production",
        }
    }
}

impl TryFrom<String> for Environment {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "production" => Ok(Self::Production),
            other => Err(format!(
                "{} is not a supported environment. Use either `local` or `production`.",
                other
            )),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub enum LogLevel {
    Debug,
    Info,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
        }
    }
}

impl TryFrom<String> for LogLevel {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            other => Err(format!(
                "{} is not a supported minimum log level. Use either `debug` or `info`.",
                other
            )),
        }
    }
}

pub trait Validatable {
    fn get_path(&self) -> PathBuf;
    fn validate(&self) -> Result<(), String> {
        if !self.get_path().exists() {
            Err(format!("File not found: {:?}", self.get_path()))
        } else {
            Ok(())
        }
    }
}

impl Validatable for ModelConfig {
    fn get_path(&self) -> PathBuf {
        self.model_dir.join(&self.onnx_file)
    }
}

impl Validatable for LabelsConfig {
    fn get_path(&self) -> PathBuf {
        self.labels_dir.join(&self.labels_file)
    }
}

pub fn get_configuration() -> Result<Config, config::ConfigError> {
    let base_path = std::env::current_dir().expect("Failed to determine the current directory");
    let configuration_directory = base_path.join("configuration");

    let environment: Environment = std::env::var("APP_ENVIRONMENT")
        .unwrap_or_else(|_| "local".into())
        .try_into()
        .expect("Failed to parse APP_ENVIRONMENT");

    let config = config::Config::builder()
        .add_source(config::File::from(
            configuration_directory.join("base.yaml"),
        ))
        .add_source(config::File::from(
            configuration_directory.join(format!("{}.yaml", environment.as_str())),
        ))
        .add_source(
            config::Environment::with_prefix("YP")
                .prefix_separator("_")
                .separator("__"),
        )
        .build()?;

    let config: Config = config.try_deserialize::<Config>()?;

    if let Err(e) = config.model.validate() {
        tracing::error!("Configuration validation failed: {}", e);
        return Err(config::ConfigError::Message(e));
    }

    if let Err(e) = config.labels.validate() {
        tracing::error!("Configuration validation failed: {}", e);
        return Err(config::ConfigError::Message(e));
    }

    Ok(config)
}
