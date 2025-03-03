use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
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
    pub min_probability: f32,
}

fn default_model_instances() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(5)
}

impl ModelConfig {
    pub fn get_model_path(&self) -> PathBuf {
        self.model_dir.join(&self.onnx_file)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.get_model_path().exists() {
            return Err(format!("Model file not found: {:?}", self.get_model_path()));
        }
        Ok(())
    }
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
            config::Environment::with_prefix("APP")
                .prefix_separator("_")
                .separator("__"),
        )
        .build()?;

    let config: Config = config.try_deserialize::<Config>()?;

    if let Err(e) = config.model.validate() {
        tracing::error!("Configuration validation failed: {}", e);
        return Err(config::ConfigError::Message(e));
    }

    Ok(config)
}
