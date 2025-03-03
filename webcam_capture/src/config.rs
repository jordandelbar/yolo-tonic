use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub app: AppConfig,
    #[serde(deserialize_with = "deserialize_log_level")]
    pub log_level: LogLevel,
    pub prediction_service: PredictionServiceConfig,
}

fn deserialize_log_level<'de, D>(deserializer: D) -> Result<LogLevel, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.try_into().map_err(serde::de::Error::custom)
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_video_stream_fps")]
    pub video_stream_fps: u64,
}

fn default_video_stream_fps() -> u64 {
    60
}

impl AppConfig {
    pub fn get_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn get_stream_delay_ms(&self) -> u64 {
        fps_to_delay_ms(self.video_stream_fps)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct PredictionServiceConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_prediction_fps")]
    pub prediction_fps: u64,
}

fn default_prediction_fps() -> u64 {
    20
}

impl PredictionServiceConfig {
    pub fn get_address(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub fn get_prediction_delay_ms(&self) -> u64 {
        fps_to_delay_ms(self.prediction_fps)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub enum Environment {
    Local,
    Production,
}

impl Environment {
    pub fn as_str(&self) -> &'static str {
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

fn fps_to_delay_ms(fps: u64) -> u64 {
    (1000.0 / fps as f64).round() as u64
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

    Ok(config)
}
