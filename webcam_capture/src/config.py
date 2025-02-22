import os

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class Environment(Enum):
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class Config:
    environment: Environment
    yolo_service_host: str
    yolo_service_port: int
    app_host: str
    app_port: int

    @property
    def get_yolo_service_address(self) -> str:
        return f"{self.yolo_service_host}:{self.yolo_service_port}"

    @property
    def get_app_address(self) -> str:
        return f"{self.app_host}:{self.app_port}"


def load_config():
    with open(f"{Path(__file__).parents[1]}/configuration/base.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    env_str = os.getenv("APP_ENVIRONMENT", "local").lower()

    try:
        environment = Environment(env_str)
    except KeyError:
        raise ValueError(
            f"Invalid environment: {env_str}. Must be 'local' or 'production'"
        )

    config_file = f"{Path(__file__).parents[1]}/configuration/{environment.value}.yaml"
    try:
        with open(config_file, "r") as f:
            env_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: {config_file} not found. Using base config for those values.")
        env_config = {}

    config_data = base_config.copy()
    config_data.update(env_config)

    return Config(
        environment=environment,
        yolo_service_host=config_data["yolo_service_host"],
        yolo_service_port=config_data["yolo_service_port"],
        app_host=config_data["app_host"],
        app_port=config_data["app_port"],
    )


cfg = load_config()
