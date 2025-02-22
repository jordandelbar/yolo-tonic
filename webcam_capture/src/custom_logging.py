import logging
import sys
from pathlib import Path
from loguru import logger
import json

from src.config import Environment


class InterceptHandler(logging.Handler):
    loglevel_mapping = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.loglevel_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id="app")
        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class CustomizeLogger:
    @classmethod
    def make_logger(cls, logging_config_path: Path, environment: Environment):
        config = cls.load_logging_config(logging_config_path)
        logging_config = config.get("logger")

        if environment == Environment.PRODUCTION:
            format = logging_config.get("format_production")
            serialize = True
        else:
            format = logging_config.get("format_local")
            serialize = False

        return cls.customize_logging(
            level=logging_config.get("level"), format=format, serialize=serialize
        )

    @classmethod
    def customize_logging(cls, level: str, format: str, serialize: bool):
        logger.remove()
        logger.add(
            sys.stdout,
            enqueue=False,
            backtrace=False,
            level=level.upper(),
            format=format,
            serialize=serialize,
        )
        logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)
        logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
        for _log in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
            logging.getLogger(_log).handlers = [InterceptHandler()]

        return logger.bind(request_id=None, method=None)

    @classmethod
    def load_logging_config(cls, config_path):
        with open(config_path) as config_file:
            return json.load(config_file)
