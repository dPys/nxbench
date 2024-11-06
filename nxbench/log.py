import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from _nxbench.config import LoggerConfig, LoggingConfig, LoggingHandlerConfig, _config

__all__ = [
    "setup_logger",
    "create_handler",
    "setup_logger_from_config",
    "update_logger",
    "on_config_change",
    "get_default_logger",
    "disable_logger",
]


def setup_logger(logger_cfg: LoggerConfig) -> None:
    """Set up a single logger based on LoggerConfig."""
    logger = logging.getLogger(logger_cfg.name)
    logger.setLevel(getattr(logging, logger_cfg.level.upper(), logging.INFO))

    logger.handlers = []

    for handler_cfg in logger_cfg.handlers:
        handler = create_handler(handler_cfg)
        if handler:
            logger.addHandler(handler)


def create_handler(handler_cfg: LoggingHandlerConfig) -> logging.Handler | None:
    """Create a logging handler based on the handler configuration."""
    formatter = logging.Formatter(handler_cfg.formatter)
    handler = None

    if handler_cfg.handler_type.lower() == "console":
        handler = logging.StreamHandler(sys.stdout)
    elif handler_cfg.handler_type.lower() == "file":
        if not handler_cfg.log_file:
            raise ValueError("log_file must be specified for file handlers.")
        if handler_cfg.rotate_logs:
            handler = TimedRotatingFileHandler(
                handler_cfg.log_file,
                when=handler_cfg.when,
                backupCount=handler_cfg.backup_count,
            )
        else:
            handler = logging.FileHandler(handler_cfg.log_file)
    else:
        raise ValueError(f"Unsupported handler type: {handler_cfg.handler_type}")

    if handler:
        handler.setLevel(getattr(logging, handler_cfg.level.upper(), logging.INFO))
        handler.setFormatter(formatter)

    return handler


def setup_logger_from_config(logging_config: LoggingConfig) -> None:
    """Set up all loggers based on the provided LoggingConfig."""
    for logger_cfg in logging_config.loggers:
        setup_logger(logger_cfg)


def update_logger(name: str, action: str, value: Any = None) -> None:
    """Update a specific logger based on action.

    Parameters
    ----------
    name : str
        The name of the logger to update.
    action : str
        The action to perform: 'add_logger', 'update_logger', 'remove_logger',
        'verbosity_level'.
    value : Any, optional
        Additional value needed for certain actions (e.g., verbosity level).
    """
    if action in ("add_logger", "update_logger"):
        logger_cfg = next(
            (
                logger
                for logger in _config.logging_config.loggers
                if logger.name == name
            ),
            None,
        )
        if logger_cfg:
            setup_logger(logger_cfg)
    elif action == "remove_logger":
        logger = logging.getLogger(name)
        logger.handlers = []
    elif action == "verbosity_level":
        verbosity_level = value  # value is expected to be an integer (0, 1, 2)
        logger_name = "nxbench"
        logger = logging.getLogger(logger_name)

        if verbosity_level == 0:
            logger.handlers = []
            logger.disabled = True
        else:
            logger.disabled = False
            level_map = {1: "INFO", 2: "DEBUG"}
            log_level = level_map.get(verbosity_level, "INFO")
            logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            else:
                for handler in logger.handlers:
                    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    else:
        raise ValueError(f"Unknown action: {action}")


def on_config_change(name: str, value: Any) -> None:
    """Handle configuration changes.

    Parameters
    ----------
    name : str
        The name of the configuration parameter that changed.
    value : any
        The new value of the configuration parameter.
    """
    if name == "verbosity_level":
        update_logger("nxbench", "verbosity_level", value)
    elif name in ("add_logger", "update_logger", "remove_logger"):
        update_logger(name, name, value)
    else:
        pass


def get_default_logger() -> logging.Logger:
    """Return a default logger instance for simple logging."""
    return logging.getLogger("nxbench")


def disable_logger(logger_name: str) -> None:
    """Disables the logger by removing all handlers."""
    logger = logging.getLogger(logger_name)
    logger.disabled = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def initialize_logging() -> None:
    """Initialize logging based on the current configuration."""
    setup_logger_from_config(_config.logging_config)
    _config.register_observer(on_config_change)
