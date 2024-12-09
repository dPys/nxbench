import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from networkx.utils.configs import Config

__all__ = [
    "NxBenchConfig",
    "LoggingHandlerConfig",
    "LoggerConfig",
    "LoggingConfig",
    "setup_logger",
    "create_handler",
    "setup_logger_from_config",
    "update_logger",
    "on_config_change",
    "get_default_logger",
    "disable_logger",
]


@dataclass
class LoggingHandlerConfig:
    """Configuration for a single logging handler."""

    handler_type: str  # 'console' or 'file'
    level: str = "INFO"  # e.g., 'DEBUG', 'INFO', 'WARNING', etc.
    formatter: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str | None = None  # only used if handler_type is 'file'
    rotate_logs: bool = True
    backup_count: int = 7
    when: str = "midnight"


@dataclass
class LoggerConfig:
    """Configuration for a single logger."""

    name: str
    level: str = "INFO"
    handlers: list[LoggingHandlerConfig] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Configuration for all loggers."""

    loggers: list[LoggerConfig] = field(default_factory=list)


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
    """Update a specific logger based on action."""
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
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    elif action == "verbosity_level":
        verbosity_level = value  # value is expected to be an integer (0, 1, 2)
        logger_name = "nxbench"
        logger = logging.getLogger(logger_name)

        if verbosity_level == 0:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
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
        update_logger(value, name, None)
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


@dataclass
class NxBenchConfig(Config):
    """Configuration for NetworkX that controls behaviors such as how to use backends
    and logging.
    """

    active: bool = False

    verbosity_level: int = 0
    backend_name: str = "nxbench"
    backend_params: dict = field(default_factory=dict)

    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    _observers: list[Callable[[str, Any], None]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self):
        """Set environment variables based on initialized fields."""
        self.set_verbosity_level(self.verbosity_level)

    def register_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Register an observer callback to be notified on configuration changes.

        Parameters
        ----------
        callback : Callable[[str, any], None]
            A function that accepts two arguments: the name of the configuration
            parameter
            that changed and its new value.
        """
        self._observers.append(callback)

    def notify_observers(self, name: str, value: Any) -> None:
        """Notify all registered observers about a configuration change.

        Parameters
        ----------
        name : str
            The name of the configuration parameter that changed.
        value : Any
            The new value of the configuration parameter.
        """
        for callback in self._observers:
            callback(name, value)

    def set_verbosity_level(self, level: int) -> None:
        """Set the verbosity level (0-2). 2=DEBUG, 1=INFO, 0=NO logging."""
        if level not in [0, 1, 2]:
            raise ValueError("Verbosity level must be 0, 1, or 2")

        self.verbosity_level = level

        level_map = {0: None, 1: "INFO", 2: "DEBUG"}
        log_level = level_map[level]

        nxbench_logger_cfg = next(
            (
                logger
                for logger in self.logging_config.loggers
                if logger.name == "nxbench"
            ),
            None,
        )

        if level == 0:
            if nxbench_logger_cfg:
                self.logging_config.loggers.remove(nxbench_logger_cfg)
                self.notify_observers("remove_logger", "nxbench")
        elif nxbench_logger_cfg:
            nxbench_logger_cfg.level = log_level
            for handler in nxbench_logger_cfg.handlers:
                handler.level = log_level
            self.notify_observers("update_logger", "nxbench")
        else:
            new_logger = LoggerConfig(
                name="nxbench",
                level=log_level,
                handlers=[
                    LoggingHandlerConfig(
                        handler_type="console",
                        level=log_level,
                        formatter="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # noqa: E501
                    )
                ],
            )
            self.logging_config.loggers.append(new_logger)
            self.notify_observers("add_logger", "nxbench")

        self.notify_observers("verbosity_level", level)


_config = NxBenchConfig()
