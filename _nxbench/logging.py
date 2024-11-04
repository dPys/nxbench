from dataclasses import dataclass, field

__all__ = ["LoggingHandlerConfig", "LoggerConfig", "LoggingConfig"]


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
