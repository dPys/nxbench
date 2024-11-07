from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from networkx.utils.configs import Config

from _nxbench.logging import LoggerConfig, LoggingConfig, LoggingHandlerConfig

__all__ = ["_config", "NxBenchConfig"]


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
                        formatter="%(asctime)s - %(name)s - %(levelname)s - % "
                        "(message)s",
                    )
                ],
            )
            self.logging_config.loggers.append(new_logger)
            self.notify_observers("add_logger", "nxbench")

        self.notify_observers("verbosity_level", level)


_config = NxBenchConfig()
