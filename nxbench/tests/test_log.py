import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from nxbench.log import (
    LoggerConfig,
    LoggingConfig,
    LoggingHandlerConfig,
    NxBenchConfig,
    _config,
    create_handler,
    disable_logger,
    get_default_logger,
    initialize_logging,
    on_config_change,
    setup_logger,
    setup_logger_from_config,
    update_logger,
)


@pytest.fixture(autouse=True)
def _reset_config():
    _config.logging_config = LoggingConfig()
    _config.verbosity_level = 0
    _config._observers = []
    _config.register_observer(on_config_change)
    yield
    _config.logging_config = LoggingConfig()
    _config.verbosity_level = 0
    _config._observers = []


def test_logging_handler_config_defaults():
    handler_cfg = LoggingHandlerConfig(handler_type="console")
    assert handler_cfg.handler_type == "console"
    assert handler_cfg.level == "INFO"
    assert (
        handler_cfg.formatter == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    assert handler_cfg.log_file is None
    assert handler_cfg.rotate_logs is True
    assert handler_cfg.backup_count == 7
    assert handler_cfg.when == "midnight"


def test_logging_handler_config_custom():
    handler_cfg = LoggingHandlerConfig(
        handler_type="file",
        level="DEBUG",
        formatter="%(levelname)s:%(message)s",
        log_file="app.log",
        rotate_logs=False,
        backup_count=5,
        when="H",
    )
    assert handler_cfg.handler_type == "file"
    assert handler_cfg.level == "DEBUG"
    assert handler_cfg.formatter == "%(levelname)s:%(message)s"
    assert handler_cfg.log_file == "app.log"
    assert handler_cfg.rotate_logs is False
    assert handler_cfg.backup_count == 5
    assert handler_cfg.when == "H"


def test_logger_config_defaults():
    logger_cfg = LoggerConfig(name="test_logger")
    assert logger_cfg.name == "test_logger"
    assert logger_cfg.level == "INFO"
    assert logger_cfg.handlers == []


def test_logger_config_custom():
    handler_cfg = LoggingHandlerConfig(handler_type="console", level="DEBUG")
    logger_cfg = LoggerConfig(
        name="custom_logger", level="DEBUG", handlers=[handler_cfg]
    )
    assert logger_cfg.name == "custom_logger"
    assert logger_cfg.level == "DEBUG"
    assert logger_cfg.handlers == [handler_cfg]


def test_logging_config_defaults():
    logging_cfg = LoggingConfig()
    assert logging_cfg.loggers == []


def test_logging_config_custom():
    handler_cfg = LoggingHandlerConfig(handler_type="console", level="DEBUG")
    logger_cfg = LoggerConfig(
        name="custom_logger", level="DEBUG", handlers=[handler_cfg]
    )
    logging_cfg = LoggingConfig(loggers=[logger_cfg])
    assert logging_cfg.loggers == [logger_cfg]


@patch("nxbench.log.logging.StreamHandler")
def test_create_console_handler(mock_stream_handler):
    handler_cfg = LoggingHandlerConfig(handler_type="console", level="DEBUG")
    handler = create_handler(handler_cfg)
    mock_stream_handler.assert_called_with(sys.stdout)
    assert handler == mock_stream_handler.return_value
    handler.setLevel.assert_called_with(logging.DEBUG)
    handler.setFormatter.assert_called()


@patch("nxbench.log.TimedRotatingFileHandler")
def test_create_rotating_file_handler(mock_timed_rotating_handler):
    handler_cfg = LoggingHandlerConfig(
        handler_type="file",
        level="INFO",
        log_file="app.log",
        rotate_logs=True,
        backup_count=5,
        when="H",
    )
    handler = create_handler(handler_cfg)
    mock_timed_rotating_handler.assert_called_with("app.log", when="H", backupCount=5)
    assert handler == mock_timed_rotating_handler.return_value
    handler.setLevel.assert_called_with(logging.INFO)
    handler.setFormatter.assert_called()


@patch("nxbench.log.logging.FileHandler")
def test_create_file_handler_without_rotation(mock_file_handler):
    handler_cfg = LoggingHandlerConfig(
        handler_type="file",
        level="WARNING",
        log_file="warnings.log",
        rotate_logs=False,
    )
    handler = create_handler(handler_cfg)
    mock_file_handler.assert_called_with("warnings.log")
    assert handler == mock_file_handler.return_value
    handler.setLevel.assert_called_with(logging.WARNING)
    handler.setFormatter.assert_called()


def test_create_handler_invalid_type():
    handler_cfg = LoggingHandlerConfig(handler_type="invalid")
    with pytest.raises(ValueError, match="Unsupported handler type: invalid"):
        create_handler(handler_cfg)


def test_create_file_handler_missing_log_file():
    handler_cfg = LoggingHandlerConfig(handler_type="file")
    with pytest.raises(
        ValueError, match="log_file must be specified for file handlers."
    ):
        create_handler(handler_cfg)


@patch("nxbench.log.create_handler")
def test_setup_logger(mock_create_handler):
    handler_cfg1 = LoggingHandlerConfig(handler_type="console", level="INFO")
    handler_cfg2 = LoggingHandlerConfig(
        handler_type="file", level="DEBUG", log_file="debug.log"
    )
    logger_cfg = LoggerConfig(
        name="test_logger", level="DEBUG", handlers=[handler_cfg1, handler_cfg2]
    )

    mock_handler1 = MagicMock()
    mock_handler2 = MagicMock()
    mock_create_handler.side_effect = [mock_handler1, mock_handler2]

    setup_logger(logger_cfg)

    logger = logging.getLogger("test_logger")
    assert logger.level == logging.DEBUG
    assert logger.handlers == [mock_handler1, mock_handler2]
    mock_create_handler.assert_any_call(handler_cfg1)
    mock_create_handler.assert_any_call(handler_cfg2)


@patch("nxbench.log.create_handler")
@patch("nxbench.log.logging.getLogger")
def test_setup_logger_level_case_insensitive(mock_get_logger, mock_create_handler):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_create_handler.return_value = mock_handler

    handler_cfg = LoggingHandlerConfig(handler_type="console", level="DEBUG")
    logger_cfg = LoggerConfig(name="case_logger", level="debug", handlers=[handler_cfg])

    setup_logger(logger_cfg)

    mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
    mock_create_handler.assert_called_once_with(handler_cfg)


@patch("nxbench.log.setup_logger")
def test_setup_logger_from_config(mock_setup_logger):
    handler_cfg = LoggingHandlerConfig(handler_type="console")
    logger_cfg1 = LoggerConfig(name="logger1", handlers=[handler_cfg])
    logger_cfg2 = LoggerConfig(name="logger2", handlers=[])
    logging_cfg = LoggingConfig(loggers=[logger_cfg1, logger_cfg2])

    setup_logger_from_config(logging_cfg)

    mock_setup_logger.assert_any_call(logger_cfg1)
    mock_setup_logger.assert_any_call(logger_cfg2)
    assert mock_setup_logger.call_count == 2


@patch("nxbench.log.setup_logger")
def test_update_logger_add_logger(mock_setup_logger):
    logger_cfg = LoggerConfig(name="new_logger", level="INFO")
    _config.logging_config.loggers.append(logger_cfg)

    update_logger("new_logger", "add_logger")

    mock_setup_logger.assert_called_once_with(logger_cfg)


@patch("nxbench.log.setup_logger")
def test_update_logger_update_logger(mock_setup_logger):
    logger_cfg = LoggerConfig(name="existing_logger", level="DEBUG")
    _config.logging_config.loggers.append(logger_cfg)

    update_logger("existing_logger", "update_logger")

    mock_setup_logger.assert_called_once_with(logger_cfg)


@patch("nxbench.log.logging.getLogger")
def test_update_logger_remove_logger(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    handler = MagicMock()
    mock_logger.handlers = [handler]

    def remove_handler_side_effect(h):
        mock_logger.handlers.remove(h)

    mock_logger.removeHandler.side_effect = remove_handler_side_effect

    update_logger("remove_logger", "remove_logger")

    mock_logger.removeHandler.assert_called_once_with(handler)
    assert mock_logger.handlers == []


@patch("nxbench.log.logging.getLogger")
def test_update_logger_verbosity_level_zero(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    handler = MagicMock()
    mock_logger.handlers = [handler]

    def remove_handler_side_effect(h):
        mock_logger.handlers.remove(h)

    mock_logger.removeHandler.side_effect = remove_handler_side_effect

    update_logger("nxbench", "verbosity_level", 0)

    mock_logger.removeHandler.assert_called_once_with(handler)
    assert mock_logger.disabled is True
    assert not mock_logger.handlers


@patch("nxbench.log.logging.getLogger")
def test_update_logger_verbosity_level_one(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    handler = MagicMock()
    mock_logger.handlers = [handler]

    update_logger("nxbench", "verbosity_level", 1)

    mock_logger.setLevel.assert_called_with(logging.INFO)
    handler.setLevel.assert_called_with(logging.INFO)
    assert mock_logger.disabled is False


@patch("nxbench.log.logging.getLogger")
def test_update_logger_verbosity_level_two(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    handler = MagicMock()
    mock_logger.handlers = [handler]

    update_logger("nxbench", "verbosity_level", 2)

    mock_logger.setLevel.assert_called_with(logging.DEBUG)
    handler.setLevel.assert_called_with(logging.DEBUG)
    assert mock_logger.disabled is False


def test_update_logger_unknown_action():
    with pytest.raises(ValueError, match="Unknown action: invalid_action"):
        update_logger("some_logger", "invalid_action")


@patch("nxbench.log.update_logger")
def test_on_config_change_verbosity_level(mock_update_logger):
    on_config_change("verbosity_level", 2)
    mock_update_logger.assert_called_once_with("nxbench", "verbosity_level", 2)


@patch("nxbench.log.update_logger")
def test_on_config_change_add_logger(mock_update_logger):
    on_config_change("add_logger", "new_logger")
    mock_update_logger.assert_called_once_with("new_logger", "add_logger", None)


@patch("nxbench.log.update_logger")
def test_on_config_change_update_logger(mock_update_logger):
    on_config_change("update_logger", "existing_logger")
    mock_update_logger.assert_called_once_with("existing_logger", "update_logger", None)


@patch("nxbench.log.update_logger")
def test_on_config_change_remove_logger(mock_update_logger):
    on_config_change("remove_logger", "old_logger")
    mock_update_logger.assert_called_once_with("old_logger", "remove_logger", None)


def test_on_config_change_unknown_name():
    try:
        on_config_change("unknown_config", "value")
    except Exception as e:
        pytest.fail(f"on_config_change raised an exception unexpectedly: {e}")


@patch("nxbench.log.logging.getLogger")
def test_get_default_logger_behavior(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    logger = get_default_logger()
    mock_get_logger.assert_called_once_with("nxbench")
    assert logger == mock_logger


@patch("nxbench.log.logging.getLogger")
def test_disable_logger(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    handler1 = MagicMock()
    handler2 = MagicMock()
    mock_logger.handlers = [handler1, handler2]

    def remove_handler_side_effect(h):
        mock_logger.handlers.remove(h)

    mock_logger.removeHandler.side_effect = remove_handler_side_effect

    disable_logger("test_disable")

    assert mock_logger.disabled is True
    assert mock_logger.handlers == []
    mock_logger.removeHandler.assert_any_call(handler1)
    mock_logger.removeHandler.assert_any_call(handler2)
    assert mock_logger.removeHandler.call_count == 2


def test_nxbench_config_defaults():
    config = NxBenchConfig()
    assert config.active is False
    assert config.verbosity_level == 0
    assert config.backend_name == "nxbench"
    assert config.backend_params == {}
    assert config.logging_config == LoggingConfig()
    assert config._observers == []


@patch.object(NxBenchConfig, "set_verbosity_level")
def test_nxbench_config_post_init(mock_set_verbosity):
    config = NxBenchConfig(verbosity_level=2)
    mock_set_verbosity.assert_called_once_with(2)
    assert config.verbosity_level == 2


def test_nxbench_config_register_observer():
    config = NxBenchConfig()
    callback = MagicMock()
    config.register_observer(callback)
    assert config._observers == [callback]


def test_nxbench_config_notify_observers():
    config = NxBenchConfig()
    callback1 = MagicMock()
    callback2 = MagicMock()
    config.register_observer(callback1)
    config.register_observer(callback2)
    config.notify_observers("test_param", "test_value")
    callback1.assert_called_once_with("test_param", "test_value")
    callback2.assert_called_once_with("test_param", "test_value")


def test_nxbench_config_set_verbosity_level_invalid():
    config = NxBenchConfig()
    with pytest.raises(ValueError, match="Verbosity level must be 0, 1, or 2"):
        config.set_verbosity_level(3)


@patch("nxbench.log.setup_logger")
def test_nxbench_config_set_verbosity_level_zero(mock_setup_logger):
    config = NxBenchConfig()
    config.logging_config.loggers.append(LoggerConfig(name="nxbench"))
    config.set_verbosity_level(0)
    assert config.verbosity_level == 0
    assert config.logging_config.loggers == []
    mock_setup_logger.assert_not_called()


@patch("nxbench.log.setup_logger")
@patch.object(NxBenchConfig, "notify_observers")
def test_nxbench_config_set_verbosity_level_one(
    mock_notify_observers, mock_setup_logger
):
    config = NxBenchConfig()
    handler_cfg = LoggingHandlerConfig(handler_type="console", level="INFO")
    logger_cfg = LoggerConfig(name="nxbench", level="INFO", handlers=[handler_cfg])
    config.logging_config.loggers.append(logger_cfg)

    config.set_verbosity_level(1)
    assert config.verbosity_level == 1
    assert len(config.logging_config.loggers) == 1
    assert config.logging_config.loggers[0].level == "INFO"
    mock_notify_observers.assert_any_call("verbosity_level", 1)


@patch("nxbench.log.setup_logger")
@patch.object(NxBenchConfig, "notify_observers")
def test_nxbench_config_set_verbosity_level_two(
    mock_notify_observers, mock_setup_logger
):
    config = NxBenchConfig()
    logger_cfg = LoggerConfig(name="nxbench", level="DEBUG", handlers=[])
    config.logging_config.loggers.append(logger_cfg)

    config.set_verbosity_level(2)
    assert config.verbosity_level == 2
    assert len(config.logging_config.loggers) == 1
    assert config.logging_config.loggers[0].level == "DEBUG"
    mock_notify_observers.assert_any_call("verbosity_level", 2)


@patch("nxbench.log.setup_logger")
@patch.object(NxBenchConfig, "notify_observers")
def test_nxbench_config_set_verbosity_level_add_logger(
    mock_notify_observers, mock_setup_logger
):
    config = NxBenchConfig()

    config.set_verbosity_level(1)
    assert config.verbosity_level == 1
    assert len(config.logging_config.loggers) == 1
    added_logger = config.logging_config.loggers[0]
    assert added_logger.name == "nxbench"
    assert added_logger.level == "INFO"
    assert len(added_logger.handlers) == 1
    handler = added_logger.handlers[0]
    assert handler.handler_type == "console"
    assert handler.level == "INFO"
    mock_notify_observers.assert_any_call("verbosity_level", 1)


@patch("nxbench.log.setup_logger_from_config")
@patch.object(NxBenchConfig, "register_observer")
def test_initialize_logging(mock_register_observer, mock_setup_logger_from_config):
    with patch("nxbench.log._config", NxBenchConfig()):
        initialize_logging()
        mock_setup_logger_from_config.assert_called_once()
        mock_register_observer.assert_called_once_with(on_config_change)


@patch("nxbench.log.logging.getLogger")
def test_get_default_logger(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    logger = get_default_logger()
    mock_get_logger.assert_called_once_with("nxbench")
    assert logger == mock_logger


@patch("nxbench.log.logging.getLogger")
def test_disable_logger_no_handlers(mock_get_logger):
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger

    disable_logger("empty_logger")
    assert mock_logger.disabled is True
    assert mock_logger.handlers == []
    mock_logger.removeHandler.assert_not_called()


@patch("nxbench.log.create_handler")
def test_setup_logger_with_no_handlers(mock_create_handler):
    logger_cfg = LoggerConfig(name="no_handler_logger", level="INFO", handlers=[])
    setup_logger(logger_cfg)
    logger = logging.getLogger("no_handler_logger")
    assert logger.level == logging.INFO
    assert logger.handlers == []
    mock_create_handler.assert_not_called()


@patch("nxbench.log.create_handler")
def test_setup_logger_with_invalid_handler(mock_create_handler):
    handler_cfg = LoggingHandlerConfig(handler_type="invalid")
    logger_cfg = LoggerConfig(name="invalid_handler_logger", handlers=[handler_cfg])
    mock_create_handler.side_effect = ValueError("Unsupported handler type: invalid")

    with pytest.raises(ValueError, match="Unsupported handler type: invalid"):
        setup_logger(logger_cfg)

    mock_create_handler.assert_called_once_with(handler_cfg)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_loggers():
    yield
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
