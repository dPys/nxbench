from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest

from nxbench.backends.core import (
    BackendManager,
    get_backend_version,
    is_available,
    logger,
)


@pytest.fixture
def mock_logger():
    with (
        patch.object(logger, "exception") as mock_exception,
        patch.object(logger, "debug") as mock_debug,
    ):
        yield mock_exception, mock_debug


# --------------------------------------------------------------------------------------
# is_available tests
# --------------------------------------------------------------------------------------


def test_is_available_success():
    """Test that is_available returns True when the module is found."""
    with patch("importlib.util.find_spec", return_value=True):
        assert is_available("fake_module") is True


def test_is_available_not_found():
    """Test that is_available returns False when the module cannot be found."""
    with patch("importlib.util.find_spec", return_value=None):
        assert is_available("fake_module") is False


def test_is_available_import_error():
    """Test that is_available returns False if an ImportError is raised."""
    with patch("importlib.util.find_spec", side_effect=ImportError):
        assert is_available("fake_module") is False


# --------------------------------------------------------------------------------------
# get_backend_version tests
# --------------------------------------------------------------------------------------


def test_get_backend_version_with_dunder_version():
    """Test that get_backend_version retrieves __version__ successfully."""
    fake_module = MagicMock()
    fake_module.__version__ = "1.2.3"
    with patch("importlib.import_module", return_value=fake_module):
        assert get_backend_version("fake_module") == "1.2.3"


def test_get_backend_version_unknown_import_error():
    """If the import fails entirely, get_backend_version returns "unknown"."""
    with patch("importlib.import_module", side_effect=ImportError):
        assert get_backend_version("fake_module") == "unknown"


def test_get_backend_version_unknown_package_not_found():
    """If the package is not found by importlib.metadata, return "unknown"."""
    fake_module = MagicMock()
    with (
        patch("importlib.import_module", return_value=fake_module),
        patch("nxbench.backends.core.get_version", side_effect=PackageNotFoundError),
    ):
        assert get_backend_version("fake_module") == "unknown"


# --------------------------------------------------------------------------------------
# BackendManager tests
# --------------------------------------------------------------------------------------


@pytest.fixture
def backend_manager():
    return BackendManager()


@pytest.fixture
def fake_conversion_func():
    def convert(graph, num_threads):
        return {"converted_graph": graph, "threads": num_threads}

    return convert


@pytest.fixture
def fake_teardown_func():
    def teardown():
        pass

    return teardown


def test_register_backend(
    backend_manager, fake_conversion_func, fake_teardown_func, mock_logger
):
    """
    Ensure that register_backend updates the registry properly
    and logs a debug message.
    """
    logger_exception_mock, logger_debug_mock = mock_logger
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import_name",
        conversion_func=fake_conversion_func,
        teardown_func=fake_teardown_func,
    )

    assert "test_backend" in backend_manager._registry
    import_name, conversion, teardown = backend_manager._registry["test_backend"]
    assert import_name == "test_import_name"
    assert conversion is fake_conversion_func
    assert teardown is fake_teardown_func

    logger_debug_mock.assert_called_with(
        "Registered backend 'test_backend' (import_name='test_import_name')."
    )


def test_is_registered(backend_manager, fake_conversion_func):
    """Test that is_registered returns True when a backend is registered."""
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import_name",
        conversion_func=fake_conversion_func,
    )
    assert backend_manager.is_registered("test_backend")
    assert not backend_manager.is_registered("unknown_backend")


def test_backend_manager_is_available_installed(backend_manager, fake_conversion_func):
    """If the module is found, is_available should return True."""
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=True):
        assert backend_manager.is_available("test_backend")


def test_backend_manager_is_available_not_installed(
    backend_manager, fake_conversion_func
):
    """If the module is not found, is_available should return False."""
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=False):
        assert not backend_manager.is_available("test_backend")


def test_configure_backend_success(backend_manager, fake_conversion_func):
    """
    configure_backend should call the registered conversion function
    and return its result.
    """
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=True):
        result = backend_manager.configure_backend(
            "test_backend", {"my_graph": True}, 4
        )
        assert result == {"converted_graph": {"my_graph": True}, "threads": 4}


def test_configure_backend_not_registered(backend_manager):
    """If the backend isn't registered, configure_backend should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend:"):
        backend_manager.configure_backend("unknown_backend", {}, 1)


def test_configure_backend_not_available(backend_manager, fake_conversion_func):
    """If the backend import_name isn't available, configure_backend should raise
    ImportError.
    """
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=False):
        with pytest.raises(
            ImportError, match="Backend 'test_backend' is not available"
        ):
            backend_manager.configure_backend("test_backend", {}, 1)


def test_configure_backend_conversion_error(backend_manager, mock_logger):
    """If the conversion_func raises an exception, it should be logged and re-raised."""
    logger_exception_mock, logger_debug_mock = mock_logger

    def bad_conversion_func(graph, num_threads):
        raise ValueError("Conversion error")

    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=bad_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=True):
        with pytest.raises(ValueError, match="Conversion error"):
            backend_manager.configure_backend("test_backend", {}, 1)

    logger_exception_mock.assert_called_once_with(
        "Error converting graph to backend 'test_backend' format."
    )


def test_get_version_installed(backend_manager, fake_conversion_func):
    """If the backend is registered and installed, get_version should return the
    backend version.
    """
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with (
        patch("nxbench.backends.core.is_available", return_value=True),
        patch("nxbench.backends.core.get_backend_version", return_value="9.8.7"),
    ):
        assert backend_manager.get_version("test_backend") == "9.8.7"


def test_get_version_not_installed(backend_manager, fake_conversion_func):
    """If the backend is registered but not installed, get_version should return
    'unknown'.
    """
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=False):
        assert backend_manager.get_version("test_backend") == "unknown"


def test_get_version_not_registered(backend_manager):
    """If the backend is not registered, get_version should return 'unknown'."""
    assert backend_manager.get_version("test_backend") == "unknown"


def test_teardown_backend_no_teardown(backend_manager, fake_conversion_func):
    """If no teardown function is registered, teardown_backend should do nothing."""
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
        teardown_func=None,
    )
    with patch("nxbench.backends.core.is_available", return_value=True):
        backend_manager.teardown_backend("test_backend")


def test_teardown_backend_not_registered(backend_manager):
    """If the backend is not registered, teardown_backend should do nothing."""
    backend_manager.teardown_backend("unknown_backend")


def test_teardown_backend_not_installed(
    backend_manager, fake_conversion_func, fake_teardown_func
):
    """If the backend is not installed, teardown_backend should do nothing."""
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
        teardown_func=fake_teardown_func,
    )
    with patch("nxbench.backends.core.is_available", return_value=False):
        backend_manager.teardown_backend("test_backend")


def test_teardown_backend_success(backend_manager, fake_conversion_func):
    """If a teardown function is registered and the backend is installed, it should be
    called.
    """
    teardown_mock = MagicMock()
    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
        teardown_func=teardown_mock,
    )

    with patch("nxbench.backends.core.is_available", return_value=True):
        backend_manager.teardown_backend("test_backend")
        teardown_mock.assert_called_once()


def test_teardown_backend_error(backend_manager, fake_conversion_func, mock_logger):
    """If the teardown function raises an error, it should be logged but not
    re-raised.
    """
    logger_exception_mock, logger_debug_mock = mock_logger
    teardown_mock = MagicMock(side_effect=ValueError("Teardown error"))

    backend_manager.register_backend(
        name="test_backend",
        import_name="test_import",
        conversion_func=fake_conversion_func,
        teardown_func=teardown_mock,
    )

    with patch("nxbench.backends.core.is_available", return_value=True):
        # teardown_backend should swallow the exception but log it
        backend_manager.teardown_backend("test_backend")

    logger_exception_mock.assert_called_once_with(
        "Error in teardown function for backend 'test_backend'"
    )
