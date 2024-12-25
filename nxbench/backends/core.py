import importlib
import logging
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Any

logger = logging.getLogger("nxbench")


def is_available(backend_name: str) -> bool:
    """
    Return True if `backend_name` is importable; False otherwise.
    E.g., 'networkx', 'nx_cugraph', 'graphblas_algorithms', etc.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec(backend_name)
    except ImportError:
        return False
    else:
        return spec is not None


def get_backend_version(backend_name: str) -> str:
    """
    Attempt to retrieve `backend_name`'s version.
    - First, check __version__ attribute.
    - If that fails, fallback to importlib.metadata.get_version().
    - If everything fails, return "unknown".
    """
    try:
        imported_backend = importlib.import_module(backend_name)
        ver = getattr(imported_backend, "__version__", None)
        if ver is None:
            ver = get_version(backend_name)
    except (ImportError, PackageNotFoundError):
        return "unknown"
    else:
        return ver


class BackendManager:
    """
    A registry for dynamically registering and configuring networkx backends.

    The registry maps:
        backend_name (str) ->
           (import_name, conversion_func, teardown_func)
    where:
        - import_name (str) is the actual module name to import
        - conversion_func(nx_graph, num_threads) -> typed backend graph
        - teardown_func() -> None (optional)
    """

    def __init__(self):
        self._registry: dict[
            str,
            tuple[
                str,  # import_name
                Callable[[Any, int], Any],  # conversion_func
                Callable[[], None] | None,  # teardown_func
            ],
        ] = {}

    def register_backend(
        self,
        name: str,
        import_name: str,
        conversion_func: Callable[[Any, int], Any],
        teardown_func: Callable[[], None] | None = None,
    ):
        """
        Register a new backend with a given name, import_name, conversion function,
        and optional teardown function.

        Example:
            register_backend(
                name="networkx",
                import_name="networkx",
                conversion_func=convert_networkx,
                teardown_func=teardown_networkx
            )
        """
        self._registry[name] = (import_name, conversion_func, teardown_func)
        logger.debug(f"Registered backend '{name}' (import_name='{import_name}').")

    def is_registered(self, name: str) -> bool:
        return name in self._registry

    def is_available(self, name: str) -> bool:
        """Check if the registered backend is actually installed (importable)."""
        if name not in self._registry:
            return False
        import_name, _, _ = self._registry[name]
        return is_available(import_name)

    def configure_backend(
        self, name: str, original_graph: Any, num_threads: int
    ) -> Any:
        """Convert the given networkx.Graph to the backend-specific representation."""
        if name not in self._registry:
            raise ValueError(f"Unsupported backend: {name}")

        import_name, converter, _ = self._registry[name]
        if not is_available(import_name):
            raise ImportError(
                f"Backend '{name}' is not available (import_name='{import_name}')."
            )

        try:
            return converter(original_graph, num_threads)
        except Exception as e:
            logger.exception(f"Error converting graph to backend '{name}' format.")
            raise

    def get_version(self, name: str) -> str:
        """
        Retrieve the version for the given backend
        by its import_name. If not installed, returns "unknown".
        """
        if name not in self._registry:
            return "unknown"
        import_name, _, _ = self._registry[name]
        if not is_available(import_name):
            return "unknown"
        return get_backend_version(import_name)

    def teardown_backend(self, name: str):
        """If a teardown function is registered, call it. Otherwise, do nothing."""
        if name not in self._registry:
            return
        import_name, _, teardown_func = self._registry[name]
        if not is_available(import_name):
            return
        if teardown_func is None:
            return
        try:
            teardown_func()
        except Exception:
            logger.exception(f"Error in teardown function for backend '{name}'")
