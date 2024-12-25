import gc
import importlib
import inspect
import logging
import os
import platform
import random
import sys
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

from nxbench.backends.registry import backend_manager
from nxbench.benchmarking.config import AlgorithmConfig, BenchmarkConfig, DatasetConfig
from nxbench.benchmarking.constants import ALGORITHM_SUBMODULES

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("nxbench")

_BENCHMARK_CONFIG: BenchmarkConfig | None = None


def configure_benchmarks(config: BenchmarkConfig | str):
    global _BENCHMARK_CONFIG  # noqa: PLW0603
    if _BENCHMARK_CONFIG is not None:
        raise ValueError("Benchmark configuration already set")
    if isinstance(config, BenchmarkConfig):
        _BENCHMARK_CONFIG = config
    elif isinstance(config, str):
        _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(config)
    else:
        raise TypeError("Invalid type for configuration")


def get_benchmark_config() -> BenchmarkConfig:
    global _BENCHMARK_CONFIG  # noqa: PLW0603
    if _BENCHMARK_CONFIG is not None:
        return _BENCHMARK_CONFIG

    config_file = os.getenv("NXBENCH_CONFIG_FILE")
    if config_file:
        config_path = Path(config_file)

        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.debug(f"Resolved config file path: {config_path}")

        _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(str(config_path))
    else:
        _BENCHMARK_CONFIG = load_default_config()

    return _BENCHMARK_CONFIG


def load_default_config() -> BenchmarkConfig:
    default_algorithms = [
        AlgorithmConfig(
            name="pagerank",
            func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
            params={"alpha": 0.85},
        ),
    ]
    default_datasets = [
        DatasetConfig(name="08blocks", source="networkrepository"),
        DatasetConfig(name="jazz", source="networkrepository"),
        DatasetConfig(name="karate", source="networkrepository"),
        DatasetConfig(name="enron", source="networkrepository"),
    ]

    env_data = {
        "num_threads": ["1", "4"],
        "backend": {
            "networkx": ["networkx==3.4.2"],
            "graphblas": ["graphblas_algorithms==2023.10.0"],
        },
        "pythons": ["3.10", "3.11"],
    }

    return BenchmarkConfig(
        algorithms=default_algorithms,
        datasets=default_datasets,
        env_data=env_data,
        machine_info={},
    )


def get_python_version() -> str:
    """Get formatted Python version string."""
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


class MemorySnapshot:
    """Class to store and diff memory snapshots."""

    def __init__(self, snapshot=None):
        """Initialize with optional tracemalloc snapshot."""
        self.snapshot = snapshot

    def take(self):
        """Take a new snapshot."""
        self.snapshot = tracemalloc.take_snapshot()

    def compare_to(self, other: "MemorySnapshot") -> tuple[int, int]:
        """Compare this snapshot to another and return (current, peak) memory diff in
        bytes.
        """
        if not self.snapshot or not other.snapshot:
            return 0, 0

        stats = self.snapshot.compare_to(other.snapshot, "lineno")
        current = sum(stat.size_diff for stat in stats)
        peak = sum(stat.size for stat in stats)
        return current, peak


@contextmanager
def memory_tracker():
    """Track memory usage of code block.

    Returns dict with 'current' and 'peak' memory usage in bytes.
    Memory usage is measured as the difference between before and after execution.
    """
    gc.collect()
    tracemalloc.start()

    baseline = MemorySnapshot()
    baseline.take()

    mem = {}
    try:
        yield mem
    finally:
        gc.collect()
        end = MemorySnapshot()
        end.take()
        current, peak = end.compare_to(baseline)

        mem["current"] = current
        mem["peak"] = peak

        tracemalloc.stop()


def get_available_algorithms():
    """Get algorithms from specified NetworkX submodules and custom
    algorithms.

    Returns
    -------
    Dict[str, Callable]
        Dictionary of available algorithms.
    """
    nx_algorithm_dict: dict[str, Callable] = {}

    for submodule in ALGORITHM_SUBMODULES:
        spec = importlib.util.find_spec(submodule)
        if spec is None:
            continue
        module = importlib.import_module(submodule)

        for attr_name in dir(module):
            if not attr_name.startswith("_") and not any(
                attr_name.startswith(prefix)
                for prefix in [
                    "is_",
                    "has_",
                    "get_",
                    "set_",
                    "contains_",
                    "write_",
                    "read_",
                    "to_",
                    "from_",
                    "generate_",
                    "make_",
                    "create_",
                    "build_",
                    "delete_",
                    "remove_",
                    "not_implemented",
                    "np_random_state",
                ]
            ):
                try:
                    attr = getattr(module, attr_name)
                except AttributeError:
                    continue
                if inspect.isfunction(attr):
                    if "approximation" in module.__name__:
                        nx_algorithm_dict[f"approximate_{attr_name}"] = attr
                    else:
                        nx_algorithm_dict[attr_name] = attr

    return nx_algorithm_dict


def get_machine_info():
    info = {
        "arch": platform.machine(),
        "cpu": platform.processor(),
        "num_cpu": str(psutil.cpu_count(logical=True)),
        "os": f"{platform.system()} {platform.release()}",
        "ram": str(psutil.virtual_memory().total),
    }
    # info["docker"] = os.path.exists("/.dockerenv")
    return info


def process_algorithm_params(
    params: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Process and separate algorithm parameters into positional and keyword arguments.

    1. Keys starting with "_" go into pos_args (list).
    2. Other keys become kwargs (dict).
    3. If a param is a string that looks like a float or int, parse it.
    4. If param is a dict containing {"func": "..."} then dynamically load that
    function.
    """
    pos_args = []
    kwargs = {}

    for key, value in params.items():
        # attempt to parse string values as float or int:
        if isinstance(value, str):
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # not numeric, leave it as-is

        if isinstance(value, dict) and "func" in value:
            module_path, func_name = value["func"].rsplit(".", 1)
            module = __import__(module_path, fromlist=[func_name])
            value = getattr(module, func_name)

        if key.startswith("_"):
            pos_args.append(value)
        else:
            kwargs[key] = value

    return pos_args, kwargs


def add_seeding(kwargs: dict, algo_func: Any, algorithm_name: str) -> dict:
    # 1. Retrieve optional fields from kwargs
    #    a) user_seed: integer to set global seeds
    #    b) use_local_random_state: boolean (if True, we might pass a local np.random.
    # RandomState)
    user_seed = kwargs.pop("seed", None)
    use_local_random_state = kwargs.pop("use_local_random_state", False)

    # 2. If user_seed is an int, set global seeds
    if isinstance(user_seed, int):
        random.seed(user_seed)
        np.random.seed(user_seed)
        logger.debug(f"Global random seeds set to {user_seed}.")

    # 3. Introspect the function signature to see if it takes 'seed' or 'random_state'
    func_sig = inspect.signature(algo_func)
    can_accept_seed = "seed" in func_sig.parameters
    can_accept_random_state = "random_state" in func_sig.parameters

    # 4. If can_accept_seed and we have an int user_seed, pass it as a kwarg
    if can_accept_seed and isinstance(user_seed, int):
        kwargs["seed"] = user_seed
        logger.debug(
            f"Passing `seed={user_seed}` to algorithm function {algorithm_name}."
        )

    # 5. If can_accept_random_state and user requested a local RandomState
    #    we create one and pass it in
    local_random_state = None
    if can_accept_random_state and use_local_random_state:
        # if user_seed is int, let's seed the local RNG with it, else use default.
        local_random_state = np.random.RandomState(
            user_seed if isinstance(user_seed, int) else None
        )
        kwargs["random_state"] = local_random_state
        logger.debug(
            f"Created local RandomState for algorithm {algorithm_name}, seed="
            f"{user_seed}."
        )
    return kwargs


def list_available_backends() -> dict[str, str]:
    """
    Return a dict of all registered backends that are installed,
    mapped to their version string.
    """
    installed = {}
    for backend_name in backend_manager._registry:
        if backend_manager.is_available(backend_name):
            installed[backend_name] = backend_manager.get_version(backend_name)
    logger.debug(f"Available backends: {installed}")
    return installed
