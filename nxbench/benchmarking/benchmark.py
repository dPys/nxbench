import asyncio
import json
import logging
import os
import time
import uuid
from itertools import product
from pathlib import Path
from typing import Any

import nest_asyncio
import networkx as nx
from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner
from prefect_dask.task_runners import DaskTaskRunner

from nxbench.backends.registry import backend_manager
from nxbench.benchmarking.config import AlgorithmConfig, DatasetConfig
from nxbench.benchmarking.utils import (
    add_seeding,
    get_benchmark_config,
    get_machine_info,
    get_python_version,
    list_available_backends,
    memory_tracker,
    process_algorithm_params,
)
from nxbench.data.loader import BenchmarkDataManager
from nxbench.validation.registry import BenchmarkValidator

logger = logging.getLogger("nxbench")

nest_asyncio.apply()

os.environ.setdefault(
    "PREFECT_API_DATABASE_CONNECTION_URL",
    "postgresql+asyncpg://prefect_user:pass@localhost:5432/prefect_db",
)
os.environ.setdefault("PREFECT_ORION_DATABASE_CONNECTION_POOL_SIZE", "10")
os.environ.setdefault("PREFECT_ORION_DATABASE_CONNECTION_MAX_OVERFLOW", "20")
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")
os.environ.setdefault("PREFECT_ORION_API_ENABLE_TASK_RUN_DATA_PERSISTENCE", "false")
os.environ.setdefault("PREFECT_CLIENT_REQUEST_TIMEOUT", "60")
os.environ.setdefault("PREFECT_HTTPX_SETTINGS", '{"limits": {"max_connections": 50}')
os.environ.setdefault("MAX_WORKERS", "4")

run_uuid = uuid.uuid4().hex


def load_config() -> dict[str, Any]:
    """Load benchmark configuration dynamically."""
    config = get_benchmark_config()
    return {
        "algorithms": config.algorithms,
        "datasets": config.datasets,
        "env_data": config.env_data,
    }


def setup_cache(
    datasets: list[DatasetConfig],
) -> dict[str, tuple[nx.Graph, dict[str, Any]]]:
    """Load and cache datasets to avoid redundant loading."""
    data_manager = BenchmarkDataManager()
    graphs = {}
    for dataset_config in datasets:
        dataset_name = dataset_config.name
        try:
            graph, metadata = data_manager.load_network_sync(dataset_config)
            graphs[dataset_name] = (graph, metadata)
            logger.debug(
                f"Cached dataset '{dataset_name}' with {graph.number_of_nodes()} "
                f"nodes and {graph.number_of_edges()} edges."
            )
        except Exception:
            logger.exception(f"Failed to load dataset '{dataset_name}'")
    return graphs


@task(name="configure_backend", cache_key_fn=None, persist_result=False)
def configure_backend(original_graph: nx.Graph, backend: str, num_thread: int) -> Any:
    """Convert an Nx graph for the specified backend."""
    return backend_manager.configure_backend(backend, original_graph, num_thread)


@task(name="run_algorithm", cache_key_fn=None, persist_result=False)
def run_algorithm(
    graph: Any, algo_config: AlgorithmConfig, num_thread: int, backend: str
) -> tuple[Any, float, int, str | None]:
    """Run the algorithm on the configured backend"""
    logger = get_run_logger()

    try:
        algo_func = algo_config.get_callable(backend)
    except ImportError as e:
        logger.exception(
            f"Could not get a callable for {algo_config.name} from {backend}."
        )
        return None, 0.0, 0, str(e)

    pos_args, kwargs = process_algorithm_params(algo_config.params)

    kwargs = add_seeding(kwargs, algo_func, algo_config.name)

    error = None
    try:
        original_env = {}
        vars_to_set = [
            "NUM_THREAD",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]
        for var_name in vars_to_set:
            original_env[var_name] = os.environ.get(var_name)
            os.environ[var_name] = str(num_thread)

        # start memory tracking
        with memory_tracker() as mem:
            start_time = time.perf_counter()
            try:
                result = algo_func(graph, *pos_args, **kwargs)
            except NotImplementedError as nie:
                logger.info(
                    f"Skipping {algo_config.name} for backend '{backend}' "
                    "because it's not implemented (NotImplementedError)."
                )
                return None, 0.0, 0, str(nie)
            except MemoryError as me:
                # gracefully handle OOM:
                logger.exception("Algorithm ran out of memory.")
                result = None
                error = f"MemoryError: {me}"
            except Exception as e:
                logger.exception("Algorithm run failed unexpectedly.")
                result = None
                error = str(e)
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

            peak_memory = mem.get("peak", 0)

    finally:
        logger.debug(f"Algorithm '{algo_config.name}' executed successfully.")
        # restore environment variables
        for var_name in vars_to_set:
            if original_env[var_name] is None:
                del os.environ[var_name]
            else:
                os.environ[var_name] = original_env[var_name]

    return result, execution_time, peak_memory, error


@task(name="validate_results", cache_key_fn=None, persist_result=False)
def validate_results(
    result: Any, algo_config: AlgorithmConfig, graph: Any
) -> tuple[str, str]:
    logger = get_run_logger()
    validator = BenchmarkValidator()
    try:
        validator.validate_result(result, algo_config.name, graph)
        logger.debug(f"Validation passed for algorithm '{algo_config.name}'.")
    except Exception as e:
        logger.warning(f"Validation warning for '{algo_config.name}'")
        return "warning", str(e)
    return "passed", ""


@task(name="collect_metrics", cache_key_fn=None, persist_result=False)
def collect_metrics(
    execution_time: float,
    execution_time_with_preloading: float,
    peak_memory: int,
    graph: Any,
    algo_config: AlgorithmConfig,
    backend: str,
    dataset_name: str,
    num_thread: int,
    validation_status: str,
    validation_message: str,
    error: str | None = None,
) -> dict[str, Any]:
    logger = get_run_logger()

    if not isinstance(graph, nx.Graph) and hasattr(graph, "to_networkx"):
        graph = graph.to_networkx()

    if error:
        metrics = {
            "execution_time": float("nan"),
            "execution_time_with_preloading": float("nan"),
            "memory_used": float("nan"),
            "num_nodes": (graph.number_of_nodes()),
            "num_edges": (graph.number_of_edges()),
            "algorithm": algo_config.name,
            "backend": backend,
            "dataset": dataset_name,
            "num_thread": num_thread,
            "error": error,
            "validation": validation_status,
            "validation_message": validation_message,
        }
    else:
        metrics = {
            "execution_time": execution_time,
            "execution_time_with_preloading": execution_time_with_preloading,
            "memory_used": peak_memory / (1024 * 1024),  # convert to MB
            "num_nodes": (graph.number_of_nodes()),
            "num_edges": (graph.number_of_edges()),
            "algorithm": algo_config.name,
            "backend": backend,
            "dataset": dataset_name,
            "num_thread": num_thread,
            "validation": validation_status,
            "validation_message": validation_message,
        }

    metrics.update(get_machine_info())
    return metrics


@task(name="teardown_specific", cache_key_fn=None, persist_result=False)
def teardown_specific(backend: str):
    """If the backend provides a teardown function, call it."""
    backend_manager.teardown_backend(backend)


async def run_single_benchmark(
    backend: str,
    num_thread: int,
    algo_config: AlgorithmConfig,
    dataset_config: DatasetConfig,
    original_graph: nx.Graph,
) -> dict[str, Any] | None:
    logger = get_run_logger()
    logger.info(
        f"Running benchmark for dataset '{dataset_config.name}' with backend "
        f"'{backend}' and {num_thread} threads."
    )

    try:
        preload_start = time.perf_counter()
        graph = configure_backend(original_graph, backend, num_thread)
        preload_time = time.perf_counter() - preload_start

        result, execution_time, peak_memory, error = run_algorithm(
            graph=graph,
            algo_config=algo_config,
            num_thread=num_thread,
            backend=backend,
        )

        execution_time_with_preloading = execution_time + preload_time

        validation_status, validation_message = validate_results(
            result, algo_config, graph
        )
        metrics = collect_metrics(
            execution_time=execution_time,
            execution_time_with_preloading=execution_time_with_preloading,
            peak_memory=peak_memory,
            graph=graph,
            algo_config=algo_config,
            backend=backend,
            dataset_name=dataset_config.name,
            num_thread=num_thread,
            validation_status=validation_status,
            validation_message=validation_message,
            error=error,
        )
    except Exception as e:
        metrics = collect_metrics(
            execution_time=float("nan"),
            execution_time_with_preloading=float("nan"),
            peak_memory=0,
            graph=original_graph,
            algo_config=algo_config,
            backend=backend,
            dataset_name=dataset_config.name,
            num_thread=num_thread,
            validation_status="failed",
            validation_message=str(e),
            error=str(e),
        )
    finally:
        teardown_specific(backend)
        logger.info("Teared down resources after benchmarking.")

    return metrics


@flow(
    name="multiverse_benchmark",
    flow_run_name=f"run_{run_uuid}",
    task_runner=ThreadPoolTaskRunner(max_workers=int(os.getenv("MAX_WORKERS"))),
)
async def benchmark_suite(
    algorithms: list[AlgorithmConfig],
    datasets: list[DatasetConfig],
    backends: list[str],
    threads: list[int],
    graphs: dict[str, tuple[nx.Graph, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run the full suite of benchmarks in parallel using asyncio."""
    logger = get_run_logger()
    logger.info("Starting benchmark suite.")

    def create_benchmark_subflow(name_suffix: str, resource_type: str, num_thread: int):
        @flow(
            name="benchmark_subflow",
            flow_run_name=name_suffix,
            task_runner=DaskTaskRunner(
                cluster_kwargs={
                    "n_workers": 1,
                    "resources": {resource_type: 1},
                    "threads_per_worker": num_thread,
                    "processes": False,
                    "memory_limit": "2GB",
                }
            ),
        )
        async def benchmark_subflow(
            backend: str,
            num_thread: int,
            algo_config: AlgorithmConfig,
            dataset_config: DatasetConfig,
            original_graph: nx.Graph,
        ) -> dict[str, Any] | None:
            return await run_single_benchmark(
                backend,
                num_thread,
                algo_config,
                dataset_config,
                original_graph,
            )

        return benchmark_subflow

    tasks = []
    for backend, num_thread, algo_config, dataset_config in product(
        backends, threads, algorithms, datasets
    ):
        dataset_name = dataset_config.name
        if dataset_name not in graphs:
            logger.warning(f"Dataset '{dataset_name}' not cached. Skipping.")
            continue
        original_graph, _ = graphs[dataset_name]
        resource_type = "GPU" if backend == "cugraph" else "process"
        name_suffix = f"{algo_config.name}_{dataset_name}_{backend}_{num_thread}"

        unique_subflow = create_benchmark_subflow(
            name_suffix, resource_type, num_thread
        )

        tasks.append(
            unique_subflow(
                backend=backend,
                num_thread=num_thread,
                algo_config=algo_config,
                dataset_config=dataset_config,
                original_graph=original_graph,
            )
        )

    return await asyncio.gather(*tasks, return_exceptions=True)


async def main_benchmark(
    results_dir: Path = Path("results"),
):
    """Execute benchmarks using Prefect."""
    final_results = []
    timestamp = str(time.strftime("%Y%m%d%H%M%S"))

    try:
        config = load_config()
        algorithms = config["algorithms"]
        datasets = config["datasets"]
        env_data = config["env_data"]

        # parse user-specified constraints for Python versions, backends, threads
        pythons = env_data.get("pythons", ["3.10"])
        backend_configs = env_data.get("backend", {"networkx": ["networkx==3.4.1"]})
        num_threads = env_data.get("num_threads", [1])
        if not isinstance(num_threads, list):
            num_threads = [num_threads]
        num_threads = [int(x) for x in num_threads]

        # check Python version
        actual_python_version = get_python_version()  # e.g. "3.10.12"
        if not any(py_ver in actual_python_version for py_ver in pythons):
            logger.error(
                f"No requested Python version matches the actual interpreter "
                f"({actual_python_version}). Aborting."
            )
            return

        available_backends = (
            list_available_backends()
        )  # e.g. {"networkx": "3.4.2", "graphblas_algorithms": "2023.10.0"}

        # filter out backends not installed or not matching pinned versions
        chosen_backends = []
        backend_version_map = {}
        for backend, requested_versions in backend_configs.items():
            installed_version = available_backends.get(backend)
            if not installed_version:
                continue

            matched = False
            for req_ver in requested_versions:
                if "==" in req_ver:
                    # e.g. "networkx==3.4.1"
                    _, pinned_version = req_ver.split("==", 1)
                    if pinned_version == installed_version:
                        matched = True
                        break
                else:
                    # if the user didn't pin a version, accept the installed version
                    matched = True
                    break

            if matched:
                chosen_backends.append(backend)
                backend_version_map[backend] = installed_version

        if not chosen_backends:
            logger.error("No valid backends found or matched. Exiting.")
            return
        logger.info(
            f"Chosen backends: {chosen_backends} "
            f"(Installed versions: {backend_version_map})"
        )

        graphs = setup_cache(datasets)

        results = await benchmark_suite(
            algorithms=algorithms,
            datasets=datasets,
            backends=chosen_backends,
            threads=num_threads,
            graphs=graphs,
        )

        for run_result in results:
            if isinstance(run_result, BaseException):
                logger.error("A subflow raised an exception: %s", run_result)
                continue

            if isinstance(run_result, dict):
                run_result["python_version"] = actual_python_version
                bname = run_result.get("backend", "unknown")
                run_result["backend_version"] = backend_version_map.get(
                    bname, "unknown"
                )
                final_results.append(run_result)

    finally:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        out_file = results_dir / f"{run_uuid}_{timestamp}.json"

        with out_file.open("w") as f:
            json.dump([r for r in final_results if isinstance(r, dict)], f, indent=4)

        logger.info(f"Benchmark suite results saved to {out_file}")
