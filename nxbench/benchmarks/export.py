import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from nxbench.benchmarks.config import BenchmarkResult, MachineInfo
from nxbench.benchmarks.utils import get_benchmark_config, get_python_version
from nxbench.data.db import BenchmarkDB
from nxbench.data.loader import BenchmarkDataManager

logger = logging.getLogger("nxbench")


class ResultsExporter:
    """Class for loading and exporting benchmark results."""

    def __init__(self, results_dir: str | Path = "results"):
        """Initialize the results exporter.

        Parameters
        ----------
        results_dir : str or Path
            Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.data_manager = BenchmarkDataManager()
        self.benchmark_config = get_benchmark_config()

        self._db = None
        self._cached_results = None

        machine_dirs = [
            d
            for d in self.results_dir.iterdir()
            if d.is_dir() and d.name not in {"__pycache__"}
        ]

        if not machine_dirs:
            logger.warning(f"No machine directories found in {results_dir}")
            self._machine_dir = None
            self._machine_info = None
            return

        self._machine_dir = machine_dirs[0]
        self._machine_info = self._load_machine_info()

        if self._machine_info:
            logger.debug(f"Machine Information: {self._machine_info}")
        else:
            logger.warning("Machine information could not be loaded.")

    def _load_machine_info(self) -> MachineInfo | None:
        """Load machine information from json file."""
        if not self._machine_dir:
            logger.warning("No machine directory available")
            return None

        machine_file = self._machine_dir / "machine.json"
        try:
            with machine_file.open() as f:
                data = json.load(f)
        except Exception:
            logger.warning(
                f"Failed to load machine information from {machine_file}", exc_info=True
            )
            return None
        else:
            try:
                machine_info = MachineInfo(**data)
                logger.debug(f"Loaded machine info: {machine_info}")
            except TypeError:
                logger.warning("MachineInfo structure mismatch")
                return None
            else:
                return machine_info

    def _parse_measurement(
        self, measurement: dict | int | float | None
    ) -> tuple[float, float]:
        """Parse measurement data into execution time and memory usage.

        Parameters
        ----------
        measurement : dict or int or float or None
            Raw measurement data

        Returns
        -------
        tuple
            (execution_time, memory_used)
        """
        if measurement is None:
            return float("nan"), float("nan")

        if isinstance(measurement, dict):
            execution_time = measurement.get("execution_time", float("nan"))
            memory_used = measurement.get("memory_used", float("nan"))

            if not isinstance(execution_time, (int, float)) or execution_time is None:
                execution_time = float("nan")
            if not isinstance(memory_used, (int, float)) or memory_used is None:
                memory_used = float("nan")

        elif isinstance(measurement, (int, float)):
            execution_time = float(measurement)
            memory_used = 0.0
        else:
            execution_time = float("nan")
            memory_used = float("nan")

        return execution_time, memory_used

    def _create_benchmark_result(
        self,
        algorithm: str,
        dataset: str,
        backend: str,
        execution_time: float,
        memory_used: float,
        num_thread: int | None,
        commit_hash: str,
        date: int,
    ) -> BenchmarkResult | None:
        """Create a benchmark result object."""
        dataset_config = next(
            (ds for ds in self.benchmark_config.datasets if ds.name == dataset),
            None,
        )

        if dataset_config is None:
            logger.warning(f"No DatasetConfig found for dataset '{dataset}'")
            graph = nx.Graph()
            graph.graph["name"] = dataset
            metadata = {}
        else:
            try:
                graph, metadata = self.data_manager.load_network_sync(dataset_config)
            except Exception:
                logger.exception(f"Failed to load network for dataset '{dataset}'")
                return None

        asv_result = {
            "algorithm": algorithm,
            "dataset": dataset,
            "backend": backend,
            "execution_time": execution_time,
            "memory_used": memory_used,
        }

        try:
            if num_thread is None and self._machine_info:
                try:
                    num_thread = int(self._machine_info.num_cpu)
                    logger.debug(
                        f"Falling back to num_thread from machine_info: {num_thread}"
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid num_cpu value in machine_info: "
                        f"{self._machine_info.num_cpu}"
                    )
                    num_thread = 1

            if num_thread is None:
                num_thread = 1

            logger.debug(f"Final num_thread for benchmark: {num_thread}")

            result = BenchmarkResult.from_asv_result(
                asv_result,
                graph,
                num_thread,
                commit_hash=commit_hash,
                date=date,
            )
            result.metadata = metadata

            logger.debug(f"BenchmarkResult created: {result}")
        except Exception:
            logger.exception(
                f"Failed to create benchmark result for {algorithm} on {dataset}"
            )
            return None
        else:
            return result

    def _parse_benchmark_name(self, bench_name: str) -> tuple[str, str, str] | None:
        """Parse the benchmark name to extract algorithm, dataset, and backend."""
        pattern = (
            r"^(?:benchmark\.)?GraphBenchmark\.track_(.+?)_([^_]+)_([^_]+)(?:_\d+)?$"
        )
        match = re.search(pattern, bench_name)
        if match:
            algorithm = match.group(1)
            dataset = match.group(2)
            backend = match.group(3)
            logger.debug(
                f"Parsed benchmark name '{bench_name}': algorithm={algorithm}, "
                f"dataset={dataset}, backend={backend}"
            )
            return algorithm, dataset, backend
        logger.warning(
            f"Benchmark name '{bench_name}' does not match expected patterns."
        )
        return None

    def load_results(self) -> list[BenchmarkResult]:
        """Load benchmark results and machine information."""
        if self._cached_results is not None:
            return self._cached_results

        if not self._machine_dir:
            logger.warning("No machine directory available for loading results")
            return []

        results = []
        for result_file in self._machine_dir.glob("*.json"):
            if result_file.name == "machine.json":
                continue

            try:
                with result_file.open() as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logger.exception(f"Failed to decode JSON from {result_file}")
                continue

            logger.debug(f"Processing result file: {result_file}")

            env_vars = data.get("params", {}).get("env_vars", {})
            logger.debug(f"'params.env_vars' extracted: {env_vars}")

            if not env_vars:
                env_vars = data.get("env_vars", {})
                logger.debug(f"Falling back to top-level 'env_vars': {env_vars}")

            num_thread_str = env_vars.get("NUM_THREAD") or env_vars.get("NUM_THREADS")
            logger.debug(f"Extracted num_thread_str from env_vars: {num_thread_str}")

            if not num_thread_str:
                env_name = data.get("env_name", "")
                logger.debug(f"'env_name' extracted: {env_name}")
                match = re.search(r"NUM_THREAD(\d+)", env_name)
                if match:
                    num_thread = int(match.group(1))
                    logger.debug(f"Extracted num_thread: {num_thread} from 'env_name'")
                elif "num_cpu" in data.get("params", {}):
                    try:
                        num_thread = int(data["params"]["num_cpu"])
                        logger.debug(
                            f"Falling back to num_cpu from params: {num_thread}"
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid num_cpu value in params: "
                            f"{data['params']['num_cpu']}"
                        )
                        num_thread = None
                else:
                    logger.warning(f"Could not extract NUM_THREAD from {result_file}")
                    num_thread = None
            else:
                try:
                    num_thread = int(num_thread_str)
                    logger.debug(f"Extracted num_thread: {num_thread} from env_vars")
                except (TypeError, ValueError):
                    logger.warning(
                        f"Invalid NUM_THREAD value in {result_file}: {num_thread_str}"
                    )
                    num_thread = None  # or set a default value

            commit_hash = data.get("commit_hash", "unknown")
            date = data.get("date", 0)

            for bench_name, bench_data in data.get("results", {}).items():
                if not isinstance(bench_data, list) or len(bench_data) < 2:
                    logger.warning(
                        f"Unexpected bench_data format for {bench_name}: {bench_data}"
                    )
                    continue

                measurements = bench_data[0]
                params_info = bench_data[1]

                logger.debug(f"'params_info' content for {bench_name}: {params_info}")

                if isinstance(params_info, list) and len(params_info) >= 2:
                    datasets = [name.strip("'") for name in params_info[0]]
                    backends = [name.strip("'") for name in params_info[1]]
                elif isinstance(params_info, dict):
                    datasets = params_info.get("datasets", [])
                    backends = params_info.get("backends", [])
                elif isinstance(params_info, list) and len(params_info) == 0:
                    parsed = self._parse_benchmark_name(bench_name)
                    if parsed:
                        algorithm_parsed, dataset, backend = parsed
                        algorithm = algorithm_parsed
                        datasets = [dataset]
                        backends = [backend]
                    else:
                        logger.warning(
                            f"Unable to parse parameters from benchmark name: "
                            f"{bench_name}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Unexpected params_info format for {bench_name}: {params_info}"
                    )
                    continue

                if not (isinstance(params_info, list) and len(params_info) == 0):
                    algorithm = bench_name.split(".")[-1].replace("track_", "")

                logger.debug(
                    f"Algorithm: {algorithm}, Dataset: {datasets}, Backend: {backends}"
                )

                if len(backends) == 1 and len(datasets) > 1:
                    backend = backends[0]
                    for dataset, measurement in zip(datasets, measurements):
                        execution_time, memory_used = self._parse_measurement(
                            measurement
                        )

                        result = self._create_benchmark_result(
                            algorithm,
                            dataset,
                            backend,
                            execution_time,
                            memory_used,
                            num_thread,
                            commit_hash,
                            date,
                        )
                        if result:
                            results.append(result)
                            logger.debug(
                                f"Added BenchmarkResult: algorithm={algorithm}, "
                                f"dataset={dataset}, "
                                f"backend={backend}, num_thread={result.num_thread}"
                            )
                elif len(backends) == len(datasets):
                    for dataset, backend, measurement in zip(
                        datasets, backends, measurements
                    ):
                        execution_time, memory_used = self._parse_measurement(
                            measurement
                        )

                        result = self._create_benchmark_result(
                            algorithm,
                            dataset,
                            backend,
                            execution_time,
                            memory_used,
                            num_thread,
                            commit_hash,
                            date,
                        )
                        if result:
                            results.append(result)
                            logger.debug(
                                f"Added BenchmarkResult: algorithm={algorithm}, "
                                f"dataset={dataset}, "
                                f"backend={backend}, num_thread={result.num_thread}"
                            )
                else:
                    logger.warning(
                        f"Mismatch between number of backends and datasets for "
                        f"{bench_name} in {result_file}"
                    )
                    continue

        self._cached_results = results
        logger.info(f"Total benchmark results loaded: {len(results)}")
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark results
        """
        results = self.load_results()
        if not results:
            logger.error("No benchmark results found.")
            raise ValueError("No benchmark results found.")

        machine_info = self.get_machine_info()
        cpu = machine_info.get("cpu", "unknown")
        os_info = machine_info.get("os", "unknown")
        version = machine_info.get("version", "unknown")
        python_version = get_python_version()

        records = []
        for result in results:
            record = {
                "algorithm": result.algorithm,
                "dataset": result.dataset,
                "backend": result.backend,
                "execution_time": result.execution_time,
                "memory_used": result.memory_used,
                "num_thread": result.num_thread,
                "num_nodes": result.num_nodes,
                "num_edges": result.num_edges,
                "is_directed": result.is_directed,
                "is_weighted": result.is_weighted,
                "commit_hash": result.commit_hash,
                "cpu": cpu,
                "os": os_info,
                "version": version,
                "python_version": python_version,
            }
            metadata = {k: v for k, v in result.metadata.items() if k != "date"}
            record.update(metadata)
            records.append(record)
        df = pd.DataFrame(records)
        logger.debug(f"DataFrame created with shape: {df.shape}")
        return df

    def to_csv(self, output_path: str | Path) -> None:
        """Export results to CSV file.

        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to CSV: {output_path}")

    def to_sql(
        self,
        db_path: str | Path | None = None,
        if_exists: str = "replace",
    ) -> None:
        """Export results to SQLite database using BenchmarkDB.

        Parameters
        ----------
        db_path : str or Path, optional
            Path to SQLite database file. If None, uses default location
        if_exists : str, default='replace'
            How to behave if table exists ('fail', 'replace', or 'append')
        """
        if self._db is None:
            self._db = BenchmarkDB(db_path)

        results = self.load_results()
        machine_info = self.get_machine_info()

        if if_exists == "replace":
            self._db.delete_results()

        self._db.save_results(
            results=results,
            machine_info=machine_info,
            python_version=get_python_version(),
            package_versions=None,  # could be added from pkg_resources if needed
        )

        logger.info(f"Results exported to SQL database: {self._db.db_path}")

    def get_machine_info(self) -> dict[str, Any]:
        """Get machine information as dictionary.

        Returns
        -------
        dict
            Machine information
        """
        if self._machine_info:
            return {
                "arch": self._machine_info.arch,
                "cpu": self._machine_info.cpu,
                "machine": self._machine_info.machine,
                "num_cpu": self._machine_info.num_cpu,
                "os": self._machine_info.os,
                "ram": self._machine_info.ram,
                "version": self._machine_info.version,
            }
        return {}

    def query_sql(
        self,
        algorithm: str | None = None,
        backend: str | None = None,
        dataset: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_pandas: bool = True,
    ) -> pd.DataFrame | list[dict]:
        """Query results from SQL database using BenchmarkDB.

        Parameters
        ----------
        algorithm : str, optional
            Filter by algorithm name
        backend : str, optional
            Filter by backend
        dataset : str, optional
            Filter by dataset
        start_date : str, optional
            Filter results after this date (ISO format)
        end_date : str, optional
            Filter results before this date (ISO format)
        as_pandas : bool, default=True
            Return results as pandas DataFrame

        Returns
        -------
        DataFrame or list of dict
            Filtered benchmark results
        """
        if self._db is None:
            self._db = BenchmarkDB()

        return self._db.get_results(
            algorithm=algorithm,
            backend=backend,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            as_pandas=as_pandas,
        )
