import json
import logging
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
            return MachineInfo(**data)

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
            result = BenchmarkResult.from_asv_result(asv_result, graph)
            result.metadata = metadata
            if self._machine_info:
                result.metadata.update(
                    {
                        "machine_arch": self._machine_info.arch,
                        "machine_cpu": self._machine_info.cpu,
                        "machine_name": self._machine_info.machine,
                        "machine_os": self._machine_info.os,
                        "machine_ram": self._machine_info.ram,
                    }
                )
        except Exception:
            logger.exception(
                f"Failed to create benchmark result for {algorithm} on {dataset}"
            )
            return None
        else:
            return result

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

            for bench_name, bench_data in data.get("results", {}).items():
                if not isinstance(bench_data, list) or len(bench_data) < 2:
                    logger.warning(f"Unexpected bench_data format for {bench_name}")
                    continue

                measurements = bench_data[0]
                params_info = bench_data[1]

                if not isinstance(params_info, list) or len(params_info) != 2:
                    logger.warning(f"Unexpected params_info format for {bench_name}")
                    continue

                datasets = [name.strip("'") for name in params_info[0]]
                backends = [name.strip("'") for name in params_info[1]]
                algorithm = bench_name.split(".")[-1].replace("track_", "")

                for (dataset, backend), measurement in zip(
                    list(zip(datasets, backends)), measurements
                ):
                    execution_time, memory_used = self._parse_measurement(measurement)

                    result = self._create_benchmark_result(
                        algorithm, dataset, backend, execution_time, memory_used
                    )
                    if result:
                        results.append(result)

        self._cached_results = results
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark results
        """
        results = self.load_results()
        records = []
        for result in results:
            record = {
                "algorithm": result.algorithm,
                "dataset": result.dataset,
                "backend": result.backend,
                "execution_time": result.execution_time,
                "memory_used": result.memory_used,
                "num_nodes": result.num_nodes,
                "num_edges": result.num_edges,
                "is_directed": result.is_directed,
                "is_weighted": result.is_weighted,
            }
            record.update(result.metadata)
            records.append(record)
        return pd.DataFrame(records)

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
