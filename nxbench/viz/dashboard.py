import json
import itertools
import logging
from pathlib import Path
from typing import Dict, List

import networkx as nx

from nxbench.benchmarks.benchmark import BenchmarkResult
from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import get_benchmark_config

logger = logging.getLogger("nxbench")


class BenchmarkDashboard:
    """Dashboard for visualizing benchmark results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data_manager = BenchmarkDataManager()
        self.benchmark_config = get_benchmark_config()

    def load_results(self) -> List[BenchmarkResult]:
        """Load benchmark results from ASV's results directory."""
        results = []

        for commit_dir in self.results_dir.iterdir():
            if commit_dir.is_dir() and commit_dir.name not in {
                "machine.json",
                "benchmarks.json",
            }:
                for env_file in commit_dir.glob("*.json"):
                    with env_file.open("r") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON from {env_file}: {e}")
                            continue

                        for bench_name, bench_data in data.get("results", {}).items():
                            if not isinstance(bench_data, list) or len(bench_data) < 2:
                                logger.warning(
                                    f"Unexpected bench_data format for {bench_name}"
                                )
                                continue

                            measurements = bench_data[0]
                            params_info = bench_data[1]

                            if not (
                                isinstance(params_info, list) and len(params_info) == 2
                            ):
                                logger.warning(
                                    f"Unexpected params_info format for {bench_name}"
                                )
                                continue

                            datasets = [name.strip("'") for name in params_info[0]]
                            backends = [name.strip("'") for name in params_info[1]]

                            param_combinations = list(
                                itertools.product(datasets, backends)
                            )

                            if len(measurements) != len(param_combinations):
                                logger.warning(
                                    f"Number of measurements ({len(measurements)}) does not match "
                                    f"number of parameter combinations ({len(param_combinations)}) "
                                    f"for benchmark {bench_name}"
                                )
                                continue

                            for (dataset, backend), measurement in zip(
                                param_combinations, measurements
                            ):
                                if isinstance(measurement, dict):
                                    execution_time = measurement.get(
                                        "execution_time", float("nan")
                                    )
                                    memory_used = measurement.get(
                                        "memory_used", float("nan")
                                    )
                                    if execution_time is None or isinstance(
                                        execution_time, str
                                    ):
                                        execution_time = float("nan")
                                    if memory_used is None or isinstance(
                                        memory_used, str
                                    ):
                                        memory_used = float("nan")
                                elif isinstance(measurement, (int, float)):
                                    execution_time = float(measurement)
                                    memory_used = 0.0
                                else:
                                    logger.warning(
                                        f"Unsupported measurement type for {bench_name}: {type(measurement)}"
                                    )
                                    execution_time = float("nan")
                                    memory_used = float("nan")

                                algorithm = bench_name.split(".")[-1].replace(
                                    "track_", ""
                                )

                                dataset_config = next(
                                    (
                                        ds
                                        for ds in self.benchmark_config.datasets
                                        if ds.name == dataset
                                    ),
                                    None,
                                )
                                if dataset_config is None:
                                    logger.warning(
                                        f"No DatasetConfig found for dataset '{dataset}'"
                                    )
                                    graph = nx.Graph()
                                    graph.graph["name"] = dataset
                                else:
                                    graph, metadata = (
                                        self.data_manager.load_network_sync(
                                            dataset_config
                                        )
                                    )

                                asv_result = {
                                    "algorithm": algorithm,
                                    "dataset": dataset,
                                    "backend": backend,
                                    "execution_time": execution_time,
                                    "memory_used": memory_used,
                                }

                                try:
                                    benchmark_result = BenchmarkResult.from_asv_result(
                                        asv_result, graph
                                    )
                                    benchmark_result.metadata = metadata
                                    results.append(benchmark_result)
                                except Exception as e:
                                    logger.error(
                                        f"Failed to create BenchmarkResult for {bench_name} "
                                        f"with dataset {dataset} and backend {backend}: {e}"
                                    )

        return results

    def compare_results(
        self, baseline: str, comparison: str, threshold: float
    ) -> List[Dict]:
        """Compare benchmark results between two algorithms or datasets.

        Parameters
        ----------
        baseline : str
            The name of the baseline algorithm or dataset.
        comparison : str
            The name of the algorithm or dataset to compare against the baseline.
        threshold : float
            The threshold for highlighting significant differences.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing comparison results.
        """
        results = self.load_results()
        comparisons = []

        baseline_results = [res for res in results if res.algorithm == baseline]
        comparison_results = [res for res in results if res.algorithm == comparison]

        for base_res in baseline_results:
            for comp_res in comparison_results:
                if (
                    base_res.dataset == comp_res.dataset
                    and base_res.backend == comp_res.backend
                ):
                    time_diff = comp_res.execution_time - base_res.execution_time
                    percent_change = (
                        (time_diff / base_res.execution_time) * 100
                        if base_res.execution_time != 0
                        else 0.0
                    )
                    significant = abs(percent_change) >= (threshold * 100)
                    comparisons.append(
                        {
                            "algorithm": base_res.algorithm,
                            "dataset": base_res.dataset,
                            "backend": base_res.backend,
                            "baseline_time": base_res.execution_time,
                            "comparison_time": comp_res.execution_time,
                            "percent_change": percent_change,
                            "significant": significant,
                        }
                    )
        return comparisons

    def generate_static_report(self):
        """Generate a static HTML report of benchmark results."""
        results = self.load_results()

        report_path = self.results_dir / "report.html"
        with report_path.open("w") as f:
            f.write("<html><head><title>Benchmark Report</title></head><body>")
            f.write("<h1>Benchmark Report</h1>")
            for res in results:
                f.write(f"<h2>Algorithm: {res.algorithm}</h2>")
                f.write(f"<p>Dataset: {res.dataset}</p>")
                f.write(f"<p>Backend: {res.backend}</p>")

                f.write(f"<p>Execution Time: {res.execution_time:.6f} seconds</p>")
                f.write(f"<p>Memory Used: {res.memory_used:.6f} MB</p>")
                f.write(f"<p>Number of Nodes: {res.num_nodes}</p>")
                f.write(f"<p>Number of Edges: {res.num_edges}</p>")
                f.write(f"<p>Directed: {res.is_directed}</p>")
                f.write(f"<p>Weighted: {res.is_weighted}</p>")
                f.write("<hr>")
            f.write("</body></html>")
        logger.info(f"Static report generated at {report_path}")
