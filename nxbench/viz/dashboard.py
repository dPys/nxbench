import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from nxbench.benchmarks.benchmark import BenchmarkResult

logger = logging.getLogger("nxbench")


class BenchmarkDashboard:
    """Dashboard for visualizing benchmark results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def load_results(self) -> List[BenchmarkResult]:
        """Load benchmark results from ASV's results directory."""
        results = []

        for commit_dir in self.results_dir.iterdir():
            if (
                commit_dir.is_dir()
                and commit_dir.name != "machine.json"
                and commit_dir.name != "benchmarks.json"
            ):
                for env_file in commit_dir.glob("*.json"):
                    with env_file.open("r") as f:
                        data = json.load(f)
                        for bench_name, bench_data in data.get("results", {}).items():
                            params = bench_data.get("params", {})
                            stats = bench_data.get("stats", {})
                            result_value = bench_data.get("result", {})

                            asv_result = {
                                "name": bench_name,
                                "stats": stats,
                                "params": params,
                                "result": result_value,
                            }

                            parts = bench_name.split("_")
                            if len(parts) >= 4:
                                algorithm = parts[2]
                                dataset = parts[3]
                                backend = parts[4] if len(parts) > 4 else "unknown"
                            else:
                                algorithm = bench_name
                                dataset = "unknown"
                                backend = "unknown"

                            dummy_graph = nx.Graph()
                            dummy_graph.graph["name"] = dataset

                            benchmark_result = BenchmarkResult.from_asv_result(
                                asv_result, dummy_graph
                            )
                            results.append(benchmark_result)
        return results

    def compare_results(
        self, baseline: str, comparison: str, threshold: float
    ) -> List[Dict]:
        """
        Compare benchmark results between two algorithms or datasets.

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
        # Load all results
        results = self.load_results()
        comparisons = []

        # Filter results for baseline and comparison
        baseline_results = [res for res in results if res.algorithm == baseline]
        comparison_results = [res for res in results if res.algorithm == comparison]

        # Compare execution times
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
                f.write("<hr>")
            f.write("</body></html>")
        logger.info(f"Static report generated at {report_path}")
