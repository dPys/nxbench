import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

from nxbench.profile.benchmark import BenchmarkResult

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


class BenchmarkDashboard:
    """Dashboard for visualizing benchmark results."""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)

    def load_results(self) -> Dict[str, List[BenchmarkResult]]:
        """Load benchmark results."""
        results = {}

        for result_file in self.results_dir.glob("*.json"):
            with open(result_file) as f:
                data = json.load(f)
                machine_name = data.get("machine_info", {}).get("node", "unknown")
                result_list = [
                    BenchmarkResult(**res) for res in data.get("benchmarks", [])
                ]
                results[machine_name] = result_list

        return results

    def compare_results(
        self, baseline: str, comparison: str, threshold: float
    ) -> List[Dict]:
        """
        Compare benchmark results between two datasets or algorithms.

        Parameters
        ----------
        baseline : str
            The name of the baseline dataset or algorithm.
        comparison : str
            The name of the dataset or algorithm to compare against the baseline.
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

        # Flatten results into a list
        all_results = []
        for machine_results in results.values():
            all_results.extend(machine_results)

        # Filter results for baseline and comparison
        baseline_results = [
            res
            for res in all_results
            if res.algorithm == baseline or res.dataset == baseline
        ]
        comparison_results = [
            res
            for res in all_results
            if res.algorithm == comparison or res.dataset == comparison
        ]

        # Compare execution times
        for base_res in baseline_results:
            for comp_res in comparison_results:
                if (
                    base_res.algorithm == comp_res.algorithm
                    and base_res.dataset == comp_res.dataset
                ):
                    time_diff = comp_res.execution_time - base_res.execution_time
                    percent_change = (time_diff / base_res.execution_time) * 100
                    significant = abs(percent_change) >= (threshold * 100)
                    comparisons.append(
                        {
                            "algorithm": base_res.algorithm,
                            "dataset": base_res.dataset,
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
            for machine, res_list in results.items():
                f.write(f"<h2>Machine: {machine}</h2>")
                f.write("<table border='1'>")
                f.write(
                    "<tr><th>Algorithm</th><th>Dataset</th><th>Execution Time (s)</th></tr>"
                )
                for res in res_list:
                    f.write(
                        f"<tr><td>{res.algorithm}</td><td>{res.dataset}</td>"
                        f"<td>{res.execution_time:.6f}</td></tr>"
                    )
                f.write("</table>")
            f.write("</body></html>")
        logger.info(f"Static report generated at {report_path}")
