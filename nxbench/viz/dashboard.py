import logging
from pathlib import Path

from nxbench.benchmarks.export import ResultsExporter
from nxbench.benchmarks.utils import get_benchmark_config
from nxbench.data.loader import BenchmarkDataManager

logger = logging.getLogger("nxbench")


class BenchmarkDashboard:
    """Dashboard for visualizing benchmark results."""

    def __init__(self, results_dir: str = "results"):
        """Initialize the dashboard.

        Parameters
        ----------
        results_dir : str
            Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.data_manager = BenchmarkDataManager()
        self.benchmark_config = get_benchmark_config()
        self.exporter = ResultsExporter(results_dir)

    def compare_results(
        self, baseline: str, comparison: str, threshold: float
    ) -> list[dict]:
        """Compare benchmark results between two algorithms or datasets.

        Parameters
        ----------
        baseline : str
            The name of the baseline algorithm or dataset
        comparison : str
            The name of the algorithm or dataset to compare against the baseline
        threshold : float
            The threshold for highlighting significant differences

        Returns
        -------
        list[dict]
            A list of dictionaries containing comparison results
        """
        results = self.exporter.load_results()

        baseline_results = [res for res in results if res.algorithm == baseline]
        comparison_results = [res for res in results if res.algorithm == comparison]

        comparisons = []
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

                    machine_info = self.exporter.get_machine_info()
                    comparisons.append(
                        {
                            "algorithm": base_res.algorithm,
                            "dataset": base_res.dataset,
                            "backend": base_res.backend,
                            "baseline_time": base_res.execution_time,
                            "comparison_time": comp_res.execution_time,
                            "percent_change": percent_change,
                            "significant": significant,
                            "machine": machine_info.get("machine", "unknown"),
                            "cpu": machine_info.get("cpu", "unknown"),
                            "os": machine_info.get("os", "unknown"),
                        }
                    )
        return comparisons

    def generate_static_report(self):
        """Generate a static HTML report of benchmark results."""
        results = self.exporter.load_results()
        machine_info = self.exporter.get_machine_info()

        report_path = self.results_dir / "report.html"
        with report_path.open("w") as f:
            f.write(
                """
                <html>
                <head>
                    <title>Benchmark Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .header { background: #f5f5f5; padding: 20px; margin-bottom:
                        30px; }
                        .result { border: 1px solid #ddd; padding: 15px; margin: 10px
                        0; }
                        .metric { margin: 5px 0; }
                        hr { margin: 30px 0; }
                    </style>
                </head>
                <body>
                """
            )

            f.write('<div class="header">')
            f.write("<h1>Benchmark Report</h1>")
            f.write("<h2>System Information</h2>")
            f.write(f"<p>Machine: {machine_info.get('machine', 'unknown')}</p>")
            f.write(f"<p>CPU: {machine_info.get('cpu', 'unknown')}</p>")
            f.write(f"<p>OS: {machine_info.get('os', 'unknown')}</p>")
            f.write(f"<p>RAM: {machine_info.get('ram', 'unknown')}</p>")
            f.write("</div>")

            for res in results:
                f.write('<div class="result">')
                f.write(f"<h2>Algorithm: {res.algorithm}</h2>")
                f.write(f'<div class="metric">Dataset: {res.dataset}</div>')
                f.write(f'<div class="metric">Backend: {res.backend}</div>')
                f.write(
                    f'<div class="metric">Execution Time: {res.execution_time:.6f} '
                    f"seconds</div>"
                )
                f.write(
                    f'<div class="metric">Memory Used: {res.memory_used:.6f} MB</div>'
                )
                f.write(f'<div class="metric">Number of Nodes: {res.num_nodes}</div>')
                f.write(f'<div class="metric">Number of Edges: {res.num_edges}</div>')
                f.write(f'<div class="metric">Directed: {res.is_directed}</div>')
                f.write(f'<div class="metric">Weighted: {res.is_weighted}</div>')
                f.write("</div>")

            f.write("</body></html>")

        logger.info(f"Static report generated at {report_path}")

    def export_results(
        self, export_format: str = "csv", output_path: Path | None = None
    ):
        """Export benchmark results in the specified format.

        Parameters
        ----------
        export_format : str
            Export format ('csv' or 'sql')
        output_path : Path, optional
            Path for output file (for CSV export)
        """
        if export_format.lower() == "csv":
            if output_path is None:
                output_path = self.results_dir / "results.csv"
            self.exporter.to_csv(output_path)
        elif export_format.lower() == "sql":
            self.exporter.to_sql(output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def get_results_df(self):
        """Get benchmark results as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark results
        """
        return self.exporter.to_dataframe()
