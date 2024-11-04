import logging
import warnings
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import subprocess

from nxbench.config import configure_benchmarks, load_default_config, BenchmarkConfig
from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import DatasetConfig
from nxbench.data.repository import NetworkRepository
from nxbench.viz.dashboard import BenchmarkDashboard

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to config file.",
)
def cli(verbose: int, config: Optional[Path]):
    """NetworkX Benchmarking Suite CLI."""
    log_level = max(logging.INFO - 10 * verbose, logging.DEBUG)
    logging.basicConfig(level=log_level)

    if config:
        benchmark_config = BenchmarkConfig.from_yaml(config)
    else:
        benchmark_config = load_default_config()
    configure_benchmarks(benchmark_config)


@cli.group()
def data():
    """Dataset management commands."""
    pass


@data.command()
@click.argument("name")
@click.option("--category", type=str, help="Dataset category.")
@click.option(
    "--force/--no-force", default=False, help="Force download even if exists."
)
def download(name: str, category: Optional[str], force: bool):
    """Download a specific dataset."""
    data_manager = BenchmarkDataManager()
    dataset_config = DatasetConfig(name=name, source=category or "networkrepository")
    try:
        graph, metadata = data_manager.load_network_sync(dataset_config)
        logger.info(f"Successfully downloaded dataset: {name}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")


@data.command()
@click.option("--category", type=str, help="Filter by category.")
@click.option("--min-nodes", type=int, help="Minimum number of nodes.")
@click.option("--max-nodes", type=int, help="Maximum number of nodes.")
@click.option("--directed/--undirected", default=None, help="Filter by directedness.")
def list_datasets(
    category: Optional[str],
    min_nodes: Optional[int],
    max_nodes: Optional[int],
    directed: Optional[bool],
):
    """List available datasets."""
    import asyncio

    async def list_networks():
        async with NetworkRepository() as repo:
            networks = await repo.list_networks(
                category=category,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                directed=directed,
            )
            df = pd.DataFrame([n.__dict__ for n in networks])
            click.echo(df.to_string())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(list_networks())


@cli.group()
def benchmark():
    """Benchmark management commands."""
    pass


@benchmark.command(name="run")
@click.option("--backend", type=str, default="all", help="Backend to benchmark.")
@click.option("--collection", type=str, default="all", help="Graph collection to use.")
@click.option("--profile/--no-profile", default=False, help="Enable profiling.")
@click.option("--asv/--no-asv", default=True, help="Use ASV for benchmarking.")
def run_benchmark(backend: str, collection: str, profile: bool, asv: bool):
    """Run benchmarks."""
    if asv:
        cmd = ["asv", "run", "--quick"]
        if profile:
            cmd.append("--profile")
        if backend != "all":
            cmd.extend(["-b", f".*{backend}.*"])
        if collection != "all":
            cmd.extend(["-b", f".*{collection}.*"])
        subprocess.run(cmd)
    else:
        from nxbench.profile.benchmark import run_benchmarks

        results = run_benchmarks(
            backend=backend, collection=collection, profile=profile
        )
        logger.info(f"Completed {len(results)} benchmarks")


@benchmark.command()
@click.argument("result_file", type=Path)
@click.option("--format", type=click.Choice(["json", "csv"]), default="csv")
def export(result_file: Path, format: str):
    """Export benchmark results."""
    dashboard = BenchmarkDashboard()
    results = dashboard.load_results()

    if format == "csv":
        pd.DataFrame([r.__dict__ for r in results]).to_csv(result_file)
    else:
        pd.DataFrame([r.__dict__ for r in results]).to_json(result_file)

    logger.info(f"Exported results to {result_file}")


@benchmark.command()
@click.argument("baseline", type=str)
@click.argument("comparison", type=str)
@click.option("--threshold", type=float, default=0.05)
@click.option("--asv/--no-asv", default=True)
def compare(baseline: str, comparison: str, threshold: float, asv: bool):
    """Compare benchmark results."""
    if asv:
        subprocess.run(["asv", "compare", baseline, comparison, "-f", str(threshold)])
    else:
        dashboard = BenchmarkDashboard()
        results = dashboard.compare_results(baseline, comparison, threshold)
        click.echo(pd.DataFrame(results).to_string())


@cli.group()
def viz():
    """Visualization commands."""
    pass


@viz.command()
@click.option("--port", type=int, default=8050)
@click.option("--debug/--no-debug", default=False)
def serve(port: int, debug: bool):
    """Launch visualization dashboard."""
    from nxbench.viz.app import run_server

    run_server(port=port, debug=debug)


@viz.command()
def publish():
    """Generate static benchmark report."""
    subprocess.run(["asv", "publish"])
    dashboard = BenchmarkDashboard()
    dashboard.generate_static_report()


@cli.group()
def validate():
    """Validation commands."""
    pass


@validate.command()
@click.argument("result_file", type=Path)
def check(result_file: Path):
    """Validate benchmark results."""
    from nxbench.validation.registry import BenchmarkValidator

    df = pd.read_json(result_file)
    validator = BenchmarkValidator()

    for _, row in df.iterrows():
        result = row["result"]
        algorithm_name = row["algorithm"]
        graph = None
        try:
            validator.validate_result(result, algorithm_name, graph, raise_errors=True)
            logger.info(f"Validation passed for algorithm '{algorithm_name}'")
        except Exception as e:
            logger.error(f"Validation failed for algorithm '{algorithm_name}': {e}")


if __name__ == "__main__":
    cli()
