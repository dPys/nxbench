import logging
import os
import warnings
from pathlib import Path
from typing import Optional
import subprocess

import click
import pandas as pd

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
@click.pass_context
def cli(ctx, verbose: int, config: Optional[Path]):
    """NetworkX Benchmarking Suite CLI."""
    # Set logging level based on verbosity
    log_level = max(logging.INFO - 10 * verbose, logging.DEBUG)
    logging.basicConfig(level=log_level)

    if config:
        os.environ["NXBENCH_CONFIG_FILE"] = str(config)
        logger.info(f"Using config file: {config}")

    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = config


@cli.group()
@click.pass_context
def data(ctx):
    """Dataset management commands."""
    pass


@data.command()
@click.argument("name")
@click.option("--category", type=str, help="Dataset category.")
@click.option(
    "--force/--no-force", default=False, help="Force download even if exists."
)
@click.pass_context
def download(ctx, name: str, category: Optional[str], force: bool):
    """Download a specific dataset."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for download: {config}")

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
@click.pass_context
def list_datasets(
    ctx,
    category: Optional[str],
    min_nodes: Optional[int],
    max_nodes: Optional[int],
    directed: Optional[bool],
):
    """List available datasets."""
    import asyncio

    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for listing datasets: {config}")

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
@click.pass_context
def benchmark(ctx):
    """Benchmark management commands."""
    pass


@benchmark.command(name="run")
@click.option("--backend", type=str, default="all", help="Backend to benchmark.")
@click.option("--collection", type=str, default="all", help="Graph collection to use.")
@click.pass_context
def run_benchmark(ctx, backend: str, collection: str):
    """Run benchmarks."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for benchmark run: {config}")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], universal_newlines=True
        ).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get git hash: {e}")
        raise click.ClickException("Could not determine git commit hash")

    cmd_parts = ["asv", "run", "--quick", f"--set-commit-hash={git_hash}"]

    if backend != "all" or collection != "all":
        benchmark_pattern = "GraphBenchmark.time_"
        if collection != "all":
            benchmark_pattern = f"{benchmark_pattern}.*{collection}"
        if backend != "all":
            benchmark_pattern = f"{benchmark_pattern}.*{backend}"
        cmd_parts.extend(["-b", f'"{benchmark_pattern}"'])

    cmd_parts.append("--python=same")

    cmd = " ".join(cmd_parts)
    logger.info(f"Running command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark run failed: {e}")
        raise click.ClickException("Benchmark run failed")


@benchmark.command()
@click.argument("result_file", type=Path)
@click.option("--format", type=click.Choice(["json", "csv"]), default="csv")
@click.pass_context
def export(ctx, result_file: Path, format: str):
    """Export benchmark results."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for export: {config}")

    dashboard = BenchmarkDashboard(results_dir="results")
    results = dashboard.load_results()

    if not results:
        logger.error("No benchmark results found.")
        click.echo("No benchmark results found.")
        return

    records = []
    for result in results:
        param_lists = result.parameters
        datasets = [d.strip("'") for d in param_lists[0]]
        backends = [b.strip("'") for b in param_lists[1]]

        algo_name = result.algorithm.split(".")[-1]
        if algo_name.startswith("time_"):
            algo_name = algo_name[5:]

        execution_times = (
            result.execution_time
            if isinstance(result.execution_time, list)
            else [result.execution_time]
        )

        for i, (dataset, backend, time) in enumerate(
            zip(datasets, backends, execution_times)
        ):
            record = {
                "algorithm": algo_name,
                "dataset": dataset,
                "backend": backend,
                "execution_time": time,
                "memory_used": (
                    result.memory_used
                    if isinstance(result.memory_used, (int, float))
                    else 0.0
                ),
                "num_nodes": result.num_nodes,
                "num_edges": result.num_edges,
                "is_directed": result.is_directed,
                "is_weighted": result.is_weighted,
            }
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(["algorithm", "dataset", "backend"])

    df["execution_time"] = df["execution_time"].map("{:.6f}".format)
    df["memory_used"] = df["memory_used"].map("{:.2f}".format)

    if format == "csv":
        df.to_csv(result_file, index=False)
    else:
        df.to_json(result_file, orient="records")

    logger.info(f"Exported results to {result_file}")


@benchmark.command()
@click.argument("baseline", type=str)
@click.argument("comparison", type=str)
@click.option("--threshold", type=float, default=0.05)
@click.pass_context
def compare(ctx, baseline: str, comparison: str, threshold: float):
    """Compare benchmark results."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for compare: {config}")

    subprocess.run(["asv", "compare", baseline, comparison, "-f", str(threshold)])


@cli.group()
@click.pass_context
def viz(ctx):
    """Visualization commands."""
    pass


@viz.command()
@click.option("--port", type=int, default=8050)
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def serve(ctx, port: int, debug: bool):
    """Launch visualization dashboard."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for viz serve: {config}")

    from nxbench.viz.app import run_server

    run_server(port=port, debug=debug)


@viz.command()
@click.pass_context
def publish(ctx):
    """Generate static benchmark report."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for viz publish: {config}")

    subprocess.run(["asv", "publish"])
    dashboard = BenchmarkDashboard()
    dashboard.generate_static_report()


@cli.group()
@click.pass_context
def validate(ctx):
    """Validation commands."""
    pass


@validate.command()
@click.argument("result_file", type=Path)
@click.pass_context
def check(ctx, result_file: Path):
    """Validate benchmark results."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for validate check: {config}")

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


def main():
    cli()


if __name__ == "__main__":
    main()
