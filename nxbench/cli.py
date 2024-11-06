import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

import click
import pandas as pd

from _nxbench.config import _config as package_config
from nxbench.benchmarks.config import DatasetConfig
from nxbench.data.loader import BenchmarkDataManager
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
def cli(ctx, verbose: int, config: Path | None):
    """NetworkX Benchmarking Suite CLI."""
    if verbose >= 2:
        verbosity_level = 2
    elif verbose == 1:
        verbosity_level = 1
    else:
        verbosity_level = 0

    package_config.set_verbosity_level(verbosity_level)

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][verbosity_level]
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


@data.command()
@click.argument("name")
@click.option("--category", type=str, help="Dataset category.")
@click.pass_context
def download(ctx, name: str, category: str | None):
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
        logger.exception(f"Failed to download dataset: {e}")


@data.command()
@click.option("--category", type=str, help="Filter by category.")
@click.option("--min-nodes", type=int, help="Minimum number of nodes.")
@click.option("--max-nodes", type=int, help="Maximum number of nodes.")
@click.option("--directed/--undirected", default=None, help="Filter by directedness.")
@click.pass_context
def list_datasets(
    ctx,
    category: str | None,
    min_nodes: int | None,
    max_nodes: int | None,
    directed: bool | None,
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


@benchmark.command(name="run")
@click.option(
    "--backend",
    type=str,
    multiple=True,
    default=["all"],
    help="Backends to benchmark. Specify multiple values to run for multiple backends.",
)
@click.option("--collection", type=str, default="all", help="Graph collection to use.")
@click.pass_context
def run_benchmark(ctx, backend: tuple[str], collection: str):
    """Run benchmarks."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for benchmark run: {config}")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], universal_newlines=True
        ).strip()
    except subprocess.CalledProcessError as e:
        logger.exception(f"Failed to get git hash: {e}")
        raise click.ClickException("Could not determine git commit hash")

    cmd_parts = [
        "asv",
        "run",
        "--quick",
        f"--set-commit-hash={git_hash}",
    ]

    verbosity_level = package_config.verbosity_level
    if verbosity_level >= 1:
        cmd_parts.append("--verbose")

    # Handle multiple backends
    if "all" not in backend:
        for b in backend:
            if b:
                benchmark_pattern = "GraphBenchmark.track_"
                if collection != "all":
                    benchmark_pattern = f"{benchmark_pattern}.*{collection}"
                benchmark_pattern = f"{benchmark_pattern}.*{b}"
                cmd_parts.extend(["-b", f'"{benchmark_pattern}"'])
    elif collection != "all":
        cmd_parts.extend(["-b", f'"GraphBenchmark.track_.*{collection}"'])

    cmd_parts.append("--python=same")

    cmd = " ".join(cmd_parts)
    logger.info(f"Running command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception(f"Benchmark run failed: {e}")
        raise click.ClickException("Benchmark run failed")


@benchmark.command()
@click.argument("result_file", type=Path)
@click.option("--format", type=click.Choice(["json", "csv"]), default="csv")
@click.pass_context
def export(ctx, result_file: Path, format: str):
    """Export benchmark results."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Using config file for export: {config}")

    dashboard = BenchmarkDashboard(results_dir="results")
    results = dashboard.load_results()

    if not results:
        logger.error("No benchmark results found.")
        click.echo("No benchmark results found.")
        return

    records = []
    for result in results:
        dataset = result.dataset.strip("'")
        backend = result.backend.strip("'")

        algo_name = result.algorithm.split(".")[-1]
        if algo_name.startswith("track_"):
            algo_name = algo_name[6:]

        execution_time = (
            result.execution_time
            if isinstance(result.execution_time, (int, float))
            else float("nan")
        )
        memory_used = (
            result.memory_used
            if isinstance(result.memory_used, (int, float))
            else float("nan")
        )

        record = {
            "algorithm": algo_name,
            "dataset": dataset,
            "backend": backend,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "num_nodes": result.num_nodes,
            "num_edges": result.num_edges,
            "is_directed": result.is_directed,
            "is_weighted": result.is_weighted,
        }
        records.append(record)

        metadata_exclude = [
            "name",
            "directed",
            "weighted",
            "n_nodes",
            "n_edges",
            "download_url",
        ]
        for key, value in result.metadata.items():
            if key in metadata_exclude:
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            if key == "source" and value == "Unknown":
                value = result.metadata.get("download_url", "Unknown")
            record[f"{key}"] = value

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
    click.echo(f"Exported results to {result_file}")


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

    subprocess.run(["asv", "compare", baseline, comparison, "-f", str(threshold)], check=False)


@cli.group()
@click.pass_context
def viz(ctx):
    """Visualization commands."""


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

    # Step 1: Run the results processing script
    try:
        subprocess.run(
            [
                "python",
                "nxbench/validation/scripts/process_results.py",
                "--results_dir",
                "results",
            ],
            check=True,
        )
        logger.info("Successfully processed results.")
    except subprocess.CalledProcessError as e:
        logger.exception(f"Failed to process results: {e}")
        raise click.ClickException("Result processing failed")

    # Step 2: Run asv publish
    subprocess.run(["asv", "publish", "--verbose"], check=False)
    dashboard = BenchmarkDashboard()
    dashboard.generate_static_report()


@cli.group()
@click.pass_context
def validate(ctx):
    """Validation commands."""


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
            logger.exception(f"Validation failed for algorithm '{algorithm_name}': {e}")


def main():
    cli()


if __name__ == "__main__":
    main()
