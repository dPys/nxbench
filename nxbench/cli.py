import asyncio
import logging
import os
from pathlib import Path

import click
import pandas as pd

from nxbench.benchmarking.benchmark import main_benchmark
from nxbench.benchmarking.config import DatasetConfig
from nxbench.benchmarking.export import ResultsExporter
from nxbench.data.loader import BenchmarkDataManager
from nxbench.data.repository import NetworkRepository
from nxbench.log import _config as package_config
from nxbench.validation.registry import BenchmarkValidator

logger = logging.getLogger("nxbench")


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to config file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="Directory to store benchmark results.",
)
@click.pass_context
def cli(ctx, verbose: int, config: Path | None, output_dir: Path):
    """Top-level CLI group."""
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
        absolute_config = config.resolve()
        os.environ["NXBENCH_CONFIG_FILE"] = str(absolute_config)
        logger.info(f"Using config file: {absolute_config}")

    try:
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Results directory is set to: {results_dir.resolve()}")
    except Exception:
        logger.exception(f"Failed to create results directory '{results_dir}'")
        raise click.ClickException(
            f"Failed to create results directory '{results_dir}'"
        )

    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = config
    ctx.obj["OUTPUT_DIR"] = output_dir.resolve()
    ctx.obj["RESULTS_DIR"] = results_dir.resolve()

    if not ctx.invoked_subcommand:
        click.echo(ctx.get_help())
        ctx.exit(0)


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
        logger.exception("Failed to download dataset")
        raise click.ClickException(f"Failed to download dataset: {e}")


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
@click.pass_context
def run_benchmark(ctx):
    """Run benchmarks using Pydra."""
    config = ctx.obj.get("CONFIG")
    output_dir = ctx.obj.get("OUTPUT_DIR", Path.cwd())
    results_dir = ctx.obj.get("RESULTS_DIR", output_dir / "results")

    if config:
        logger.debug(f"Config file used for benchmark run: {config}")

    try:
        asyncio.run(main_benchmark(results_dir))
    except Exception as e:
        logger.exception("Error during benchmark run")
        raise click.ClickException(f"Benchmark run failed: {e}")


@benchmark.command()
@click.argument("result_file", type=Path, metavar="RESULT_FILE")
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv", "sql"]),
    default="csv",
    help="Format to export results in",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the output file. If not provided, the output file will be derived "
    "from RESULT_FILE.",
)
@click.pass_context
def export(ctx, result_file: Path, output_format: str, output_file: Path | None):
    """Export benchmark results.

    RESULT_FILE is the path to the input benchmark results file (JSON or CSV).
    """
    config = ctx.obj.get("CONFIG")

    if config:
        logger.debug(f"Using config file for export: {config}")

    try:
        exporter = ResultsExporter(results_file=result_file)

        if output_file is None:
            output_file = result_file.with_suffix(f".{output_format}")
            logger.debug(
                f"No output file specified. Using inferred path: {output_file}"
            )

        exporter.export_results(output_path=output_file, form=output_format)

        logger.info(f"Exported results to {output_file}")
        click.echo(f"Exported results to {output_file}")

    except Exception as e:
        logger.exception("Failed to export results")
        click.echo(f"Error exporting results: {e!s}", err=True)
        raise click.Abort


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


@cli.group()
@click.pass_context
def validate(ctx):
    """Validate."""


@validate.command()
@click.argument("result_file", type=Path)
@click.pass_context
def check(ctx, result_file: Path):
    """Validate benchmark results."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for validate check: {config}")

    try:
        df = pd.read_json(result_file)
    except FileNotFoundError as e:
        logger.exception("File not found for validate check")
        raise click.ClickException(f"File {result_file} does not exist")
    except ValueError as e:
        logger.exception("Invalid JSON in results file")
        raise click.ClickException(str(e))

    validator = BenchmarkValidator()

    for _, row in df.iterrows():
        result = row["result"]
        algorithm_name = row["algorithm"]
        graph = None
        try:
            validator.validate_result(result, algorithm_name, graph, raise_errors=True)
            logger.info(f"Validation passed for algorithm '{algorithm_name}'")
        except Exception as e:
            logger.exception(f"Validation failed for algorithm '{algorithm_name}'")
            raise click.ClickException(f"Validation failed for '{algorithm_name}': {e}")


def main():
    cli()


if __name__ == "__main__":
    main()
