import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import click
import pandas as pd

from nxbench.benchmarks.config import DatasetConfig
from nxbench.data.loader import BenchmarkDataManager
from nxbench.data.repository import NetworkRepository
from nxbench.log import _config as package_config
from nxbench.viz.dashboard import BenchmarkDashboard

logger = logging.getLogger("nxbench")


def validate_executable(path: str | Path) -> Path:
    """Validate an executable path.

    Parameters
    ----------
    path : str or Path
        Path to executable to validate

    Returns
    -------
    Path
        Validated executable path

    Raises
    ------
    ValueError
        If path is not a valid executable
    """
    executable = Path(path).resolve()
    if not executable.exists():
        raise ValueError(f"Executable not found: {executable}")
    if not os.access(executable, os.X_OK):
        raise ValueError(f"Path is not executable: {executable}")
    return executable


def safe_run(
    cmd: Sequence[str | Path],
    check: bool = True,
    capture_output: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Safely run a subprocess command with optional output capture.

    Parameters
    ----------
    cmd : sequence of str or Path
        Command and arguments to run. First item must be path to executable.
    check : bool, default=True
        Whether to check return code
    capture_output : bool, default=False
        Whether to capture stdout and stderr
    **kwargs : dict
        Additional arguments to subprocess.run

    Returns
    -------
    subprocess.CompletedProcess
        Completed process info

    Raises
    ------
    ValueError
        If command is empty
    TypeError
        If command contains invalid argument types
    subprocess.SubprocessError
        If command fails and check=True
    """
    if not cmd:
        raise ValueError("Empty command")

    executable = validate_executable(cmd[0])
    safe_cmd = [str(executable)]

    for arg in cmd[1:]:
        if not isinstance(arg, (str, Path)):
            raise TypeError(f"Command argument must be str or Path, got {type(arg)}")
        safe_cmd.append(str(arg))

    return subprocess.run(  # noqa: S603
        safe_cmd,
        capture_output=capture_output,
        text=True,
        shell=False,
        check=check,
        **kwargs,
    )


def get_git_executable() -> Path | None:
    """Get full path to git executable."""
    git_path = shutil.which("git")
    if git_path is None:
        return None
    try:
        return validate_executable(git_path)
    except ValueError:
        return None


def get_asv_executable() -> Path | None:
    """Get full path to asv executable."""
    asv_path = shutil.which("asv")
    if asv_path is None:
        return None
    try:
        return validate_executable(asv_path)
    except ValueError:
        return None


def get_python_executable() -> Path:
    """Get full path to Python executable."""
    return validate_executable(sys.executable)


def get_git_hash() -> str:
    """Get current git commit hash."""
    git_path = get_git_executable()
    if git_path is None:
        return "unknown"

    try:
        proc = safe_run([git_path, "rev-parse", "HEAD"], capture_output=True)
        return proc.stdout.strip()
    except (subprocess.SubprocessError, ValueError):
        return "unknown"


def run_asv_command(
    args: Sequence[str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run ASV command with security checks.

    Parameters
    ----------
    args : sequence of str
        Command arguments
    check : bool, default=True
        Whether to check return code

    Returns
    -------
    subprocess.CompletedProcess
        Completed process info

    Raises
    ------
    click.ClickException
        If command fails
    """
    asv_path = get_asv_executable()
    if asv_path is None:
        raise click.ClickException("ASV executable not found")

    safe_args = []
    for arg in args:
        if not isinstance(arg, str):
            raise click.ClickException(f"Invalid argument type: {type(arg)}")
        safe_args.append(arg)

    try:
        return safe_run([asv_path, *safe_args], check=check)
    except (subprocess.SubprocessError, ValueError) as e:
        raise click.ClickException(str(e))


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
    except Exception:
        logger.exception("Failed to download dataset")


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
        git_hash = get_git_hash()
    except subprocess.CalledProcessError:
        logger.exception("Failed to get git hash")
        raise click.ClickException("Could not determine git commit hash")

    cmd_args = ["run", "--quick", f"--set-commit-hash={git_hash}"]

    if package_config.verbosity_level >= 1:
        cmd_args.append("--verbose")

    if "all" not in backend:
        for b in backend:
            if b:
                benchmark_pattern = "GraphBenchmark.track_"
                if collection != "all":
                    benchmark_pattern = f"{benchmark_pattern}.*{collection}"
                benchmark_pattern = f"{benchmark_pattern}.*{b}"
                cmd_args.extend(["-b", benchmark_pattern])
    elif collection != "all":
        cmd_args.extend(["-b", f"GraphBenchmark.track_.*{collection}"])

    cmd_args.append("--python=same")

    try:
        run_asv_command(cmd_args)
    except subprocess.CalledProcessError:
        logger.exception("Benchmark run failed")
        raise click.ClickException("Benchmark run failed")


@benchmark.command()
@click.argument("result_file", type=Path)
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv", "sql"]),
    default="csv",
    help="Format to export results in",
)
@click.pass_context
def export(ctx, result_file: Path, output_format: str):
    """Export benchmark results.

    Parameters
    ----------
    result_file : Path
        Output file path for results
    output_format : str
        Format to export results in (json, csv, or sql)
    """
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Using config file for export: {config}")

    dashboard = BenchmarkDashboard(results_dir="results")

    try:
        if output_format == "sql":
            dashboard.export_results(format="sql", output_path=result_file)
        else:
            df = dashboard.get_results_df()

            if df.empty:
                logger.error("No benchmark results found.")
                click.echo("No benchmark results found.")
                return

            df = df.sort_values(["algorithm", "dataset", "backend"])

            df["execution_time"] = df["execution_time"].map("{:.6f}".format)
            df["memory_used"] = df["memory_used"].map("{:.2f}".format)

            if output_format == "csv":
                df.to_csv(result_file, index=False)
            else:
                df.to_json(result_file, orient="records")

        logger.info(f"Exported results to {result_file}")
        click.echo(f"Exported results to {result_file}")

    except Exception as e:
        logger.exception("Failed to export results")
        click.echo(f"Error exporting results: {e!s}", err=True)
        raise click.Abort


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

    cmd_args = [
        "compare",
        baseline,
        comparison,
        "-f",
        str(threshold),
    ]
    run_asv_command(cmd_args, check=False)


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

    try:
        python_path = get_python_executable()
    except ValueError as e:
        raise click.ClickException(str(e))

    process_script = Path("nxbench/validation/scripts/process_results.py").resolve()
    if not process_script.exists():
        raise click.ClickException(f"Processing script not found: {process_script}")

    try:
        process_script.relative_to(Path.cwd())
    except ValueError:
        raise click.ClickException("Script path must be within project directory")

    try:
        safe_run([python_path, process_script, "--results_dir", "results"])
        logger.info("Successfully processed results.")
    except (subprocess.SubprocessError, ValueError) as e:
        logger.exception("Failed to process results")
        raise click.ClickException(str(e))

    run_asv_command(["publish", "--verbose"], check=False)
    dashboard = BenchmarkDashboard()
    dashboard.generate_static_report()


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
        except Exception:
            logger.exception(f"Validation failed for algorithm '{algorithm_name}'")


def main():
    cli()


if __name__ == "__main__":
    main()
