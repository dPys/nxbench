import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from importlib import resources
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
    """Validate an executable path."""
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
    """
    Safely run a subprocess command with optional output capture.

    Parameters
    ----------
    cmd : Sequence[str | Path]
        The command and arguments to execute.
    check : bool, default=True
        If True, raise an exception if the command fails.
    capture_output : bool, default=False
        If True, capture stdout and stderr.
    **kwargs : dict
        Additional keyword arguments to pass to subprocess.run.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process.

    Raises
    ------
    TypeError
        If a command argument is not of type str or Path.
    ValueError
        If a command argument contains potentially unsafe characters.
    """
    if not cmd:
        raise ValueError("Empty command")

    executable = validate_executable(cmd[0])
    safe_cmd = [str(executable)]

    for arg in cmd[1:]:
        if not isinstance(arg, (str, Path)):
            raise TypeError(f"Command argument must be str or Path, got {type(arg)}")
        if ";" in str(arg) or "&&" in str(arg) or "|" in str(arg):
            raise ValueError(f"Potentially unsafe argument: {arg}")
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


def get_git_hash(repo_path: Path) -> str:
    """Get current git commit hash within the specified repository path."""
    git_path = get_git_executable()
    if git_path is None:
        return "unknown"

    try:
        proc = subprocess.run(  # noqa: S603
            [str(git_path), "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (subprocess.SubprocessError, ValueError):
        return "unknown"


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


def find_project_root() -> Path:
    """Find the project root directory (one containing .git)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return current.parent


def ensure_asv_config_in_root():
    """Ensure asv.conf.json is present at the project root as a symlink."""
    project_root = find_project_root()
    target = project_root / "asv.conf.json"
    if not target.exists():
        with resources.path("nxbench.configs", "asv.conf.json") as config_path:
            target.symlink_to(config_path)
    return project_root


def has_git(project_root):
    return (project_root / ".git").exists()


def run_asv_command(
    args: Sequence[str], check: bool = True, use_commit_hash: bool = True
) -> subprocess.CompletedProcess:
    """Run ASV command with dynamic asv.conf.json based on DVCS presence."""
    asv_path = get_asv_executable()
    if asv_path is None:
        raise click.ClickException("ASV executable not found")

    project_root = find_project_root()
    _has_git = has_git(project_root)
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Has .git: {_has_git}")

    try:
        with resources.open_text("nxbench.configs", "asv.conf.json") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        raise click.ClickException("asv.conf.json not found in package resources.")

    if not _has_git:
        logger.debug(
            "No .git directory found. Modifying asv.conf.json for remote repo and "
            "virtualenv."
        )
        config_data["repo"] = str(project_root.resolve())
        config_data["environment_type"] = "virtualenv"
    else:
        logger.debug("Found .git directory. Using existing repository settings.")

    try:
        import nxbench

        nxbench_path = Path(nxbench.__file__).resolve().parent
        benchmark_dir = nxbench_path / "benchmarks"
        if not benchmark_dir.exists():
            logger.error(f"Benchmark directory not found: {benchmark_dir}")
        config_data["benchmark_dir"] = str(benchmark_dir)
        logger.debug(f"Set benchmark_dir to: {benchmark_dir}")
    except ImportError:
        raise click.ClickException("Failed to import nxbench. Ensure it is installed.")
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    config_data["pythons"] = [str(get_python_executable())]

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_config_path = Path(tmpdir) / "asv.conf.json"
        with temp_config_path.open("w") as f:
            json.dump(config_data, f, indent=4)
        logger.debug(f"Temporary asv.conf.json created at: {temp_config_path}")

        safe_args = []
        for arg in args:
            if not isinstance(arg, str):
                raise click.ClickException(f"Invalid argument type: {type(arg)}")
            if ";" in arg or "&&" in arg or "|" in arg:
                raise click.ClickException(f"Potentially unsafe argument: {arg}")
            safe_args.append(arg)

        if "--config" not in safe_args:
            safe_args = ["--config", str(temp_config_path), *safe_args]
            logger.debug(f"Added --config {temp_config_path} to ASV arguments.")

        if use_commit_hash and _has_git:
            try:
                git_hash = get_git_hash(project_root)
                if git_hash != "unknown":
                    safe_args.append(f"--set-commit-hash={git_hash}")
                    logger.debug(f"Set commit hash to: {git_hash}")
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not determine git commit hash. Proceeding without it."
                )

        old_cwd = Path.cwd()
        if _has_git:
            os.chdir(project_root)
            logger.debug(f"Changed working directory to project root: {project_root}")

        try:
            asv_command = [str(asv_path), *safe_args]
            logger.debug(f"Executing ASV command: {' '.join(map(str, asv_command))}")
            return safe_run(asv_command)
        except subprocess.CalledProcessError:
            logger.exception("ASV command failed.")
            raise click.ClickException("ASV command failed.")
        except (subprocess.SubprocessError, ValueError):
            logger.exception("ASV subprocess error occurred.")
            raise click.ClickException("ASV subprocess error occurred.")
        finally:
            if _has_git:
                os.chdir(old_cwd)
                logger.debug(f"Restored working directory to: {old_cwd}")


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
@click.option(
    "--use-commit-hash/--no-commit-hash",
    default=False,
    help="Whether to use git commit hash for benchmarking.",
)
@click.pass_context
def run_benchmark(ctx, backend: tuple[str], collection: str, use_commit_hash: bool):
    """Run benchmarks."""
    config = ctx.obj.get("CONFIG")
    if config:
        logger.debug(f"Config file used for benchmark run: {config}")

    cmd_args = ["run", "--quick"]

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
        run_asv_command(cmd_args, use_commit_hash=use_commit_hash)
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
    """Export benchmark results."""
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
