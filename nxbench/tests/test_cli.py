import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from nxbench.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_nxbench_logger():
    """
    Automatically runs for each test, capturing and restoring the nxbench logger state.
    This prevents the CLI tests (which can set verbosity=0 and disable logging) from
    causing side-effects that break other tests (e.g. BenchmarkValidator).
    """
    logger = logging.getLogger("nxbench")
    prev_disabled = logger.disabled
    prev_level = logger.level
    yield
    logger.disabled = prev_disabled
    logger.setLevel(prev_level)


def test_cli_no_args(runner):
    result = runner.invoke(cli, [])
    # Suppose your code returns 0 for no subcommand
    assert result.exit_code == 0
    assert "Usage:" in result.output


@patch("nxbench.cli.BenchmarkDataManager.load_network_sync")
def test_data_download_ok(mock_load_sync, runner):
    mock_load_sync.return_value = ("fake_graph", {"meta": "test"})
    args = ["data", "download", "test_dataset", "--category", "my_category"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    mock_load_sync.assert_called_once()


@patch("nxbench.cli.NetworkRepository")
def test_data_list_datasets_ok(mock_repo_cls, runner):
    mock_repo_instance = AsyncMock()
    mock_repo_cls.return_value.__aenter__.return_value = mock_repo_instance
    FakeMeta = MagicMock()
    FakeMeta.__dict__ = {
        "name": "Net1",
        "category": "cat1",
        "nodes": 100,
        "directed": False,
    }
    mock_repo_instance.list_networks.return_value = [FakeMeta]

    args = [
        "data",
        "list-datasets",
        "--category",
        "cat1",
        "--min-nodes",
        "50",
        "--max-nodes",
        "1000",
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Net1" in result.output


@patch("nxbench.cli.main_benchmark", new_callable=AsyncMock)
def test_benchmark_run_ok(mock_main_benchmark, runner):
    mock_main_benchmark.return_value = None
    args = ["benchmark", "run"]
    result = runner.invoke(cli, args, catch_exceptions=True)
    assert result.exit_code == 0, result.output
    mock_main_benchmark.assert_awaited_once()


@patch("nxbench.cli.ResultsExporter")
def test_benchmark_export_ok(mock_exporter_cls, runner):
    mock_exporter = mock_exporter_cls.return_value
    mock_exporter.export_results.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "results.json"
        data = [{"algorithm": "algo", "result": 42}]
        result_file.write_text(json.dumps(data))

        output_file = Path(tmpdir) / "exported.csv"
        args = [
            "benchmark",
            "export",
            str(result_file),
            "--output-format",
            "csv",
            "--output-file",
            str(output_file),
        ]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output

        mock_exporter_cls.assert_called_once_with(results_file=result_file, config={})
        mock_exporter.export_results.assert_called_once_with(
            output_path=output_file, form="csv"
        )


@patch("nxbench.cli.BenchmarkValidator")
def test_validate_check_ok(mock_validator_cls, runner):
    mock_validator = mock_validator_cls.return_value
    mock_validator.validate_result.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "results.json"
        data = [
            {"algorithm": "algo1", "result": 1},
            {"algorithm": "algo2", "result": 2},
        ]
        result_file.write_text(json.dumps(data))

        args = ["validate", "check", str(result_file)]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output

        calls = mock_validator.validate_result.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == 1
        assert calls[0].args[1] == "algo1"
        assert calls[1].args[0] == 2
        assert calls[1].args[1] == "algo2"


@patch("nxbench.viz.app.run_server")
def test_viz_serve_ok(mock_run_server, runner):
    args = ["viz", "serve", "--port", "9999", "--debug"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    mock_run_server.assert_called_once_with(port=9999, debug=True)


def test_data_command_no_subcommand(runner):
    """Test `nxbench data` with no subcommand."""
    result = runner.invoke(cli, ["data"])
    # Typically returns 0 and shows usage
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "download" in result.output
    assert "list-datasets" in result.output


def test_benchmark_command_no_subcommand(runner):
    """Test `nxbench benchmark` with no subcommand."""
    result = runner.invoke(cli, ["benchmark"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output
    assert "export" in result.output


def test_viz_command_no_subcommand(runner):
    """Test `nxbench viz` with no subcommand."""
    result = runner.invoke(cli, ["viz"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "serve" in result.output


def test_validate_command_no_subcommand(runner):
    """Test `nxbench validate` with no subcommand."""
    result = runner.invoke(cli, ["validate"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "check" in result.output


def test_cli_with_verbose_one(runner):
    """Test CLI with one `-v`."""
    result = runner.invoke(cli, ["-v"])
    # The main usage or help text
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_with_non_existent_config(runner):
    """Test CLI with a non-existent config file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_config = Path(tmpdir) / "does_not_exist.yml"
        args = ["--config", str(bad_config)]
        result = runner.invoke(cli, args)
        assert result.exit_code != 0
        assert "does_not_exist.yml" in result.output


@patch("nxbench.cli.os.environ", new_callable=dict)
def test_cli_with_valid_config(mock_environ, runner):
    """Test CLI sets environment variable if config file is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_config = Path(tmpdir) / "config.yml"
        valid_config.write_text("fake_config: true")

        args = ["--config", str(valid_config)]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        # Confirm environment var got set
        assert mock_environ.get("NXBENCH_CONFIG_FILE") == str(valid_config.resolve())


def test_cli_fails_creating_results_dir(runner):
    """
    Test if CLI fails to create the results directory.
    We'll mock `Path.mkdir` to raise an exception, triggering the
    click.ClickException handler.
    """
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = PermissionError("Mocked permission error")
        result = runner.invoke(cli, [])
        assert result.exit_code != 0
        assert "Failed to create results directory" in result.output


@patch("nxbench.cli.BenchmarkDataManager.load_network_sync")
def test_data_download_fails(mock_load_sync, runner):
    """Test `data download` if load_network_sync raises an exception."""
    mock_load_sync.side_effect = RuntimeError("Mocked download error")
    result = runner.invoke(cli, ["data", "download", "bad_dataset"])
    assert result.exit_code != 0
    assert "Failed to download dataset" in result.output


@patch("nxbench.cli.ResultsExporter.export_results")
def test_benchmark_export_fails(mock_export_results, runner):
    """Test `benchmark export` when `export_results` raises an exception."""
    mock_export_results.side_effect = ValueError("Mocked export error")
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "results.json"
        data = [{"algorithm": "algo", "result": 42}]
        result_file.write_text(json.dumps(data))

        result = runner.invoke(cli, ["benchmark", "export", str(result_file)])
        assert result.exit_code != 0
        assert "Error exporting results: Mocked export error" in result.output


def test_benchmark_export_no_such_input_file(runner):
    """Test `benchmark export` when the input file does not exist."""
    result = runner.invoke(cli, ["benchmark", "export", "non_existent.json"])
    assert result.exit_code != 0
    assert (
        "No such file or directory" in result.output
        or "Error exporting results:" in result.output
    )


def test_validate_check_no_such_file(runner):
    """Test `validate check` when the JSON file doesn't exist."""
    result = runner.invoke(cli, ["validate", "check", "non_existent.json"])
    assert result.exit_code != 0
    # This may differ depending on how pandas or your code handles file-not-found
    assert "No such file or directory" in result.output or "Error" in result.output


def test_validate_check_invalid_json(runner):
    """Test `validate check` with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_json = Path(tmpdir) / "results.json"
        bad_json.write_text("invalid-json")

        result = runner.invoke(cli, ["validate", "check", str(bad_json)])
        assert result.exit_code != 0
        assert "Expected object or value" in result.output or "Error" in result.output


@patch("nxbench.cli.BenchmarkValidator.validate_result")
def test_validate_check_fails_validation(mock_validate_result, runner):
    """Test `validate check` when a validator raises an exception."""
    mock_validate_result.side_effect = RuntimeError("Mocked validation error")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "results.json"
        data = [{"algorithm": "algo1", "result": 1}]
        result_file.write_text(json.dumps(data))

        result = runner.invoke(cli, ["validate", "check", str(result_file)])
        assert result.exit_code != 0

        assert "Validation failed for 'algo1': Mocked validation error" in result.output


@patch("nxbench.cli.main_benchmark", new_callable=AsyncMock)
def test_benchmark_run_with_exception(mock_main_benchmark, runner):
    """Test `benchmark run` when `main_benchmark` raises an exception."""
    mock_main_benchmark.side_effect = RuntimeError("Mocked run error")
    result = runner.invoke(cli, ["benchmark", "run"], catch_exceptions=True)
    assert result.exit_code != 0
    assert "Mocked run error" in result.output or "Traceback" in result.output
