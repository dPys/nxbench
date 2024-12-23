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

        mock_exporter_cls.assert_called_once_with(results_file=result_file)
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
