import pytest
import yaml
import os
from unittest.mock import patch, MagicMock

import networkx as nx

from nxbench.benchmarks.config import (
    AlgorithmConfig,
    DatasetConfig,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkMetrics,
)
from nxbench.benchmarks.utils import (
    configure_benchmarks,
    get_benchmark_config,
    load_default_config,
)


@pytest.fixture(autouse=True)
def reset_benchmark_config():
    import nxbench.benchmarks.config

    original_config = nxbench.benchmarks.utils._BENCHMARK_CONFIG
    nxbench.benchmarks.utils._BENCHMARK_CONFIG = None
    yield
    nxbench.benchmarks.utils._BENCHMARK_CONFIG = original_config


class TestAlgorithmConfig:
    def test_valid_initialization(self):
        with patch("builtins.__import__") as mock_import:
            mock_module = MagicMock()
            mock_func = MagicMock()
            mock_import.return_value = mock_module
            setattr(mock_module, "pagerank", mock_func)

            algo = AlgorithmConfig(
                name="pagerank",
                func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
                params={"alpha": 0.85},
            )

            assert algo.name == "pagerank"
            assert (
                algo.func == "networkx.algorithms.link_analysis.pagerank_alg.pagerank"
            )
            assert algo.params == {"alpha": 0.85}
            assert algo.func_ref == mock_func

    def test_invalid_func_import(self, caplog):
        algo = AlgorithmConfig(
            name="invalid_algo",
            func="nonexistent.module.function",
        )
        assert algo.func_ref is None
        assert "Failed to import function 'nonexistent.module.function'" in caplog.text

    def test_valid_validation_function(self):
        with patch("builtins.__import__") as mock_import:
            mock_module = MagicMock()
            mock_val_func = MagicMock()
            mock_import.return_value = mock_module
            setattr(mock_module, "validate_func", mock_val_func)

            algo = AlgorithmConfig(
                name="pagerank",
                func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
                validate_result="validators.pagerank_validator.validate_func",
            )

            assert algo.validate_ref == mock_val_func

    def test_invalid_validation_function(self, caplog):
        algo = AlgorithmConfig(
            name="pagerank",
            func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
            validate_result="validators.nonexistent.validate",
        )
        assert algo.validate_ref is None
        assert (
            "Failed to import validation function 'validators.nonexistent.validate'"
            in caplog.text
        )


class TestDatasetConfig:
    def test_valid_initialization(self):
        ds = DatasetConfig(
            name="jazz",
            source="networkrepository",
            params={"param1": "value1"},
            metadata={"description": "Jazz musicians network"},
        )
        assert ds.name == "jazz"
        assert ds.source == "networkrepository"
        assert ds.params == {"param1": "value1"}
        assert ds.metadata == {"description": "Jazz musicians network"}


class TestBenchmarkConfig:
    def test_load_from_valid_yaml(self, tmp_path):
        yaml_content = """
algorithms:
  - name: pagerank
    func: networkx.algorithms.link_analysis.pagerank_alg.pagerank
    params:
      alpha: 0.85
    groups:
      - centrality
datasets:
  - name: jazz
    source: networkrepository
machine_info:
  cpu: "Intel i7"
  ram: "16GB"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = BenchmarkConfig.from_yaml(config_file)

        assert len(config.algorithms) == 1
        assert config.algorithms[0].name == "pagerank"
        assert config.algorithms[0].params == {"alpha": 0.85}
        assert config.algorithms[0].groups == ["centrality"]

        assert len(config.datasets) == 1
        assert config.datasets[0].name == "jazz"
        assert config.datasets[0].source == "networkrepository"

        assert config.machine_info == {"cpu": "Intel i7", "ram": "16GB"}

    def test_load_from_nonexistent_yaml(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkConfig.from_yaml("nonexistent_config.yaml")

    def test_load_from_invalid_yaml_structure(self, tmp_path, caplog):
        yaml_content = """
algorithms:
  pagerank:
    func: networkx.algorithms.link_analysis.pagerank_alg.pagerank
datasets:
  - name: jazz
    source: networkrepository
"""
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text(yaml_content)

        config = BenchmarkConfig.from_yaml(config_file)

        assert len(config.algorithms) == 0
        assert "should be a list" in caplog.text

    def test_to_yaml(self, tmp_path):
        config = BenchmarkConfig(
            algorithms=[
                AlgorithmConfig(
                    name="pagerank",
                    func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
                    params={"alpha": 0.85},
                    groups=["centrality"],
                )
            ],
            datasets=[
                DatasetConfig(
                    name="jazz",
                    source="networkrepository",
                )
            ],
            machine_info={"cpu": "Intel i7", "ram": "16GB"},
        )

        output_file = tmp_path / "output_config.yaml"
        config.to_yaml(output_file)

        loaded_data = yaml.safe_load(output_file.read_text())

        assert "algorithms" in loaded_data
        assert len(loaded_data["algorithms"]) == 1
        assert loaded_data["algorithms"][0]["name"] == "pagerank"
        assert loaded_data["algorithms"][0]["params"] == {"alpha": 0.85}
        assert loaded_data["algorithms"][0]["groups"] == ["centrality"]

        assert "datasets" in loaded_data
        assert len(loaded_data["datasets"]) == 1
        assert loaded_data["datasets"][0]["name"] == "jazz"
        assert loaded_data["datasets"][0]["source"] == "networkrepository"

        assert loaded_data["machine_info"] == {"cpu": "Intel i7", "ram": "16GB"}


class TestGlobalConfiguration:
    def test_configure_with_instance(self):
        config = load_default_config()
        configure_benchmarks(config)
        current_config = get_benchmark_config()
        assert current_config == config

    def test_configure_with_yaml(self, tmp_path):
        yaml_content = """
algorithms:
  - name: louvain_communities
    func: networkx.algorithms.community.louvain.louvain_communities
    requires_undirected: true
datasets:
  - name: 08blocks
    source: networkrepository
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        configure_benchmarks(config_file)
        current_config = get_benchmark_config()

        assert len(current_config.algorithms) == 1
        assert current_config.algorithms[0].name == "louvain_communities"
        assert current_config.algorithms[0].requires_undirected is True

        assert len(current_config.datasets) == 1
        assert current_config.datasets[0].name == "08blocks"

    def test_reconfigure_error(self):
        config = load_default_config()
        configure_benchmarks(config)
        with pytest.raises(ValueError, match="Benchmark configuration already set"):
            configure_benchmarks(config)

    def test_get_benchmark_config_with_env(self, tmp_path, monkeypatch):
        yaml_content = """
algorithms:
  - name: pagerank
    func: networkx.algorithms.link_analysis.pagerank_alg.pagerank
datasets:
  - name: jazz
    source: networkrepository
"""
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("NXBENCH_CONFIG_FILE", str(config_file))

        config = get_benchmark_config()
        assert len(config.algorithms) == 1
        assert config.algorithms[0].name == "pagerank"

    def test_get_benchmark_config_env_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NXBENCH_CONFIG_FILE", "nonexistent.yaml")

        with pytest.raises(FileNotFoundError):
            get_benchmark_config()

    def test_get_benchmark_config_load_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_benchmark_config()
            default_config = load_default_config()
            assert config == default_config

        assert len(config.algorithms) == 2
        assert config.algorithms[0].name == "pagerank"
        assert config.algorithms[1].name == "louvain_communities"

        assert len(config.datasets) == 8
        assert config.datasets[0].name == "08blocks"
        assert config.datasets[1].name == "jazz"
        assert config.datasets[2].name == "karate"
        assert config.datasets[3].name == "patentcite"
        assert config.datasets[4].name == "IMDB"
        assert config.datasets[5].name == "citeseer"
        assert config.datasets[6].name == "enron"
        assert config.datasets[7].name == "twitter"


class TestBenchmarkResult:
    def test_from_asv_result_valid(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3)])

        asv_result = {
            "execution_time": 0.123,
            "memory_used": 45.6,
            "dataset": "jazz",
            "backend": "networkx",
            "algorithm": "pagerank",
        }

        result = BenchmarkResult.from_asv_result(asv_result, graph)

        assert result.algorithm == "pagerank"
        assert result.dataset == "jazz"
        assert result.execution_time == 0.123
        assert result.memory_used == 45.6
        assert result.num_nodes == 3
        assert result.num_edges == 2
        assert result.is_directed is False
        assert result.is_weighted is False
        assert result.backend == "networkx"
        assert result.metadata == {}

    def test_from_asv_result_non_numeric(self, caplog):
        graph = nx.Graph()
        graph.add_nodes_from([1])
        graph.add_edge(1, 2)

        asv_result = {
            "execution_time": "fast",
            "memory_used": "low",
            "dataset": "jazz",
            "backend": "networkx",
            "algorithm": "pagerank",
        }

        result = BenchmarkResult.from_asv_result(asv_result, graph)

        assert result.execution_time != "fast"
        assert result.execution_time != 0.0
        assert result.memory_used != "low"
        assert result.memory_used != 0.0
        assert "Non-numeric execution_time" in caplog.text
        assert "Non-numeric memory_used" in caplog.text

    def test_from_asv_result_missing_graph(self):
        asv_result = {
            "execution_time": 0.123,
            "memory_used": 45.6,
            "dataset": "jazz",
            "backend": "networkx",
            "algorithm": "pagerank",
        }

        with pytest.raises(AttributeError):
            BenchmarkResult.from_asv_result(asv_result, None)


class TestBenchmarkMetrics:
    def test_valid_initialization(self):
        metrics = BenchmarkMetrics(
            execution_time=0.123,
            memory_used=45.6,
        )
        assert metrics.execution_time == 0.123
        assert metrics.memory_used == 45.6
