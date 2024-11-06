from unittest.mock import MagicMock

import networkx as nx
import pytest
import yaml


@pytest.fixture
def simple_graph():
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (1, 3),
        ]
    )
    return G


@pytest.fixture
def directed_graph():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),
            ("A", "C"),
        ]
    )
    return G


@pytest.fixture
def flow_graph():
    G = nx.DiGraph()
    G.add_edge("s", "A", capacity=10)
    G.add_edge("s", "B", capacity=5)
    G.add_edge("A", "B", capacity=15)
    G.add_edge("A", "t", capacity=10)
    G.add_edge("B", "t", capacity=10)
    G.graph["source"] = "s"
    G.graph["sink"] = "t"
    return G


@pytest.fixture
def temporary_yaml_config(tmp_path, mock_validator):
    config_data = {
        "validators": {
            "custom_algo_yaml": {
                "validator": "mock_validator",
                "params": {"param1": 20},
                "expected_type": "dict",
                "required": True,
                "extra_checks": ["check2"],
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_validator():
    return MagicMock()
