import pytest
import tempfile
import warnings
import networkx as nx
from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import DatasetConfig

warnings.filterwarnings("ignore")


@pytest.fixture
def data_manager():
    """Fixture for initializing BenchmarkDataManager."""
    return BenchmarkDataManager()


def test_generate_graph(data_manager):
    """Test generating a synthetic graph."""
    config = DatasetConfig(
        name="test_generated_graph",
        source="generator",
        params={"generator": "networkx.erdos_renyi_graph", "n": 100, "p": 0.1},
        metadata={"directed": False, "weighted": False},
    )

    graph, metadata = data_manager._generate_graph(config)
    assert isinstance(
        graph, nx.Graph
    ), "Generated graph should be an instance of nx.Graph"
    assert graph.number_of_nodes() == 100, "Graph should have 100 nodes"
    assert graph.number_of_edges() > 0, "Graph should have edges"


@pytest.mark.asyncio
async def test_load_jazz(data_manager):
    config = DatasetConfig(
        name="jazz", source="networkrepository", params={}, metadata=None
    )
    graph, metadata = await data_manager.load_network(config)
    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


@pytest.mark.asyncio
async def test_load_08blocks(data_manager):
    config = DatasetConfig(
        name="08blocks", source="networkrepository", params={}, metadata=None
    )
    graph, metadata = await data_manager.load_network(config)
    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


@pytest.mark.asyncio
async def test_load_networkrepository_graph():
    """Test loading a graph from the network repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = BenchmarkDataManager(data_dir=temp_dir)

        config = DatasetConfig(
            name="08blocks",
            source="networkrepository",
            params={},
            metadata={"directed": False, "weighted": True},
        )

        try:
            graph, metadata = await data_manager.load_network(config)
            assert isinstance(
                graph, nx.Graph
            ), "Repository graph should be an instance of nx.Graph"
            assert graph.number_of_nodes() > 0, "Graph should have nodes"
            assert graph.number_of_edges() > 0, "Graph should have edges"
            assert "name" in metadata, "Metadata should contain 'name'"
        except FileNotFoundError:
            pytest.skip("Network repository file not found; skipping test")
        except Exception as e:
            pytest.fail(f"Failed to load network from repository: {e}")
