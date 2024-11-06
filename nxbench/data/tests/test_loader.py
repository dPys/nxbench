import warnings
import pytest
import networkx as nx
from pathlib import Path
from nxbench.benchmarks.config import DatasetConfig

warnings.filterwarnings("ignore")


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
    assert not graph.is_directed(), "Graph should be undirected"
    assert "directed" in metadata, "Metadata should include 'directed'"
    assert metadata["directed"] is False, "Metadata 'directed' should be False"
    assert "weighted" in metadata, "Metadata should include 'weighted'"
    assert metadata["weighted"] is False, "Metadata 'weighted' should be False"


@pytest.mark.asyncio
async def test_load_jazz(data_manager, create_edge_file):
    """Test loading the 'jazz' dataset."""
    edge_content = """# Jazz musicians collaboration
A B 1.0
B C 2.0
C D 3.0
D A 4.0
"""
    create_edge_file("jazz.edges", edge_content)

    config = DatasetConfig(
        name="jazz",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )
    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 4, "Graph should have 4 edges"
    assert not graph.is_directed(), "Graph should be undirected"
    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert isinstance(data["weight"], float), "Weight should be a float"


@pytest.mark.asyncio
async def test_load_08blocks(data_manager, create_edge_file):
    """Test loading the '08blocks' dataset."""
    edge_content = """# 08blocks graph
1 2 1.5
2 3 2.5
3 4 3.5
4 1 4.5
"""
    create_edge_file("08blocks.edges", edge_content)

    config = DatasetConfig(
        name="08blocks",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )
    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 4, "Graph should have 4 edges"
    assert not graph.is_directed(), "Graph should be undirected"
    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert isinstance(data["weight"], float), "Weight should be a float"


@pytest.mark.asyncio
async def test_load_nr_graph(data_manager, create_edge_file):
    """Test loading various graphs from the network repository."""
    test_cases = [
        {
            "filename": "patentcite.edges",
            "content": """# Patent Citation Network
A B 1.0
B C 2.0
C D 3.0
D A 4.0
""",
            "metadata": {"directed": False, "weighted": True},
            "expected_edges": 4,
            "expect_weights": True,
        },
        {
            "filename": "imdb.edges",
            "content": """# IMDB Collaboration
1 2
2 3
3 4
4 1
""",
            "metadata": {"directed": False, "weighted": False},
            "expected_edges": 4,
            "expect_weights": False,
        },
        {
            "filename": "citeseer.edges",
            "content": """# Citeseer Dataset
X Y 1.0
Y Z 2.0
Z W 3.0
W X 4.0
""",
            "metadata": {"directed": False, "weighted": True},
            "expected_edges": 4,
            "expect_weights": True,
        },
        {
            "filename": "twitter.edges",
            "content": """# Twitter Followers
U V
V W
W X
X U
""",
            "metadata": {"directed": False, "weighted": False},
            "expected_edges": 4,
            "expect_weights": False,
        },
    ]

    for case in test_cases:
        create_edge_file(case["filename"], case["content"])

        config = DatasetConfig(
            name=Path(case["filename"]).stem,
            source="networkrepository",
            params={},
            metadata=case["metadata"],
        )

        graph, metadata = await data_manager.load_network(config)

        assert isinstance(graph, nx.Graph) or isinstance(
            graph, nx.DiGraph
        ), f"Graph should be NetworkX Graph or DiGraph for {case['filename']}"
        assert (
            graph.number_of_nodes() > 0
        ), f"Graph should have nodes for {case['filename']}"
        assert (
            graph.number_of_edges() == case["expected_edges"]
        ), f"Graph should have {case['expected_edges']} edges for {case['filename']}"

        if case["expect_weights"]:
            for u, v, data in graph.edges(data=True):
                assert (
                    "weight" in data
                ), f"Edge ({u}, {v}) should have a 'weight' for {case['filename']}"
                assert isinstance(
                    data["weight"], float
                ), f"Weight should be a float for edge ({u}, {v}) in {case['filename']}"
        else:
            for u, v, data in graph.edges(data=True):
                assert (
                    "weight" not in data
                ), f"Edge ({u}, {v}) should not have a 'weight' for {case['filename']}"


@pytest.mark.asyncio
async def test_load_unweighted_with_comments(data_manager, create_edge_file):
    """Test loading an unweighted edge list with comments and different delimiters."""
    edge_content = """% This is a comment
# Another comment line
A,B
B\tC
C D
D\tE
E,F
"""
    create_edge_file("mixed_delimiters.edges", edge_content)

    config = DatasetConfig(
        name="mixed_delimiters",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": False},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(
        graph, nx.DiGraph
    ), "Graph should be NetworkX Graph or DiGraph"
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 3, "Graph should have 3 edges"

    for u, v, data in graph.edges(data=True):
        assert "weight" not in data, f"Edge ({u}, {v}) should not have a 'weight'"


@pytest.mark.asyncio
async def test_load_weighted_with_invalid_weights(data_manager, create_edge_file):
    """Test loading a weighted edge list where some weights are invalid."""
    edge_content = """# Weighted Edge List with invalid weights
A B 1.0
B C two
C D 3.0
D A four
"""
    create_edge_file("invalid_weights.edges", edge_content)

    config = DatasetConfig(
        name="invalid_weights",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(
        graph, nx.DiGraph
    ), "Graph should be NetworkX Graph or DiGraph"
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 4, "Graph should have 4 edges"

    for u, v, data in graph.edges(data=True):
        assert (
            "weight" not in data
        ), f"Edge ({u}, {v}) should not have a 'weight' due to invalid weights"


@pytest.mark.asyncio
async def test_load_edge_list_with_self_loops_and_duplicates(
    data_manager, create_edge_file
):
    """Test loading an edge list with self-loops and duplicate edges."""
    edge_content = """# Edge list with self-loops and duplicates
A B 1.0
B C 2.0
C D 3.0
D A 4.0
A A 5.0  # Self-loop
B C 2.0  # Duplicate edge
"""
    create_edge_file("self_loops_duplicates.edges", edge_content)

    config = DatasetConfig(
        name="self_loops_duplicates",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(
        graph, nx.DiGraph
    ), "Graph should be NetworkX Graph or DiGraph"
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert (
        graph.number_of_edges() == 4
    ), "Graph should have 4 edges (self-loop removed, duplicate ignored)"

    self_loops = list(nx.selfloop_edges(graph))
    assert len(self_loops) == 0, "Graph should not have self-loops"

    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert isinstance(
            data["weight"], float
        ), f"Weight should be a float for edge ({u}, {v})"


@pytest.mark.asyncio
async def test_load_edge_list_with_non_sequential_ids(data_manager, create_edge_file):
    """Test loading an edge list with non-sequential and non-integer node IDs."""
    edge_content = """# Edge list with non-sequential and non-integer node IDs
node1 node2 1.0
node3 node4 2.0
node5 node6 3.0
node1 node3 4.0
"""
    create_edge_file("non_sequential_ids.edges", edge_content)

    config = DatasetConfig(
        name="non_sequential_ids",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(
        graph, nx.DiGraph
    ), "Graph should be NetworkX Graph or DiGraph"
    assert graph.number_of_nodes() == 6, "Graph should have 6 nodes"
    assert graph.number_of_edges() == 4, "Graph should have 4 edges"

    for node in graph.nodes():
        assert isinstance(node, str), f"Node ID {node} should be a string"

    expected_weights = {
        ("node1", "node2"): 1.0,
        ("node3", "node4"): 2.0,
        ("node5", "node6"): 3.0,
        ("node1", "node3"): 4.0,
    }
    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert data["weight"] == expected_weights.get(
            (u, v), expected_weights.get((v, u))
        ), f"Incorrect weight for edge ({u}, {v})"


@pytest.mark.asyncio
async def test_load_mtx_graph(data_manager, create_edge_file):
    """Test loading a Matrix Market (.mtx) graph."""
    mtx_content = """%%MatrixMarket matrix coordinate real symmetric
% Example Matrix Market file
4 4 6
1 2 1.0
1 3 2.0
2 3 3.0
1 4 4.0
2 4 5.0
3 4 6.0
"""
    create_edge_file("example.mtx", mtx_content)

    config = DatasetConfig(
        name="example",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph), "Graph should be an instance of nx.Graph"
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 6, "Graph should have 6 edges"

    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert isinstance(
            data["weight"], float
        ), f"Weight should be a float for edge ({u}, {v})"


@pytest.mark.asyncio
async def test_load_invalid_mtx_graph(data_manager, create_edge_file):
    """Test loading an invalid Matrix Market (.mtx) graph."""
    mtx_content = """%%MatrixMarket matrix coordinate real symmetric
% Example invalid Matrix Market file
4 4 6
1 2 1.0
1 3 2.0
"""
    create_edge_file("invalid_example.mtx", mtx_content)

    config = DatasetConfig(
        name="invalid_example",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    with pytest.raises(Exception):
        graph, metadata = await data_manager.load_network(config)


@pytest.mark.asyncio
async def test_load_edge_list_with_extra_columns(data_manager, create_edge_file):
    """Test loading an edge list with extra columns."""
    edge_content = """# Edge list with extra columns
A B 1.0 extra_column
B C 2.0 another_extra
C D 3.0
D A 4.0
"""
    create_edge_file("extra_columns.edges", edge_content)

    config = DatasetConfig(
        name="extra_columns",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert isinstance(graph, nx.Graph) or isinstance(
        graph, nx.DiGraph
    ), "Graph should be NetworkX Graph or DiGraph"
    assert graph.number_of_nodes() == 4, "Graph should have 4 nodes"
    assert graph.number_of_edges() == 4, "Graph should have 4 edges"

    expected_weights = {
        ("A", "B"): 1.0,
        ("B", "C"): 2.0,
        ("C", "D"): 3.0,
        ("D", "A"): 4.0,
    }
    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert data["weight"] == expected_weights.get(
            (u, v), expected_weights.get((v, u))
        ), f"Incorrect weight for edge ({u}, {v})"
