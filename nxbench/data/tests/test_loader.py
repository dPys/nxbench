import warnings
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import networkx as nx
import pytest

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

    assert isinstance(graph, (nx.Graph, nx.DiGraph))
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

    assert isinstance(graph, (nx.Graph, nx.DiGraph))
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

        assert isinstance(graph, (nx.Graph, nx.DiGraph)), "Graph should be NetworkX "
        f"Graph or DiGraph for {case['filename']}"
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

    assert isinstance(
        graph, (nx.Graph, nx.DiGraph)
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

    assert isinstance(
        graph, (nx.Graph, nx.DiGraph)
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

    assert isinstance(
        graph, (nx.Graph, nx.DiGraph)
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

    assert isinstance(
        graph, (nx.Graph, nx.DiGraph)
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

    with pytest.raises(ValueError, match="Matrix Market file not in expected format"):
        await data_manager.load_network(config)


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

    assert isinstance(
        graph, (nx.Graph, nx.DiGraph)
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


def test_load_metadata(data_manager):
    """Test that metadata is loaded correctly."""
    metadata = data_manager._metadata_df
    assert not metadata.empty, "Metadata DataFrame should not be empty"
    expected_names = [
        "jazz",
        "08blocks",
        "patentcite",
        "imdb",
        "citeseer",
        "mixed_delimiters",
        "invalid_weights",
        "self_loops_duplicates",
        "non_sequential_ids",
        "example",
        "extra_columns",
        "twitter",
        "invalid_example",
    ]
    assert set(metadata["name"]) == set(
        expected_names
    ), "Metadata names do not match expected names"


def test_get_metadata(data_manager):
    """Test retrieving metadata for a network."""
    metadata = data_manager.get_metadata("jazz")
    assert metadata["name"] == "jazz", "Metadata 'name' should be 'jazz'"
    assert metadata["directed"] is False, "Metadata 'directed' should be False"
    assert metadata["weighted"] is True, "Metadata 'weighted' should be True"


@patch("nxbench.data.loader.zipfile.ZipFile")
@pytest.mark.asyncio
async def test_load_network_retry(mock_zipfile_class, data_manager):
    """Test network loading retry behavior."""
    data_manager.get_metadata = MagicMock(
        return_value={
            "download_url": "http://example.com/test.zip",
            "directed": False,
            "weighted": True,
        }
    )

    config = DatasetConfig(name="test", source="networkrepository", params={})

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content.read = AsyncMock(side_effect=[b"data", b""])

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value.__aenter__.return_value = mock_response

    mock_zipfile = MagicMock()
    mock_zipfile.__enter__.return_value.extractall = MagicMock()
    mock_zipfile_class.return_value = mock_zipfile

    with pytest.raises(FileNotFoundError):
        await data_manager.load_network(config, session=mock_session)

    mock_session.get.assert_called_once_with("http://example.com/test.zip")


def test_load_weighted_graph(data_manager, tmp_path):
    """Test loading weighted graph formats."""
    content = """# Test weighted graph
1 2 1.5
2 3 2.5
3 1 3.5
"""
    test_file = tmp_path / "test.edges"
    test_file.write_text(content)

    graph = data_manager._load_graph_file(
        test_file, {"directed": False, "weighted": True}
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
    for _, _, data in graph.edges(data=True):
        assert "weight" in data
        assert isinstance(data["weight"], float)


def test_normalize_graph(data_manager, tmp_path):
    """Test graph normalization and cleanup."""
    content = """1 1
1 2
2 3"""
    test_file = tmp_path / "test.edges"
    test_file.write_text(content)

    normalized = data_manager._load_graph_file(test_file, {"directed": False})

    assert len(list(nx.selfloop_edges(normalized))) == 0
    assert all(isinstance(n, str) for n in normalized.nodes())
    assert normalized.number_of_nodes() == 3
    assert normalized.number_of_edges() == 2


def test_load_graph_file_unsupported_format(data_manager, tmp_path):
    unsupported_file = tmp_path / "unsupported.xyz"
    unsupported_file.write_text("This is a test file with an unsupported format.")
    with pytest.raises(ValueError, match="Unsupported file format"):
        data_manager._load_graph_file(unsupported_file, {})


def test_get_metadata_missing_network(data_manager):
    with pytest.raises(ValueError, match="Network .* not found in metadata cache"):
        data_manager.get_metadata("nonexistent_network")


@patch(
    "nxbench.data.loader.BenchmarkDataManager._download_file", new_callable=AsyncMock
)
@patch("nxbench.data.loader.zipfile.ZipFile")
@pytest.mark.asyncio
async def test_download_and_extract_network_corrupted_zip(
    mock_zipfile_class, mock_download_file, data_manager
):
    mock_download_file.return_value = None

    mock_zipfile_class.side_effect = zipfile.BadZipFile

    with pytest.raises(zipfile.BadZipFile):
        await data_manager._download_and_extract_network(
            "test_network", "http://example.com/test.zip"
        )


def test_load_empty_edge_file(data_manager, tmp_path):
    empty_file = tmp_path / "empty.edges"
    empty_file.touch()
    with pytest.raises(ValueError, match="contains no valid edges"):
        data_manager._load_graph_file(empty_file, {"directed": False})


@pytest.mark.asyncio
async def test_find_graph_file_no_matching_format(data_manager, tmp_path):
    extracted_folder = tmp_path / "extracted"
    extracted_folder.mkdir()
    (extracted_folder / "unrelated_file.txt").touch()
    graph_file = data_manager._find_graph_file(extracted_folder)
    assert graph_file is None, "No graph file should be found in the folder."


def test_generate_empty_graph(data_manager):
    config = DatasetConfig(
        name="empty_generated_graph",
        source="generator",
        params={"generator": "networkx.empty_graph", "n": 0},
        metadata={"directed": False},
    )
    graph, _ = data_manager._generate_graph(config)
    assert graph.number_of_nodes() == 0, "Generated graph should have no nodes."
    assert graph.number_of_edges() == 0, "Generated graph should have no edges."


@pytest.mark.asyncio
async def test_find_graph_file_deeply_nested(data_manager, tmp_path):
    nested_folder = tmp_path / "level1" / "level2"
    nested_folder.mkdir(parents=True)
    graph_file = nested_folder / "test.edges"
    graph_file.touch()
    found_file = data_manager._find_graph_file(tmp_path)
    assert (
        found_file == graph_file
    ), "The graph file should be discovered in nested directories."
