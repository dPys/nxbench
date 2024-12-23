import importlib.resources as importlib_resources
import tempfile
import warnings
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import networkx as nx
import pandas as pd
import pytest

from nxbench.benchmarks.config import DatasetConfig
from nxbench.data.loader import BenchmarkDataManager

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
    assert graph.number_of_edges() == 2, "Graph should have 2 edges"

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


def test_load_network_sync(data_manager):
    """Test the synchronous load_network_sync method."""
    with patch.object(
        data_manager,
        "load_network",
        return_value=(nx.Graph(), {"directed": False, "weighted": False}),
    ) as mock_load_network:
        config = DatasetConfig(
            name="sync_test_graph",
            source="generator",
            params={"generator": "networkx.empty_graph"},
            metadata={"directed": False, "weighted": False},
        )
        graph, metadata = data_manager.load_network_sync(config)

        mock_load_network.assert_called_once_with(config)
        assert isinstance(graph, nx.Graph), "Graph should be an instance of nx.Graph"
        assert metadata["directed"] is False, "Metadata 'directed' should be False"
        assert metadata["weighted"] is False, "Metadata 'weighted' should be False"


@pytest.mark.asyncio
async def test_generate_graph_exception(data_manager):
    """Test that exceptions during graph generation are logged and raised."""
    config = DatasetConfig(
        name="exception_graph",
        source="generator",
        params={"generator": "networkx.invalid_generator", "n": 100, "p": 0.1},
        metadata={"directed": False, "weighted": False},
    )

    with patch.object(
        data_manager,
        "get_metadata",
        return_value={"directed": False, "weighted": False},
    ):
        with patch(
            "nxbench.data.loader.generate_graph",
            side_effect=Exception("Generator failed"),
        ) as mock_generate_graph:
            with pytest.raises(Exception, match="Generator failed"):
                await data_manager.load_network(config)

            mock_generate_graph.assert_called_once_with(
                "networkx.invalid_generator", {"n": 100, "p": 0.1}, False
            )


def test_generate_graph_missing_generator_name(data_manager):
    """Test that a ValueError is raised when generator_name is missing."""
    config = DatasetConfig(
        name="missing_generator",
        source="generator",
        params={},
        metadata={"directed": False, "weighted": False},
    )

    with pytest.raises(ValueError, match="Generator name must be specified in params."):
        data_manager._generate_graph(config)


@pytest.mark.asyncio
async def test_load_local_graph_success(data_manager, create_edge_file, tmp_path):
    """Test loading a local graph successfully."""
    local_path = tmp_path / "local_test.edges"
    edge_content = """A B 1.0
    B C 2.0
    C A 3.0
    """
    local_path.write_text(edge_content)

    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame([{"name": "local_test", "directed": False, "weighted": True}]),
        ],
        ignore_index=True,
    )

    config = DatasetConfig(
        name="local_test",
        source="local",
        params={"path": str(local_path)},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = data_manager._load_local_graph(config)

    assert isinstance(graph, nx.Graph), "Graph should be an instance of nx.Graph"
    assert graph.number_of_nodes() == 3, "Graph should have 3 nodes"
    assert graph.number_of_edges() == 3, "Graph should have 3 edges"
    for u, v, data in graph.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) should have a 'weight'"
        assert isinstance(data["weight"], float), "Weight should be a float"


def test_load_local_graph_file_not_found(data_manager):
    """Test that FileNotFoundError is raised when local graph file is missing."""
    config = DatasetConfig(
        name="missing_local_graph",
        source="local",
        params={"path": "nonexistent_path.edges"},
        metadata={"directed": False, "weighted": False},
    )

    with pytest.raises(
        FileNotFoundError, match="Network file not found in any location"
    ):
        data_manager._load_local_graph(config)


def test_find_graph_file_supported(data_manager, tmp_path):
    """Test that _find_graph_file finds a supported graph file."""
    supported_file = tmp_path / "supported_test.graphml"
    supported_file.touch()

    found_file = data_manager._find_graph_file(tmp_path)

    assert found_file == supported_file, "Should find the supported graph file."


def test_find_graph_file_no_supported(data_manager, tmp_path):
    """Test that _find_graph_file returns None when no supported graph files are
    present.
    """
    unrelated_file = tmp_path / "unrelated.txt"
    unrelated_file.touch()

    found_file = data_manager._find_graph_file(tmp_path)

    assert (
        found_file is None
    ), "Should return None when no supported graph file is found."


@patch("aiohttp.ClientSession.get")
@pytest.mark.asyncio
async def test_download_file_failure(mock_get, data_manager):
    """Test that a ConnectionError is raised when download fails with non-200 status."""
    mock_response = MagicMock()
    mock_response.status = 404
    mock_get.return_value.__aenter__.return_value = mock_response

    url = "http://example.com/failing_download.zip"
    dest = data_manager.data_dir / "failing_download.zip"

    with pytest.raises(ConnectionError, match="Failed to download file from"):
        await data_manager._download_file(url, dest)

    mock_get.assert_called_once_with(url)


@patch("zipfile.ZipFile.extractall")
@patch("zipfile.ZipFile")
@pytest.mark.asyncio
async def test_download_and_extract_network_no_graph_file(
    mock_zipfile_class, mock_extractall, data_manager, create_edge_file, tmp_path
):
    """Test that FileNotFoundError is raised when no graph file is found after
    extraction.
    """
    mock_zip = MagicMock()
    mock_zipfile_class.return_value.__enter__.return_value = mock_zip
    mock_zip.extractall = MagicMock()

    with patch.object(data_manager, "_find_graph_file", return_value=None):
        with patch.object(data_manager, "_download_file", return_value=None):
            with pytest.raises(FileNotFoundError, match=r"No such file or directory"):
                await data_manager._download_and_extract_network(
                    "no_graph", "http://example.com/no_graph.zip"
                )


@patch("zipfile.ZipFile.extractall")
@patch("zipfile.ZipFile")
@patch("pathlib.Path.rename", side_effect=Exception("Rename failed"))
@pytest.mark.asyncio
async def test_download_and_extract_network_rename_failure(
    mock_rename, mock_zipfile_class, mock_extractall, data_manager, tmp_path
):
    """Test that an exception is raised when renaming the extracted graph file fails."""
    mock_zip = MagicMock()
    mock_zipfile_class.return_value.__enter__.return_value = mock_zip
    mock_zip.extractall = MagicMock()

    extracted_graph_file = tmp_path / "test.edges"

    with patch.object(
        data_manager, "_find_graph_file", return_value=extracted_graph_file
    ):
        with patch.object(data_manager, "_download_file", return_value=None):
            with pytest.raises(Exception, match="Rename failed"):
                await data_manager._download_and_extract_network(
                    "rename_failure", "http://example.com/rename_failure.zip"
                )

            mock_rename.assert_called_once_with(
                data_manager.data_dir / extracted_graph_file.name
            )

    mock_zipfile_class.assert_called_once_with(
        data_manager.data_dir / "rename_failure.zip", "r"
    )
    mock_zip.extractall.assert_called_once_with(
        data_manager.data_dir / "rename_failure_extracted"
    )


@patch(
    "nxbench.data.loader.BenchmarkDataManager._download_file", new_callable=AsyncMock
)
@patch("nxbench.data.loader.BenchmarkDataManager._find_graph_file", return_value=None)
@patch("zipfile.ZipFile")
@pytest.mark.asyncio
async def test_download_and_extract_network_no_suitable_file(
    mock_zipfile_class, mock_find_graph_file, mock_download_file, data_manager
):
    """Test that FileNotFoundError is raised when no suitable graph file exists after
    download.
    """
    mock_zip = MagicMock()
    mock_zipfile_class.return_value.__enter__.return_value = mock_zip
    mock_zip.extractall = MagicMock()

    config = DatasetConfig(
        name="download_no_suitable_file",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": False},
    )

    with patch.object(
        data_manager,
        "get_metadata",
        return_value={
            "download_url": "http://example.com/no_suitable.zip",
            "directed": False,
            "weighted": False,
        },
    ):
        with pytest.raises(
            FileNotFoundError, match=r"\[Errno 2\] No such file or directory"
        ):
            await data_manager.load_network(config)

    mock_download_file.assert_awaited_once_with(
        "http://example.com/no_suitable.zip",
        data_manager.data_dir / "download_no_suitable_file.zip",
        None,
    )
    mock_find_graph_file.assert_called_once_with(
        data_manager.data_dir / "download_no_suitable_file_extracted"
    )


@pytest.mark.asyncio
async def test_load_nr_graph_missing_download_url(data_manager):
    """Test that ValueError is raised when download URL is missing in metadata."""
    config = DatasetConfig(
        name="missing_url_graph",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": False},  # No 'download_url'
    )

    with pytest.raises(
        ValueError, match="No download URL found for network missing_url_graph"
    ):
        await data_manager._load_nr_graph(
            "missing_url_graph", {"directed": False, "weighted": False}
        )


def test_load_graph_file_node_id_conversion(data_manager, create_edge_file):
    """Test that node IDs are converted to strings if they are not."""
    edge_content = """1 2 1.0
    2 3 2.0
    3 4 3.0
    """
    create_edge_file("numeric_nodes.edges", edge_content)

    with patch.object(
        data_manager,
        "get_metadata",
        return_value={"directed": False, "weighted": True, "name": "numeric_nodes"},
    ):
        config = DatasetConfig(
            name="numeric_nodes",
            source="networkrepository",
            params={},
            metadata={"directed": False, "weighted": True},
        )

        graph, metadata = data_manager.load_network_sync(config)

        assert all(
            isinstance(node, str) for node in graph.nodes()
        ), "All node IDs should be strings."


@pytest.mark.asyncio
async def test_load_graphml_file(data_manager, tmp_path):
    """Test loading a .graphml graph file."""
    graphml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <graph id="G" edgedefault="undirected">
        <node id="n0"/>
        <node id="n1"/>
        <node id="n2"/>
        <edge id="e0" source="n0" target="n1" />
        <edge id="e1" source="n1" target="n2" />
        <edge id="e2" source="n2" target="n0" />
      </graph>
    </graphml>
    """
    graphml_file = tmp_path / "test_graph.graphml"
    graphml_file.write_text(graphml_content)

    with patch.object(data_manager, "_find_graph_file", return_value=graphml_file):
        graph = data_manager._load_graph_file(graphml_file, {"directed": False})

    assert isinstance(graph, nx.Graph), "Graph should be an instance of nx.Graph"
    assert graph.number_of_nodes() == 3, "Graph should have 3 nodes"
    assert graph.number_of_edges() == 3, "Graph should have 3 edges"


@pytest.mark.asyncio
async def test_load_edge_list_parsing_exception(data_manager, create_edge_file):
    """Test that an exception during edge list parsing is logged and raised."""
    edge_content = """A B 1.0
    B C invalid_weight
    C D 3.0
    """
    create_edge_file("parsing_exception.edges", edge_content)

    config = DatasetConfig(
        name="parsing_exception",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    with patch.object(
        data_manager, "_load_graph_file", side_effect=ValueError("Parsing failed")
    ):
        with pytest.raises(
            ValueError, match="Network parsing_exception not found in metadata cache"
        ):
            await data_manager.load_network(config)


@pytest.mark.asyncio
async def test_load_mtx_exception(data_manager, create_edge_file):
    """Test that an exception during .mtx file loading is logged and raised."""
    mtx_content = """%%MatrixMarket matrix coordinate real symmetric
    % Incomplete Matrix Market file
    4 4 6
    1 2 1.0
    """
    create_edge_file("malformed.mtx", mtx_content)

    config = DatasetConfig(
        name="malformed",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    with pytest.raises(
        ValueError, match="Network malformed not found in metadata cache"
    ):
        await data_manager.load_network(config)


def test_load_network_invalid_source(data_manager):
    """Test that ValueError is raised for an invalid network source."""
    config = DatasetConfig(
        name="invalid_source_graph",
        source="invalidsource",
        params={},
        metadata={"directed": False, "weighted": False},
    )

    with pytest.raises(
        ValueError, match="Network invalid_source_graph not found in metadata cache"
    ):
        data_manager.load_network_sync(config)


def test_load_metadata_exception():
    """Test that RuntimeError is raised when metadata loading fails."""
    with patch.object(
        BenchmarkDataManager,
        "_load_metadata",
        side_effect=RuntimeError("Metadata load failed"),
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(RuntimeError, match="Metadata load failed"):
                BenchmarkDataManager(data_dir=temp_dir)


@pytest.mark.asyncio
async def test_no_suitable_file_found_after_extract(data_manager, tmp_path):
    data_manager.data_dir = tmp_path
    extracted_folder = data_manager.data_dir / "missing_graph_extracted"
    extracted_folder.mkdir(exist_ok=True)

    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [
                    {
                        "name": "missing_graph",
                        "directed": False,
                        "weighted": False,
                        "download_url": "http://example.com/missing.zip",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with patch.object(data_manager, "_download_file", return_value=None):
        with patch("zipfile.ZipFile") as mock_zip_class:
            mock_zip = MagicMock()
            mock_zip.extractall = MagicMock()
            mock_zip_class.return_value.__enter__.return_value = mock_zip

            data_manager._find_graph_file = MagicMock(return_value=None)

            with pytest.raises(
                FileNotFoundError, match="No suitable graph file found after extracting"
            ):
                await data_manager._download_and_extract_network(
                    "missing_graph", "http://example.com/missing.zip"
                )


@pytest.mark.asyncio
async def test_load_nr_graph_no_suitable_file_found(data_manager):
    """Test that FileNotFoundError is raised if no suitable file is found after
    downloading.
    """
    with patch.object(data_manager, "_download_and_extract_network", return_value=None):
        with patch.object(
            data_manager,
            "_load_graph_file",
            side_effect=FileNotFoundError("Mocked no file"),
        ):
            data_manager.get_metadata = MagicMock(
                return_value={
                    "download_url": "http://example.com/test.zip",
                    "directed": False,
                }
            )
            config = DatasetConfig(
                name="not_found_after_download", source="networkrepository", params={}
            )
            with pytest.raises(
                FileNotFoundError,
                match="No suitable graph file found after downloading",
            ):
                await data_manager.load_network(config)


@pytest.mark.asyncio
async def test_convert_numeric_nodes_to_strings_in_mtx(
    data_manager, create_edge_file, tmp_path
):
    data_manager.data_dir = tmp_path

    mtx_content = """%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
2 3 2.0
3 4 3.0
"""
    local_mtx_path = data_manager.data_dir / "numeric_nodes_mtx.mtx"
    local_mtx_path.write_text(mtx_content)

    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [{"name": "numeric_nodes_mtx", "directed": False, "weighted": True}]
            ),
        ],
        ignore_index=True,
    )

    config = DatasetConfig(
        name="numeric_nodes_mtx",
        source="local",
        params={"path": str(local_mtx_path)},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert all(isinstance(node, (str, int)) for node in graph.nodes())


@pytest.mark.asyncio
async def test_load_unweighted_edgelist_failure(data_manager, create_edge_file):
    edge_content = """A B
B C
"""
    create_edge_file("unweighted_failure.edges", edge_content)

    # Add to metadata so we don't fail with "not found in cache"
    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [
                    {
                        "name": "unweighted_failure",
                        "directed": False,
                        "weighted": False,
                        "download_url": "http://example.com/unweighted_failure.edges",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with patch(
        "networkx.read_edgelist", side_effect=Exception("Unweighted parse failed")
    ):
        config = DatasetConfig(
            name="unweighted_failure",
            source="networkrepository",
            params={},
            metadata={"directed": False, "weighted": False},
        )

        with pytest.raises(Exception, match="Unweighted parse failed"):
            await data_manager.load_network(config)


@pytest.mark.asyncio
async def test_load_edges_unexpected_error_parsing_weights(
    data_manager, create_edge_file
):
    edge_content = """A B 1.0
B C not_a_float
C D 3.0
"""
    create_edge_file("unexpected_error.edges", edge_content)

    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [
                    {
                        "name": "unexpected_error",
                        "directed": False,
                        "weighted": True,
                        "download_url": "http://example.com/unexpected_error.edges",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with patch("builtins.open", side_effect=TypeError("Unexpected error")):
        config = DatasetConfig(
            name="unexpected_error",
            source="networkrepository",
            params={},
            metadata={"directed": False, "weighted": True},
        )
        graph, metadata = await data_manager.load_network(config)

    assert (
        graph.number_of_nodes() > 0
    ), "We should still get nodes from the fallback parse"


@pytest.mark.asyncio
async def test_load_mtx_with_corrected_file_exists(data_manager, tmp_path):
    """Test loading a Matrix Market file when a corrected version already exists."""
    corrected_mtx_file = tmp_path / "example_corrected.mtx"
    corrected_mtx_content = """%%MatrixMarket matrix coordinate real general
3 3 2
1 2 1.0
2 3 2.0
"""
    corrected_mtx_file.write_text(corrected_mtx_content)

    original_mtx_file = tmp_path / "example.mtx"
    original_mtx_file.write_text("This file is intentionally incorrect or not used")

    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [{"name": "example_corrected", "directed": False, "weighted": True}]
            ),
        ],
        ignore_index=True,
    )

    corrected_mtx_file.rename(data_manager.data_dir / "example_corrected_corrected.mtx")
    original_mtx_file.rename(data_manager.data_dir / "example_corrected.mtx")

    config = DatasetConfig(
        name="example_corrected",
        source="networkrepository",
        params={},
        metadata={"directed": False, "weighted": True},
    )

    graph, metadata = await data_manager.load_network(config)

    assert graph.number_of_nodes() == 3, "Should load from the corrected file"
    assert graph.number_of_edges() == 2
    assert "example_corrected" in data_manager._network_cache, "Should cache the graph"


@pytest.mark.asyncio
async def test_load_network_from_cache(data_manager):
    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [
                    {
                        "name": "cache_test_graph",
                        "directed": False,
                        "weighted": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    config = DatasetConfig(
        name="cache_test_graph",
        source="generator",
        params={"generator": "networkx.empty_graph", "n": 5},
        metadata={"directed": False, "weighted": False},
    )

    graph1, metadata1 = await data_manager.load_network(config)

    graph2, metadata2 = await data_manager.load_network(config)

    assert nx.is_isomorphic(
        graph1, graph2
    ), "They should have same structure, even if not the same object"
    assert metadata1 == metadata2, "Metadata should match for cached graph"


def test_load_network_invalid_source_expanded(data_manager):
    data_manager._metadata_df = pd.concat(
        [
            data_manager._metadata_df,
            pd.DataFrame(
                [
                    {
                        "name": "invalid_source_expanded",
                        "directed": False,
                        "weighted": False,
                        "download_url": "http://example.com/not_used",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    config = DatasetConfig(
        name="invalid_source_expanded",
        source="doesnotexist",
        params={},
        metadata={"directed": False, "weighted": False},
    )

    with pytest.raises(ValueError, match="Invalid network source: doesnotexist"):
        data_manager.load_network_sync(config)


def test_load_metadata_failure():
    """Test that a RuntimeError is raised when loading metadata fails."""
    with patch.object(
        importlib_resources, "open_text", side_effect=Exception("Mocked exception")
    ):
        with pytest.raises(RuntimeError, match="Failed to load network metadata"):
            _ = BenchmarkDataManager()
