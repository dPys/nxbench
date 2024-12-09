import tempfile
import zipfile
from pathlib import Path

import networkx as nx
import pytest

from nxbench.data.utils import (
    get_connected_components,
    lcc,
    normalize_name,
    safe_extract,
)

# ========================
# Tests for normalize_name
# ========================


@pytest.mark.parametrize(
    ("input_name", "expected"),
    [
        ("My Network", "My-Network"),
        ("Network@2024", "Network-2024"),
        ("Network #1!", "Network-1"),
        ("--Special--Characters--", "Special-Characters"),
        ("Multiple___Special***Chars", "Multiple-Special-Chars"),
        ("###", ""),
        ("", ""),
        ("NoChangeNeeded", "NoChangeNeeded"),
        ("Leading and trailing ###", "Leading-and-trailing"),
        ("Mixed CASE and 123 Numbers", "Mixed-CASE-and-123-Numbers"),
    ],
)
def test_normalize_name(input_name, expected):
    assert normalize_name(input_name) == expected


# ==================================
# Tests for get_connected_components
# ==================================


def test_get_connected_components_undirected_single_component():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    components = get_connected_components(G)
    assert len(components) == 1
    assert set(components[0]) == {1, 2, 3}


def test_get_connected_components_undirected_multiple_components():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    components = get_connected_components(G)
    assert len(components) == 2
    assert set(components[0]).issubset({1, 2, 3}) or set(components[0]).issubset({4, 5})
    assert set(components[1]).issubset({1, 2, 3}) or set(components[1]).issubset({4, 5})


def test_get_connected_components_directed_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    components = get_connected_components(G)
    assert len(components) == 1
    assert set(components[0]) == {1, 2, 3}


def test_get_connected_components_directed_not_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    components = get_connected_components(G)
    assert len(components) == 2
    assert set(components[0]) == {1, 2, 3} or set(components[0]) == {4, 5}
    assert set(components[1]) == {1, 2, 3} or set(components[1]) == {4, 5}


def test_get_connected_components_empty_graph():
    G = nx.Graph()
    components = get_connected_components(G)
    assert len(components) == 0


def test_get_connected_components_single_node():
    G = nx.Graph()
    G.add_node(1)
    components = get_connected_components(G)
    assert len(components) == 1
    assert set(components[0]) == {1}


def test_get_connected_components_self_loops():
    G = nx.Graph()
    G.add_edges_from([(1, 1), (2, 2), (3, 3)])
    components = get_connected_components(G)
    assert len(components) == 3
    assert set(components[0]) == {1}
    assert set(components[1]) == {2}
    assert set(components[2]) == {3}


def test_get_connected_components_directed_with_self_loops():
    G = nx.DiGraph()
    G.add_edges_from([(1, 1), (2, 2), (1, 2)])
    components = get_connected_components(G)
    assert len(components) == 1
    assert set(components[0]) == {1, 2}


# =============
# Tests for lcc
# =============


def test_lcc_single_component():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 3
    assert set(lcc_graph.nodes()) == {1, 2, 3}


def test_lcc_multiple_components():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 2
    assert set(lcc_graph.nodes()) == {1, 2, 3}


def test_lcc_directed_graph():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.DiGraph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 2
    assert set(lcc_graph.nodes()) == {1, 2, 3}


def test_lcc_empty_graph():
    G = nx.Graph()
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 0
    assert lcc_graph.number_of_edges() == 0


def test_lcc_single_node():
    G = nx.Graph()
    G.add_node(1)
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 1
    assert lcc_graph.number_of_edges() == 0
    assert set(lcc_graph.nodes()) == {1}


def test_lcc_with_self_loops():
    G = nx.Graph()
    G.add_edges_from([(1, 1), (1, 2), (2, 3), (4, 5)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 2
    assert set(lcc_graph.nodes()) == {1, 2, 3}
    assert list(nx.selfloop_edges(lcc_graph)) == []


def test_lcc_multiple_largest_components():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (3, 4), (5, 6)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 2
    assert lcc_graph.number_of_edges() == 1
    possible_components = [{1, 2}, {3, 4}, {5, 6}]
    assert set(lcc_graph.nodes()) in possible_components


def test_lcc_returns_copy():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    lcc_graph = lcc(G)
    lcc_graph.add_edge(3, 4)
    assert not G.has_edge(3, 4)


# ===============================
# Additional Tests for Robustness
# ===============================


def test_lcc_graph_with_isolated_nodes():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    G.add_node(4)  # Isolated node
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 3
    assert set(lcc_graph.nodes()) == {1, 2, 3}
    assert lcc_graph.number_of_edges() == 2


def test_get_connected_components_directed_graph_not_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])
    components = get_connected_components(G)
    assert len(components) == 2
    assert set(components[0]) == {1, 2, 3}
    assert set(components[1]) == {4, 5}


def test_lcc_directed_graph_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.DiGraph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 3
    assert set(lcc_graph.nodes()) == {1, 2, 3}


def test_normalize_name_all_special_chars():
    input_name = "@#$%^&*()!"
    expected = ""
    assert normalize_name(input_name) == expected


def test_normalize_name_trailing_hyphens():
    input_name = "Network---"
    expected = "Network"
    assert normalize_name(input_name) == expected


def test_normalize_name_leading_hyphens():
    input_name = "---Network"
    expected = "Network"
    assert normalize_name(input_name) == expected


# ========================
# Tests for safe_extract
# ========================


def test_safe_extract_valid_archive():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "valid.zip"
        extracted_path = tmpdir_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "This is a test file.")
            zf.writestr("folder/file2.txt", "This is another test file.")

        safe_extract(str(zip_path), str(extracted_path))
        assert (extracted_path / "file1.txt").exists()
        assert (extracted_path / "folder" / "file2.txt").exists()


def test_safe_extract_malicious_path_absolute():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "malicious_absolute.zip"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/etc/passwd", "This should not be extracted.")

        with pytest.raises(ValueError, match="Malicious path in archive"):
            safe_extract(str(zip_path), str(tmpdir_path))


def test_safe_extract_malicious_path_relative():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "malicious_relative.zip"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../malicious.txt", "This should not be extracted.")

        with pytest.raises(ValueError, match="Malicious path in archive"):
            safe_extract(str(zip_path), str(tmpdir_path))


def test_safe_extract_empty_archive():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "empty.zip"
        extracted_path = tmpdir_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            pass

        safe_extract(str(zip_path), str(extracted_path))
        assert extracted_path.exists()
        assert not any(extracted_path.iterdir())
