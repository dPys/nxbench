import tempfile
import textwrap
import zipfile
from pathlib import Path

import networkx as nx
import pytest

from nxbench.data.utils import (
    detect_delimiter,
    fix_matrix_market_file,
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
    # Node 1 and 2 form a weakly connected component
    assert len(components) == 1
    assert set(components[0]) == {1, 2}


def test_get_connected_components_directed_graph_not_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])
    components = get_connected_components(G)
    assert len(components) == 2
    assert set(components[0]) == {1, 2, 3}
    assert set(components[1]) == {4, 5}


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
    # Largest connected component is {1,2,3}
    assert lcc_graph.number_of_nodes() == 3
    # Self loop on node 1 is removed
    assert lcc_graph.number_of_edges() == 2
    assert set(lcc_graph.nodes()) == {1, 2, 3}
    # Ensure no self-loop remains
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


def test_lcc_graph_with_isolated_nodes():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    G.add_node(4)  # Isolated node
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.Graph)
    assert lcc_graph.number_of_nodes() == 3
    assert set(lcc_graph.nodes()) == {1, 2, 3}
    assert lcc_graph.number_of_edges() == 2


def test_lcc_directed_graph_strongly_connected():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    lcc_graph = lcc(G)
    assert isinstance(lcc_graph, nx.DiGraph)
    assert lcc_graph.number_of_nodes() == 3
    assert lcc_graph.number_of_edges() == 3
    assert set(lcc_graph.nodes()) == {1, 2, 3}


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

        with zipfile.ZipFile(zip_path, "w"):
            pass

        safe_extract(str(zip_path), str(extracted_path))
        assert extracted_path.exists()
        # No files inside
        assert not any(extracted_path.iterdir())


# =============================
# Tests for fix_matrix_market_file
# =============================


def test_fix_matrix_market_file_file_not_found():
    with pytest.raises(FileNotFoundError):
        fix_matrix_market_file(Path("nonexistent_file.mtx"))


def test_fix_matrix_market_file_no_header(tmp_path):
    file_path = tmp_path / "test_no_header.mtx"
    file_path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="No %%MatrixMarket header"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_non_coordinate(tmp_path):
    file_path = tmp_path / "test_non_coordinate.mtx"
    # We have a header but it's not 'coordinate'
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix array real general
    1 2 3
    """
    )
    file_path.write_text(file_contents)
    with pytest.raises(ValueError, match="only applies to coordinate format"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_no_dimension_line(tmp_path):
    file_path = tmp_path / "test_no_dimension.mtx"
    # Valid coordinate header, but no dimension line after skipping comments
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real general
    % A comment line
    """
    )
    file_path.write_text(file_contents)
    with pytest.raises(ValueError, match="No dimension or data lines found"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_dimension_line_not_enough_integers(tmp_path):
    file_path = tmp_path / "test_dimension_not_enough.mtx"
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real general
    % some comment
    5
    1 1
    """
    )
    file_path.write_text(file_contents)
    with pytest.raises(ValueError, match="does not have enough integers"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_no_data_lines(tmp_path):
    file_path = tmp_path / "test_no_data_lines.mtx"
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real general
    5 5
    """
    )
    file_path.write_text(file_contents)
    with pytest.raises(ValueError, match="No data lines found"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_incomplete_data_line(tmp_path):
    file_path = tmp_path / "test_incomplete_data_line.mtx"
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real general
    5 5
    1
    """
    )
    file_path.write_text(file_contents)
    with pytest.raises(ValueError, match="does not have two coordinates"):
        fix_matrix_market_file(file_path)


def test_fix_matrix_market_file_already_valid(tmp_path):
    """If the dimension line already has 3 integers, we simply rewrite the file."""
    file_path = tmp_path / "test_already_valid.mtx"
    file_contents = textwrap.dedent(
        """\
        %%MatrixMarket matrix coordinate real general
        5 5 2
        1 1 10.0
        2 2 12.0
        """
    )
    file_path.write_text(file_contents)

    out_path = fix_matrix_market_file(file_path)
    assert out_path.exists()
    assert out_path.read_text() == file_contents


def test_fix_matrix_market_file_success_non_symmetric(tmp_path):
    """Dimension line has 2 integers, data lines exist, 'general' => not symmetric."""
    file_path = tmp_path / "test_success_general.mtx"
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real general
    4 5
    1 1 10.0
    2 5 3.14
    3 2 1.00
    4 5 2.72
    """
    )
    file_path.write_text(file_contents)

    out_path = fix_matrix_market_file(file_path)
    corrected = out_path.read_text().splitlines()

    assert corrected[0] == "%%MatrixMarket matrix coordinate real general"
    # Dimension line must have "4 5 4" => M=4, N=5, NNZ=4
    assert corrected[1] == "4 5 4"
    # Data lines
    assert corrected[2] == "1 1 10.0"
    assert corrected[3] == "2 5 3.14"
    assert corrected[4] == "3 2 1.00"
    assert corrected[5] == "4 5 2.72"


def test_fix_matrix_market_file_success_symmetric(tmp_path):
    """Dimension line has 2 integers, data lines exist, 'symmetric' => ensure square
    matrix.
    """
    file_path = tmp_path / "test_success_symmetric.mtx"
    file_contents = textwrap.dedent(
        """\
    %%MatrixMarket matrix coordinate real symmetric
    3 4
    1 1 10.0
    2 4 3.14
    3 2 1.00
    3 4 2.72
    """
    )
    file_path.write_text(file_contents)

    out_path = fix_matrix_market_file(file_path)
    corrected = out_path.read_text().splitlines()

    # Header
    assert corrected[0] == "%%MatrixMarket matrix coordinate real symmetric"
    # M and N should be forced to max(3, 4) = 4
    # NNZ is 4 from the data lines
    assert corrected[1] == "4 4 4"
    # Data lines remain the same
    assert corrected[2] == "1 1 10.0"
    assert corrected[3] == "2 4 3.14"
    assert corrected[4] == "3 2 1.00"
    assert corrected[5] == "3 4 2.72"


# =========================
# Tests for detect_delimiter
# =========================


def test_detect_delimiter_comma(tmp_path):
    file_path = tmp_path / "test_comma.txt"
    file_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == ","


def test_detect_delimiter_tab(tmp_path):
    file_path = tmp_path / "test_tab.txt"
    file_path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == "\t"


def test_detect_delimiter_space(tmp_path):
    file_path = tmp_path / "test_space.txt"
    file_path.write_text("a b c\n1 2 3\n4 5 6\n")
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == " "


def test_detect_delimiter_semicolon(tmp_path):
    file_path = tmp_path / "test_semicolon.txt"
    file_path.write_text("a;b;c\n1;b;3\n4;5;6\n")
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == ";"


def test_detect_delimiter_most_common(tmp_path):
    """When multiple delimiters appear, test which one is more frequent."""
    file_path = tmp_path / "test_most_common.txt"
    # 2 commas per line, 3 spaces per line => total spaces used is 6, commas is 4,
    # so space should win
    file_path.write_text("a b c, d\n1 2 3, 4\nx y z, w\n")
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == " "


def test_detect_delimiter_ignores_comments_and_blank_lines(tmp_path):
    file_path = tmp_path / "test_ignore_comments.txt"
    file_contents = textwrap.dedent(
        """\
    # This is a comment
    % Another comment
    a,b,c
    # Another comment
    1,2,3
    """
    )
    file_path.write_text(file_contents)
    delimiter = detect_delimiter(file_path, sample_size=5)
    assert delimiter == ","


def test_detect_delimiter_no_valid_delimiter(tmp_path):
    file_path = tmp_path / "test_no_valid_delimiter.txt"
    file_path.write_text("# This line is a comment\n% Another comment\n\n---\n")
    with pytest.raises(ValueError, match="No valid delimiter found"):
        detect_delimiter(file_path, sample_size=5)
