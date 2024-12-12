import re
import zipfile
from collections import Counter
from pathlib import Path

import networkx as nx


def normalize_name(name: str) -> str:
    """Normalize the network name for URL construction.
    Preserves the original casing and replaces special characters with hyphens.
    Collapses multiple hyphens into a single hyphen and strips leading/trailing hyphens.
    """
    normalized = re.sub(r"[^a-zA-Z0-9\-]+", "-", name)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = normalized.strip("-")
    return normalized


def get_connected_components(G: nx.Graph) -> list:
    """Retrieve connected components of a graph."""
    if nx.is_directed(G):
        if nx.is_strongly_connected(G):
            return [set(G.nodes())]
        return list(nx.weakly_connected_components(G))
    return list(nx.connected_components(G))


def lcc(G: nx.Graph) -> nx.Graph:
    """Extract the largest connected component (LCC) of the graph.

    Removes self-loops from the extracted subgraph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    nx.Graph
        A subgraph containing the largest connected component without self-loops.
        If the input graph has no nodes, it returns the input graph.
    """
    if G.number_of_nodes() == 0:
        return G

    connected_components = get_connected_components(G)
    largest_cc = max(connected_components, key=len)
    subgraph = G.subgraph(largest_cc).copy()
    subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
    return subgraph


def safe_extract(filepath, extracted_path):
    extracted_path = Path(extracted_path)
    if not extracted_path.exists():
        extracted_path.mkdir(parents=True)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name:
                raise ValueError(f"Malicious path in archive: {name}")
        zf.extractall(extracted_path)


def fix_matrix_market_file(in_path: Path) -> Path:
    if not in_path.exists() or not in_path.is_file():
        raise FileNotFoundError(
            f"Input file '{in_path!s}' does not exist or is not a file."
        )

    with in_path.open("r") as f:
        lines = [line.rstrip("\n") for line in f]

    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("%%MatrixMarket"):
            header_index = i
            break

    if header_index is None:
        raise ValueError("No %%MatrixMarket header line found.")

    header_line = lines[header_index]
    if "coordinate" not in header_line:
        raise ValueError(
            "This fix only applies to coordinate format Matrix Market files."
        )

    symmetric = "symmetric" in header_line.lower()
    content_lines = lines[header_index + 1 :]

    non_comment_lines = [ln for ln in content_lines if ln and not ln.startswith("%")]

    if not non_comment_lines:
        raise ValueError("No dimension or data lines found after header and comments.")

    dimension_line = non_comment_lines[0]
    parts = dimension_line.split()

    out_file_path = in_path.with_name(f"{in_path.stem}_corrected{in_path.suffix}")

    if len(parts) == 3:
        out_file_path.write_text("\n".join(lines) + "\n")
        return out_file_path

    if len(parts) < 2:
        raise ValueError(
            f"Dimension line '{dimension_line}' does not have enough integers."
        )

    data_lines = non_comment_lines[1:]
    if not data_lines:
        raise ValueError("No data lines found; cannot infer NNZ, M, N.")

    # parse data lines to determine M, N, and NNZ
    max_row = 0
    max_col = 0
    NNZ = 0
    for line in data_lines:
        coords = line.split()
        if len(coords) < 2:
            raise ValueError(f"Data line '{line}' does not have two coordinates.")

        r, c = map(int, coords[:2])  # row and col are 1-based
        if r > max_row:
            max_row = r
        if c > max_col:
            max_col = c
        NNZ += 1

    # infer M and N from max indices
    M = max_row
    N = max_col

    # if symmetric and not square, make it square by taking max dimension
    if symmetric and M != N:
        dim = max(M, N)
        M = dim
        N = dim

    # construct corrected dimension line
    corrected_dimension_line = f"{M} {N} {NNZ}"

    # extract comment lines after header and before dimension line:
    after_header = lines[header_index + 1 :]
    dim_line_index_in_after = None
    for idx, val in enumerate(after_header):
        if val.strip() == dimension_line:
            dim_line_index_in_after = idx
            break

    if dim_line_index_in_after is None:
        raise ValueError(
            "Could not locate dimension line in the file after header. File may be "
            "malformed."
        )

    # comment lines before dimension line:
    comment_lines_before_dim = []
    for val in after_header[:dim_line_index_in_after]:
        if val.startswith("%"):
            comment_lines_before_dim.append(val)
        elif not val.strip():
            pass

    with out_file_path.open("w") as out_f:
        for i in range(header_index + 1):
            out_f.write(lines[i] + "\n")

        for cl in comment_lines_before_dim:
            out_f.write(cl + "\n")

        out_f.write(corrected_dimension_line + "\n")

        for dl in data_lines:
            out_f.write(dl + "\n")

    return out_file_path


def detect_delimiter(file_path: Path, sample_size: int = 5) -> str:
    """Detect the most common delimiter in the first few lines of a file."""
    delimiters = [",", "\t", " ", ";"]
    delimiter_counts = Counter()

    with file_path.open("r") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            for delimiter in delimiters:
                if delimiter in line:
                    delimiter_counts[delimiter] += line.count(delimiter)

    if delimiter_counts:
        return delimiter_counts.most_common(1)[0][0]

    raise ValueError("No valid delimiter found in the file.")
