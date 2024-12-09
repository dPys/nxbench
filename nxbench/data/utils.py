import re
import zipfile
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
