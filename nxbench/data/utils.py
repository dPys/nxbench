import re
import networkx as nx


def normalize_name(name: str) -> str:
    """Normalize the network name for URL construction.
    Preserves the original casing and replaces special characters with hyphens.
    """
    normalized = re.sub(r"[^a-zA-Z0-9\-]+", "-", name)
    normalized = normalized.strip("-")
    return normalized


def get_connected_components(G: nx.Graph) -> list:
    if nx.is_directed(G):
        if nx.is_strongly_connected(G):
            return [G.nodes()]
        return nx.weakly_connected_components(G)
    return nx.connected_components(G)


def lcc(G: nx.Graph) -> nx.Graph:
    """Extract the largest connected component (LCC) of the graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    nx.Graph
        A subgraph containing the largest connected component. If the input graph
        has no nodes, it returns the input graph.
    """
    if G.number_of_nodes() == 0:
        return G

    connected_components = get_connected_components(G)
    largest_cc = max(connected_components, key=len)
    return G.subgraph(largest_cc).copy()
