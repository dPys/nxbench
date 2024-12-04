import importlib
import logging
import warnings

import networkx as nx

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


def generate_graph(
    generator_name: str, gen_params: dict, directed: bool = False
) -> nx.Graph:
    """Generate a synthetic network using networkx generator functions."""
    if not generator_name:
        raise ValueError("Generator name must be specified.")

    try:
        module_path, func_name = generator_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        generator = getattr(module, func_name)
    except Exception as e:
        raise ValueError(f"Invalid generator {generator_name}") from e

    try:
        graph = generator(**gen_params)
    except Exception as e:
        raise ValueError(
            f"Failed to generate graph with {generator_name} and params {gen_params}"
        ) from e

    if directed and not graph.is_directed():
        graph = graph.to_directed()
    elif not directed and graph.is_directed():
        graph = graph.to_undirected()

    return graph
