import logging
import os
from importlib import import_module

import networkx as nx

from nxbench.backends.core import BackendManager

logger = logging.getLogger("nxbench")

backend_manager = BackendManager()


# ---- NetworkX backend ----
def convert_networkx(original_graph: nx.Graph, num_threads: int):
    return original_graph


def teardown_networkx():
    pass


backend_manager.register_backend(
    name="networkx",
    import_name="networkx",
    conversion_func=convert_networkx,
    teardown_func=teardown_networkx,
)


# ---- Nx-Parallel backend ----
def convert_parallel(original_graph: nx.Graph, num_threads: int):
    nxp = import_module("nx_parallel")
    nx.config.backends.parallel.active = True
    nx.config.backends.parallel.n_jobs = num_threads
    nx.config.backends.parallel.backend = "loky"
    return nxp.ParallelGraph(original_graph)


def teardown_parallel():
    import networkx as nx

    if hasattr(nx.config.backends, "parallel"):
        nx.config.backends.parallel.active = False
        nx.config.backends.parallel.n_jobs = 1


backend_manager.register_backend(
    name="parallel",
    import_name="nx_parallel",
    conversion_func=convert_parallel,
    teardown_func=teardown_parallel,
)


# ---- Nx-CuGraph backend ----
def convert_cugraph(original_graph: nx.Graph, num_threads: int):
    nxc = import_module("nx_cugraph")
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

    edge_attr = "weight" if nx.is_weighted(original_graph) else None
    return nxc.from_networkx(original_graph, edge_attrs=edge_attr)


def teardown_cugraph():
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "False"


backend_manager.register_backend(
    name="cugraph",
    import_name="nx_cugraph",
    conversion_func=convert_cugraph,
    teardown_func=teardown_cugraph,
)


# ---- GraphBLAS backend ----
def convert_graphblas(original_graph: nx.Graph, num_threads: int):
    gb = import_module("graphblas")
    ga = import_module("graphblas_algorithms")

    gb.ss.config["nthreads"] = num_threads
    return ga.Graph.from_networkx(original_graph)


def teardown_graphblas():
    pass


backend_manager.register_backend(
    name="graphblas",
    import_name="graphblas_algorithms",
    conversion_func=convert_graphblas,
    teardown_func=teardown_graphblas,
)
