"""Validation functions for graph algorithm benchmarking results."""

import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


__all__ = [
    "ValidationError",
    "validate_graph_result",
    "validate_node_scores",
    "validate_communities",
    "validate_path_lengths",
    "validate_flow",
    "validate_similarity_scores",
    "validate_edge_scores",
    "validate_scalar_result",
]


class ValidationError(Exception):
    """Custom exception for validation failures."""


def validate_graph_result(
    result: Any,
) -> None:
    """Validate graph algorithm results."""
    if result is None:
        raise ValidationError("Algorithm returned None")

    if hasattr(result, "stats"):
        stats = result.stats
        if stats.get("mean") <= 0:
            raise ValidationError("Invalid timing value")
        if stats.get("std") < 0:
            raise ValidationError("Invalid timing standard deviation")


def validate_node_scores(
    result: dict[Any, float],
    graph: nx.Graph | nx.DiGraph,
    score_range: tuple[float, float] = (0.0, 1.0),
    require_normalized: bool = True,
    tolerance: float = 1e-6,
    normalization_factor: float | None = None,
    scale_by_n: bool = False,
) -> None:
    """Validate node-based scoring algorithms (e.g. centrality measures).

    Parameters
    ----------
    result : dict
        Dictionary mapping nodes to scores
    graph : nx.Graph or nx.DiGraph
        Original graph
    score_range : tuple of float, default=(0.0, 1.0)
        Expected range for scores (min, max)
    require_normalized : bool, default=True
        Whether scores should be normalized
    tolerance : float, default=1e-6
        Numerical tolerance for floating point comparisons
    normalization_factor : float, optional
        Custom normalization factor to divide summed scores by. If provided,
        this takes precedence over scale_by_n.
    scale_by_n : bool, default=False
        Whether to scale expected sum by number of nodes. Some measures like
        betweenness centrality scale differently based on graph size.

    Raises
    ------
    ValidationError
        If validation fails

    Notes
    -----
    The validation handles different normalization schemes:

    1. Basic normalization (sum to 1.0)
    2. Node count scaling (sum to n)
    3. Custom normalization factor
    4. No normalization
    """
    result = dict(result)

    if not isinstance(graph, nx.Graph) and hasattr(graph, "to_networkx"):
        graph = graph.to_networkx()

    graph_nodes = set(graph.nodes())

    if set(result.keys()) != graph_nodes:
        raise ValidationError(
            f"Result nodes {set(result.keys())} don't match "
            f"graph nodes {graph_nodes}"
        )

    min_score, max_score = score_range
    scores = np.array(list(result.values()))

    if np.any(np.isnan(scores)):
        raise ValidationError("Result contains NaN values")

    if np.any(np.isinf(scores)):
        raise ValidationError("Result contains infinite values")

    if np.any(scores < min_score - tolerance):
        raise ValidationError(
            f"Scores below minimum {min_score}: "
            f"{scores[scores < min_score - tolerance]}"
        )

    if np.any(scores > max_score + tolerance):
        raise ValidationError(
            f"Scores above maximum {max_score}: "
            f"{scores[scores > max_score + tolerance]}"
        )

    if require_normalized:
        score_sum = np.sum(scores)
        n = len(graph)

        if normalization_factor is not None:
            expected_sum = normalization_factor
            normalized_sum = score_sum / normalization_factor
        elif scale_by_n:
            expected_sum = float(n)
            normalized_sum = score_sum / expected_sum
        else:
            expected_sum = 1.0
            normalized_sum = score_sum

        if not np.isclose(normalized_sum, 1.0, rtol=tolerance):
            raise ValidationError(
                f"Normalized scores sum to {normalized_sum}, expected 1.0 "
                f"(raw sum: {score_sum}, norm factor: {normalization_factor})"
            )


def validate_communities(
    result: list[set],
    graph: nx.Graph | nx.DiGraph,
    allow_overlap: bool = False,
    min_community_size: int | None = None,
    check_connectivity: bool = True,
) -> None:
    """Validate community detection results.

    Parameters
    ----------
    result : list of sets
        List of node sets representing communities
    graph : nx.Graph or nx.DiGraph
        Original graph
    allow_overlap : bool, default=False
        Whether communities can overlap
    min_community_size : int, optional
        Minimum allowed community size
    check_connectivity : bool, default=True
        Whether to verify that communities are internally connected

    Raises
    ------
    ValidationError
        If validation fails
    """
    if not isinstance(result, list) or not all(isinstance(c, set) for c in result):
        raise ValidationError("Result must be a list of sets")

    all_nodes = set().union(*result)

    if all_nodes != set(graph.nodes()):
        raise ValidationError(
            f"Communities don't cover all graph nodes. "
            f"Missing: {set(graph.nodes()) - all_nodes}, "
            f"Extra: {all_nodes - set(graph.nodes())}"
        )

    if not allow_overlap:
        node_count = defaultdict(int)
        for community in result:
            for node in community:
                node_count[node] += 1

        overlapping = {n for n, count in node_count.items() if count > 1}
        if overlapping:
            raise ValidationError(
                f"Found overlapping communities for nodes: {overlapping}"
            )

    if min_community_size is not None:
        small_communities = [c for c in result if len(c) < min_community_size]
        if small_communities:
            raise ValidationError(
                f"Communities smaller than minimum size {min_community_size}: "
                f"{small_communities}"
            )

    if check_connectivity:
        for i, community in enumerate(result):
            subgraph = graph.subgraph(community)
            if not nx.is_connected(subgraph.to_undirected()):
                raise ValidationError(
                    f"Community {i} (size {len(community)}) is not internally connected"
                )


def validate_path_lengths(
    result: dict[Any, dict[Any, float]],
    graph: nx.Graph | nx.DiGraph,
    check_symmetry: bool = False,
    allow_infinity: bool = True,
) -> None:
    """Validate all-pairs shortest path lengths.

    Parameters
    ----------
    result : dict of dict
        Dictionary mapping source nodes to dictionaries mapping target nodes to
        distances
    graph : nx.Graph or nx.DiGraph
        Original graph
    check_symmetry : bool, default=False
        Whether to verify that distances are symmetric (for undirected graphs)
    allow_infinity : bool, default=True
        Whether infinite distances are allowed for disconnected nodes

    Raises
    ------
    ValidationError
        If validation fails
    """
    if not isinstance(result, dict):
        result = dict(result)

    result = {k: dict(v) for k, v in result.items()}

    if set(result.keys()) != set(graph.nodes()):
        raise ValidationError("Result nodes don't match graph nodes")

    nodes = set(graph.nodes())

    for source, distances in result.items():
        if set(distances.keys()) != nodes:
            raise ValidationError(f"Missing target nodes for source {source}")

        for target, distance in distances.items():
            if distance < 0:
                raise ValidationError(
                    f"Negative distance {distance} from {source} to {target}"
                )

            if not allow_infinity and np.isinf(distance):
                raise ValidationError(
                    f"Infinite distance from {source} to {target} not allowed"
                )

            if not np.isinf(distance) and distance > len(nodes) - 1:
                raise ValidationError(
                    f"Distance {distance} from {source} to {target} exceeds "
                    f"maximum possible distance {len(nodes) - 1}"
                )

    if check_symmetry and not graph.is_directed():
        for source in nodes:
            for target in nodes:
                if result[source][target] != result[target][source]:
                    raise ValidationError(
                        f"Asymmetric distances for undirected graph: "
                        f"{source}->{target}: {result[source][target]}, "
                        f"{target}->{source}: {result[target][source]}"
                    )


def validate_flow(
    result: tuple[float, dict[Any, dict[Any, float]]],
    graph: nx.Graph | nx.DiGraph,
    check_conservation: bool = True,
    tolerance: float = 1e-6,
) -> None:
    """Validate maximum flow results.

    Parameters
    ----------
    result : tuple
        (flow_value, flow_dict) tuple from max flow algorithm
    graph : nx.Graph or nx.DiGraph
        Original graph
    check_conservation : bool, default=True
        Whether to verify flow conservation at all nodes
    tolerance : float, default=1e-6
        Numerical tolerance for flow conservation checks

    Raises
    ------
    ValidationError
        If validation fails
    """
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValidationError(
            f"Expected (flow_value, flow_dict) tuple, got {type(result)}"
        )

    flow_value, flow_dict = result

    if not isinstance(flow_dict, dict):
        raise ValidationError(f"Expected dict flow_dict, got {type(flow_dict)}")

    if flow_value < 0:
        raise ValidationError(f"Negative flow value: {flow_value}")

    for source, targets in flow_dict.items():
        if source not in graph:
            raise ValidationError(f"Invalid source node in flow: {source}")

        for target, flow in targets.items():
            if target not in graph:
                raise ValidationError(f"Invalid target node in flow: {target}")

            if flow < 0:
                raise ValidationError(f"Negative flow {flow} from {source} to {target}")

            if "capacity" in graph[source][target]:
                capacity = graph[source][target]["capacity"]
                if flow > capacity + tolerance:
                    raise ValidationError(
                        f"Flow {flow} exceeds capacity {capacity} "
                        f"from {source} to {target}"
                    )

    if check_conservation:
        for node in graph.nodes():
            if node not in (graph.graph.get("source"), graph.graph.get("sink")):
                incoming = sum(
                    flow_dict.get(u, {}).get(node, 0) for u in graph.predecessors(node)
                )
                outgoing = sum(
                    flow_dict.get(node, {}).get(v, 0) for v in graph.successors(node)
                )
                if not np.isclose(incoming, outgoing, rtol=tolerance):
                    raise ValidationError(
                        f"Flow conservation violated at node {node}: "
                        f"incoming={incoming}, outgoing={outgoing}"
                    )


def validate_edge_scores(
    edge_scores: dict,
    graph: nx.Graph | nx.DiGraph,
    score_range: tuple = (0.0, 1.0),
) -> None:
    """Validate edge scores for a given graph.

    Parameters
    ----------
    edge_scores : dict
        Dictionary of edge scores where keys are tuples representing edges (u, v)
        and values are the scores associated with those edges.
    graph : networkx.Graph or networkx.DiGraph
        The graph for which the edge scores are being validated.
    score_range : tuple, default=(0.0, 1.0)
        The range (min, max) within which the edge scores should lie.

    Raises
    ------
    ValidationError
        If edge scores do not satisfy the expected conditions.
    """
    for u, v in graph.edges:
        if (u, v) not in edge_scores and (v, u) not in edge_scores:
            raise ValidationError(f"Edge ({u}, {v}) is missing a score.")

    min_score, max_score = score_range
    for edge, score in edge_scores.items():
        if not (min_score <= score <= max_score):
            raise ValidationError(
                f"Score for edge {edge} is {score}, which is outside the range "
                f"{score_range}."
            )

    if min_score < 0:
        raise ValidationError("Edge scores cannot be negative.")


def validate_similarity_scores(
    result: Iterable[tuple[Any, Any, float]],
    graph: nx.Graph | nx.DiGraph,
    score_range: tuple[float, float] = (0.0, 1.0),
    require_symmetric: bool = True,
    tolerance: float = 1e-6,
) -> None:
    """Validate link prediction similarity scores.

    Parameters
    ----------
    result : iterable of tuples
        (node_u, node_v, score) tuples from similarity measure
    graph : nx.Graph or nx.DiGraph
        Original graph
    score_range : tuple of float, default=(0.0, 1.0)
        Expected range for similarity scores
    require_symmetric : bool, default=True
        Whether similarity scores should be symmetric
    tolerance : float, default=1e-6
        Numerical tolerance for comparisons

    Raises
    ------
    ValidationError
        If validation fails
    """
    scores = list(result)

    if not all(len(item) == 3 for item in scores):
        raise ValidationError(
            "Each result item must be a (node_u, node_v, score) tuple"
        )

    min_score, max_score = score_range
    seen_pairs = set()
    symmetric_scores = defaultdict(dict)

    for u, v, score in scores:
        if u not in graph:
            raise ValidationError(f"Invalid node in result: {u}")
        if v not in graph:
            raise ValidationError(f"Invalid node in result: {v}")

        if u == v:
            raise ValidationError(f"Self-loop in result: {u}")

        if isinstance(graph, nx.DiGraph):
            pair = (u, v)
        else:
            pair = tuple(sorted([u, v]))

        if pair in seen_pairs:
            raise ValidationError(f"Duplicate node pair in result: {pair}")
        seen_pairs.add(pair)

        if np.isnan(score):
            raise ValidationError(f"NaN score for pair {(u, v)}")
        if np.isinf(score):
            raise ValidationError(f"Infinite score for pair {(u, v)}")
        if score < min_score - tolerance:
            raise ValidationError(
                f"Score {score} below minimum {min_score} for pair {(u, v)}"
            )
        if score > max_score + tolerance:
            raise ValidationError(
                f"Score {score} above maximum {max_score} for pair {(u, v)}"
            )

        symmetric_scores[u][v] = score

    if require_symmetric:
        if isinstance(graph, (nx.Graph, nx.DiGraph)):
            for u in symmetric_scores:
                for v, score in symmetric_scores[u].items():
                    if v in symmetric_scores and u in symmetric_scores[v]:
                        if not np.isclose(
                            score, symmetric_scores[v][u], rtol=tolerance
                        ):
                            raise ValidationError(
                                f"Asymmetric scores: {u}->{v}={score}, "
                                f"{v}->{u}={symmetric_scores[v][u]}"
                            )


def validate_scalar_result(
    result: Any,
    graph: nx.Graph | nx.DiGraph,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """Validate scalar result (e.g., float, int)."""
    if not isinstance(result, (int, float)):
        raise ValidationError("Expected result of type float or int")

    # heck min
    if min_value is not None and result < min_value:
        raise ValidationError(f"Result {result} is less than minimum {min_value}")

    # check max
    if max_value is not None and result > max_value:
        raise ValidationError(f"Result {result} is greater than maximum {max_value}")
