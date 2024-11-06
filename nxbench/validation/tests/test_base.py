import pytest
import networkx as nx
import numpy as np

from nxbench.validation.base import (
    ValidationError,
    validate_graph_result,
    validate_node_scores,
    validate_communities,
    validate_path_lengths,
    validate_flow,
    validate_similarity_scores,
    validate_edge_scores,
    validate_scalar_result,
)


def test_validate_graph_result_valid():
    class Result:
        stats = {"mean": 1.5, "std": 0.5}

    validate_graph_result(Result())


def test_validate_graph_result_none():
    with pytest.raises(ValidationError, match=r"Algorithm returned None"):
        validate_graph_result(None)


def test_validate_graph_result_invalid_mean(simple_graph):
    class Result:
        stats = {"mean": 0.0, "std": 0.5}

    with pytest.raises(ValidationError, match=r"Invalid timing value"):
        validate_graph_result(Result())


def test_validate_graph_result_invalid_std(simple_graph):
    class Result:
        stats = {"mean": 1.0, "std": -0.1}

    with pytest.raises(ValidationError, match=r"Invalid timing standard deviation"):
        validate_graph_result(Result())


def test_validate_node_scores_valid(simple_graph):
    scores = {node: 1.0 / len(simple_graph) for node in simple_graph.nodes()}
    validate_node_scores(scores, simple_graph)


def test_validate_node_scores_non_dict(simple_graph):
    scores = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValidationError, match=r"Expected dict result"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_mismatched_nodes(simple_graph):
    scores = {1: 0.25, 2: 0.25, 3: 0.25}  # Missing node 4
    with pytest.raises(ValidationError, match=r"don't match graph nodes"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_nan(simple_graph):
    scores = {node: 0.25 for node in simple_graph.nodes()}
    scores[1] = np.nan
    with pytest.raises(ValidationError, match=r"Result contains NaN values"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_inf(simple_graph):
    scores = {node: 0.25 for node in simple_graph.nodes()}
    scores[2] = np.inf
    with pytest.raises(ValidationError, match=r"Result contains infinite values"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_out_of_range_low(simple_graph):
    scores = {node: -0.1 for node in simple_graph.nodes()}
    with pytest.raises(ValidationError, match=r"Scores below minimum"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_out_of_range_high(simple_graph):
    scores = {node: 1.1 for node in simple_graph.nodes()}
    with pytest.raises(ValidationError, match=r"Scores above maximum"):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_not_normalized(simple_graph):
    scores = {node: 0.2 for node in simple_graph.nodes()}  # Sum = 0.8
    with pytest.raises(
        ValidationError, match=r"Normalized scores sum to .* expected 1\.0"
    ):
        validate_node_scores(scores, simple_graph)


def test_validate_node_scores_normalized_sum(simple_graph):
    scores = {node: 0.25 for node in simple_graph.nodes()}  # Sum = 1.0
    validate_node_scores(scores, simple_graph)


def test_validate_node_scores_normalized_sum_n(simple_graph):
    scores = {node: 1.0 for node in simple_graph.nodes()}  # Sum = 4
    validate_node_scores(scores, simple_graph, scale_by_n=True)


def test_validate_node_scores_custom_normalization(simple_graph):
    scores = {node: 0.5 for node in simple_graph.nodes()}  # Sum = 2.0
    validate_node_scores(scores, simple_graph, normalization_factor=2.0)


def test_validate_node_scores_normalization_failure(simple_graph):
    scores = {node: 0.4 for node in simple_graph.nodes()}  # Sum = 1.6
    with pytest.raises(
        ValidationError, match=r"Normalized scores sum to .* expected 1\.0"
    ):
        validate_node_scores(scores, simple_graph)


def test_validate_communities_valid(simple_graph):
    communities = [{1, 2}, {3, 4}]
    validate_communities(communities, simple_graph)


def test_validate_communities_not_list_of_sets(simple_graph):
    communities = [[1, 2], [3, 4]]
    with pytest.raises(ValidationError, match=r"list of sets"):
        validate_communities(communities, simple_graph)


def test_validate_communities_incomplete(simple_graph):
    communities = [{1, 2}]
    with pytest.raises(ValidationError, match=r"don't cover all graph nodes"):
        validate_communities(communities, simple_graph)


def test_validate_communities_overlap_not_allowed(simple_graph):
    communities = [{1, 2}, {2, 3}, {4}]
    with pytest.raises(ValidationError, match=r"overlapping communities"):
        validate_communities(communities, simple_graph)


def test_validate_communities_overlap_allowed(simple_graph):
    communities = [{1, 2}, {2, 3}, {4}]
    validate_communities(communities, simple_graph, allow_overlap=True)


def test_validate_communities_min_size(simple_graph):
    communities = [{1}, {2, 3, 4}]
    with pytest.raises(ValidationError, match=r"smaller than minimum size"):
        validate_communities(communities, simple_graph, min_community_size=2)


def test_validate_communities_min_size_valid(simple_graph):
    communities = [{1, 2}, {3, 4}]
    validate_communities(communities, simple_graph, min_community_size=2)


def test_validate_communities_connectivity(simple_graph):
    # remove edge to make community {1, 3} disconnected
    disconnected_graph = simple_graph.copy()
    disconnected_graph.remove_edge(1, 3)
    communities = [{1, 3}, {2, 4}]
    with pytest.raises(ValidationError, match=r"not internally connected"):
        validate_communities(communities, disconnected_graph)


def test_validate_communities_connectivity_disabled(simple_graph):
    # Remove edge to make community {1, 3} disconnected
    disconnected_graph = simple_graph.copy()
    disconnected_graph.remove_edge(1, 3)
    communities = [{1, 3}, {2, 4}]
    validate_communities(communities, disconnected_graph, check_connectivity=False)


def test_validate_path_lengths_valid(simple_graph):
    # Compute all-pairs shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(simple_graph))
    validate_path_lengths(lengths, simple_graph)


def test_validate_path_lengths_negative_distance(simple_graph):
    lengths = dict(nx.all_pairs_shortest_path_length(simple_graph))
    lengths[1][2] = -1.0
    with pytest.raises(ValidationError, match=r"Negative distance"):
        validate_path_lengths(lengths, simple_graph)


def test_validate_path_lengths_missing_node(simple_graph):
    lengths = dict(nx.all_pairs_shortest_path_length(simple_graph))
    del lengths[1][2]
    with pytest.raises(ValidationError, match=r"Missing target nodes"):
        validate_path_lengths(lengths, simple_graph)


def test_validate_path_lengths_exceeds_max(simple_graph):
    lengths = dict(nx.all_pairs_shortest_path_length(simple_graph))
    lengths[1][2] = len(simple_graph.nodes())  # Max possible distance is n-1=3
    with pytest.raises(ValidationError, match=r"exceeds maximum possible distance"):
        validate_path_lengths(lengths, simple_graph)


def test_validate_path_lengths_asymmetric(simple_graph):
    G = nx.Graph(simple_graph)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    # Manually introduce asymmetry
    lengths[1][2] = lengths[2][1] + 1
    with pytest.raises(ValidationError, match=r"Asymmetric distances"):
        validate_path_lengths(lengths, G, check_symmetry=True)


def test_validate_path_lengths_symmetric(simple_graph):
    G = nx.Graph(simple_graph)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    validate_path_lengths(lengths, G, check_symmetry=True)


def test_validate_path_lengths_directed_symmetry(directed_graph):
    lengths = dict(nx.all_pairs_shortest_path_length(directed_graph))
    # Since the graph is directed, symmetry is not required
    validate_path_lengths(lengths, directed_graph, check_symmetry=True)


def test_validate_path_lengths_allow_infinity():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(1, 2)
    lengths = {
        1: {1: 0, 2: 1, 3: np.inf},
        2: {1: 1, 2: 0, 3: np.inf},
        3: {1: np.inf, 2: np.inf, 3: 0},
    }
    validate_path_lengths(lengths, G, allow_infinity=True)


def test_validate_path_lengths_disallow_infinity():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(1, 2)
    lengths = {
        1: {1: 0, 2: 1, 3: np.inf},
        2: {1: 1, 2: 0, 3: np.inf},
        3: {1: np.inf, 2: np.inf, 3: 0},
    }
    with pytest.raises(ValidationError, match=r"Infinite distance"):
        validate_path_lengths(lengths, G, allow_infinity=False)


# Tests for validate_flow
def test_validate_flow_valid(flow_graph):
    flow_value = 15.0
    flow_dict = {"s": {"A": 10, "B": 5}, "A": {"B": 0, "t": 10}, "B": {"t": 5}, "t": {}}
    validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_invalid_tuple(flow_graph):
    with pytest.raises(
        ValidationError, match=r"Expected \(flow_value, flow_dict\) tuple"
    ):
        validate_flow("invalid", flow_graph)


def test_validate_flow_negative_flow(flow_graph):
    flow_value = -5.0
    flow_dict = {}
    with pytest.raises(ValidationError, match=r"Negative flow value"):
        validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_invalid_source(flow_graph):
    flow_value = 10.0
    flow_dict = {
        "x": {"A": 10},
    }
    with pytest.raises(ValidationError, match=r"Invalid source node in flow"):
        validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_invalid_target(flow_graph):
    flow_value = 10.0
    flow_dict = {
        "s": {"x": 10},
    }
    with pytest.raises(ValidationError, match=r"Invalid target node in flow"):
        validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_exceeds_capacity(flow_graph):
    flow_value = 20.0
    flow_dict = {"s": {"A": 15, "B": 5}, "A": {"t": 10}, "B": {"t": 5}, "t": {}}
    with pytest.raises(ValidationError, match=r"exceeds capacity"):
        validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_conservation_violation(flow_graph):
    flow_value = 15.0
    flow_dict = {"s": {"A": 10, "B": 5}, "A": {"t": 9}, "B": {"t": 5}, "t": {}}
    with pytest.raises(ValidationError, match=r"Flow conservation violated at node"):
        validate_flow((flow_value, flow_dict), flow_graph)


def test_validate_flow_conservation_disabled(flow_graph):
    flow_value = 15.0
    flow_dict = {"s": {"A": 10, "B": 5}, "A": {"t": 9}, "B": {"t": 5}, "t": {}}
    validate_flow((flow_value, flow_dict), flow_graph, check_conservation=False)


def test_validate_edge_scores_valid(simple_graph):
    edge_scores = {edge: 0.5 for edge in simple_graph.edges()}
    validate_edge_scores(edge_scores, simple_graph)


def test_validate_edge_scores_missing_edge(simple_graph):
    edge_scores = {edge: 0.5 for edge in list(simple_graph.edges())[:-1]}
    with pytest.raises(ValidationError, match=r"is missing a score"):
        validate_edge_scores(edge_scores, simple_graph)


def test_validate_edge_scores_out_of_range(simple_graph):
    edge_scores = {edge: 1.5 for edge in simple_graph.edges()}
    with pytest.raises(ValidationError, match=r"is outside the range"):
        validate_edge_scores(edge_scores, simple_graph)


def test_validate_edge_scores_negative(simple_graph):
    edge_scores = {edge: -0.1 for edge in simple_graph.edges()}
    with pytest.raises(ValidationError, match=r"is outside the range"):
        validate_edge_scores(edge_scores, simple_graph)


def test_validate_edge_scores_directed(directed_graph):
    edge_scores = {edge: 0.7 for edge in directed_graph.edges()}
    validate_edge_scores(edge_scores, directed_graph)


def test_validate_edge_scores_undirected_duplicate(simple_graph):
    edge_scores = {}
    for u, v in simple_graph.edges():
        edge_scores[(u, v)] = 0.5
        edge_scores[(v, u)] = 0.5
    validate_edge_scores(edge_scores, simple_graph)


def test_validate_similarity_scores_valid(simple_graph):
    scores = []
    for u, v in simple_graph.edges():
        scores.append((u, v, 0.8))
    validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_invalid_tuple_length(simple_graph):
    scores = [(1, 2, 0.5), (2, 3)]
    with pytest.raises(ValidationError, match=r"Each result item must be a"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_invalid_node(simple_graph):
    scores = [(1, 2, 0.5), (2, 5, 0.6)]  # Node 5 does not exist
    with pytest.raises(ValidationError, match=r"Invalid node in result: 5"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_self_loop(simple_graph):
    scores = [(1, 1, 0.5)]
    with pytest.raises(ValidationError, match=r"Self-loop in result: 1"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_duplicate_pair(simple_graph):
    scores = [(1, 2, 0.5), (2, 1, 0.6)]
    with pytest.raises(ValidationError, match=r"Duplicate node pair in result:"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_out_of_range(simple_graph):
    scores_low = [(1, 2, -0.1)]
    with pytest.raises(ValidationError, match=r"Score .* below minimum"):
        validate_similarity_scores(scores_low, simple_graph)

    scores_high = [(2, 3, 1.1)]
    with pytest.raises(ValidationError, match=r"Score .* above maximum"):
        validate_similarity_scores(scores_high, simple_graph)


def test_validate_similarity_scores_nan(simple_graph):
    scores = [(1, 2, np.nan), (2, 3, 0.8)]
    with pytest.raises(ValidationError, match=r"NaN score for pair"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_inf(simple_graph):
    scores = [(1, 2, 0.7), (2, 3, np.inf)]
    with pytest.raises(ValidationError, match=r"Infinite score for pair"):
        validate_similarity_scores(scores, simple_graph)


def test_validate_similarity_scores_asymmetric(directed_graph):
    scores = [("A", "B", 0.7), ("B", "A", 0.8)]
    with pytest.raises(ValidationError, match=r"Asymmetric scores"):
        validate_similarity_scores(scores, directed_graph)


def test_validate_similarity_scores_symmetric(simple_graph):
    scores = [(1, 2, 0.7), (3, 4, 0.8)]
    validate_similarity_scores(scores, simple_graph, require_symmetric=True)


def test_validate_similarity_scores_no_symmetry(simple_graph):
    scores = [(1, 2, 0.7), (2, 3, 0.8)]
    validate_similarity_scores(scores, simple_graph, require_symmetric=False)


def test_validate_scalar_result_valid():
    validate_scalar_result(3.14, None)


def test_validate_scalar_result_invalid_type():
    with pytest.raises(
        ValidationError, match=r"Expected result of type <class 'float'>"
    ):
        validate_scalar_result("not a float", None)


def test_validate_scalar_result_min_value():
    validate_scalar_result(5, None, expected_type=int, min_value=3)
    with pytest.raises(ValidationError, match=r"is less than minimum"):
        validate_scalar_result(2, None, expected_type=int, min_value=3)


def test_validate_scalar_result_max_value():
    validate_scalar_result(5, None, expected_type=int, max_value=10)
    with pytest.raises(ValidationError, match=r"is greater than maximum"):
        validate_scalar_result(15, None, expected_type=int, max_value=10)


def test_validate_scalar_result_custom_type():
    validate_scalar_result(10, None, expected_type=int)
    with pytest.raises(ValidationError, match=r"Expected result of type <class 'int'>"):
        validate_scalar_result(10.5, None, expected_type=int)
