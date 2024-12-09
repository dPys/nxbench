import pytest

from nxbench.data.synthesize import generate_graph

valid_generators = [
    (
        "networkx.generators.random_graphs.erdos_renyi_graph",
        {"n": 10, "p": 0.5},
        False,
        {"is_directed": False, "expected_nodes": 10, "expected_edges_range": (0, 45)},
    ),
    (
        "networkx.generators.random_graphs.erdos_renyi_graph",
        {"n": 10, "p": 0.5},
        True,
        {"is_directed": True, "expected_nodes": 10, "expected_edges_range": (0, 90)},
    ),
    (
        "networkx.generators.random_graphs.barabasi_albert_graph",
        {"n": 15, "m": 2},
        False,
        {"is_directed": False, "expected_nodes": 15, "expected_edges_range": (14, 30)},
    ),
    (
        "networkx.generators.classic.path_graph",
        {"n": 5},
        False,
        {"is_directed": False, "expected_nodes": 5, "expected_edges": 4},
    ),
    (
        "networkx.generators.classic.cycle_graph",
        {"n": 7},
        True,
        {"is_directed": True, "expected_nodes": 7, "expected_edges": 14},
    ),
]


@pytest.mark.parametrize(
    ("generator_name", "gen_params", "directed", "expectations"), valid_generators
)
def test_generate_graph_valid(generator_name, gen_params, directed, expectations):
    """Test generate_graph with valid generator names and parameters."""
    graph = generate_graph(generator_name, gen_params, directed)

    assert graph.is_directed() == expectations["is_directed"], (
        f"Graph directedness mismatch: expected {expectations['is_directed']}, "
        f"got {graph.is_directed()}"
    )

    assert graph.number_of_nodes() == expectations["expected_nodes"], (
        f"Number of nodes mismatch: expected {expectations['expected_nodes']}, "
        f"got {graph.number_of_nodes()}"
    )

    if "expected_edges" in expectations:
        assert graph.number_of_edges() == expectations["expected_edges"], (
            f"Number of edges mismatch: expected {expectations['expected_edges']}, "
            f"got {graph.number_of_edges()}"
        )
    elif "expected_edges_range" in expectations:
        lower, upper = expectations["expected_edges_range"]
        assert lower <= graph.number_of_edges() <= upper, (
            f"Number of edges {graph.number_of_edges()} not in expected range "
            f"{expectations['expected_edges_range']}"
        )


invalid_generators = [
    ("", {"n": 10, "p": 0.5}, False, ValueError, "Generator name must be specified."),
    (
        "networkx.invalid_module.erdos_renyi_graph",
        {"n": 10, "p": 0.5},
        False,
        ValueError,
        "Invalid generator",
    ),
    (
        "networkx.generators.random_graphs.invalid_graph",
        {"n": 10, "p": 0.5},
        False,
        ValueError,
        "Invalid generator",
    ),
    (
        "networkx.generators.random_graphs.erdos_renyi_graph",
        {"n": -5, "p": 1.5},
        False,
        ValueError,
        "Failed to generate graph",
    ),
    (
        "networkx.generators.classic.path_graph",
        {"n": 0},
        False,
        None,
        "empty",
    ),
]


@pytest.mark.parametrize(
    (
        "generator_name",
        "gen_params",
        "directed",
        "expected_exception",
        "expected_message",
    ),
    invalid_generators,
)
def test_generate_graph_invalid(
    generator_name, gen_params, directed, expected_exception, expected_message
):
    """Test generate_graph with invalid generator names or parameters."""
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            generate_graph(generator_name, gen_params, directed)

        assert expected_message in str(
            exc_info.value
        ), f"Exception message does not contain '{expected_message}'. "
        f"Actual message: '{exc_info.value}'"
    elif expected_message == "empty":
        graph = generate_graph(generator_name, gen_params, directed)
        assert graph.number_of_nodes() == 0, "Expected an empty graph when n=0"
        assert graph.number_of_edges() == 0, "Expected no edges in an empty graph"
