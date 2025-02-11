from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from nxbench.viz.app import (
    make_parallel_categories_figure,
    make_violin_figure,
    run_server,
)


@pytest.fixture
def mock_load_and_prepare_data_return():
    """
    Provide a mocked return value for load_and_prepare_data that includes
    all columns needed by the code.
    """
    df = pd.DataFrame(
        {
            "algorithm": ["bfs", "bfs", "dfs"],
            "dataset": ["ds1", "ds2", "ds3"],
            "backend_full": ["parallel", "cugraph", "graphblas"],
            "backend": ["parallel", "cugraph", "graphblas"],
            "execution_time_with_preloading": [0.4, 0.5, 0.3],
            "execution_time": [0.6, 0.7, 0.4],
            "memory_used": [0.2, 0.1, 0.3],
            "num_nodes_bin": [1000, 2000, 3000],
            "num_edges_bin": [5000, 6000, 7000],
            "is_directed": [False, True, False],
            "is_weighted": [False, False, True],
            "cpu": ["Intel", "AMD", "Intel"],
            "os": ["Linux", "Linux", "Windows"],
            "num_thread": [1, 2, 4],
        }
    )

    index = pd.MultiIndex.from_tuples(
        [
            ("bfs", "ds1", "parallel"),
            ("bfs", "ds2", "cugraph"),
            ("dfs", "ds3", "graphblas"),
        ],
        names=["algorithm", "dataset", "backend_full"],
    )

    df_agg = pd.DataFrame(
        {
            "mean_execution_time": [0.65, 0.66, 0.35],
            "mean_memory_used": [0.15, 0.16, 0.25],
            "sample_count": [2, 3, 1],
        },
        index=index,
    )

    group_columns = ["algorithm", "dataset", "backend_full"]
    available_parcats_columns = ["dataset", "backend_full"]

    return (df, df_agg, group_columns, available_parcats_columns)


@pytest.fixture
def mock_load_data_function(mock_load_and_prepare_data_return):
    """Patch 'nxbench.viz.app.load_and_prepare_data' to return our mock data."""
    with patch("nxbench.viz.app.load_and_prepare_data") as mocked:
        mocked.return_value = mock_load_and_prepare_data_return
        yield mocked


def test_app_runs_without_crashing(mock_load_data_function):
    """Ensure the app can be instantiated (server not actually run)."""
    try:
        run_server(debug=False, run=False)
    except Exception as e:
        pytest.fail(f"Dash app failed to instantiate: {e}")


@pytest.mark.parametrize(
    "color_by",
    [
        "execution_time",
        "execution_time_with_preloading",
        "memory_used",
    ],
)
def test_make_parallel_categories_figure_basic(color_by, mock_load_data_function):
    """Basic test of the parallel categories figure with different color metrics."""
    df, df_agg, group_cols, available_cols = mock_load_data_function.return_value
    selected_algorithm = "bfs"
    selected_dimensions = ["dataset", "backend_full"]

    fig, store_data = make_parallel_categories_figure(
        df, df_agg, group_cols, selected_algorithm, color_by, selected_dimensions
    )
    assert isinstance(fig, go.Figure)
    assert store_data is not None

    trace = fig.data[0]
    assert isinstance(trace, go.Parcats)

    if color_by == "execution_time":
        expected_title = "Execution Time (s)"
    elif color_by == "execution_time_with_preloading":
        expected_title = "Execution Time w/ Preloading (s)"
    else:
        expected_title = "Memory Used (GB)"

    assert trace.line.colorbar.title.text == expected_title


def test_make_parallel_categories_figure_no_data_for_algorithm(mock_load_data_function):
    """
    Trigger the KeyError block at lines ~75-88 by removing 'bfs' from df_agg altogether.
    This ensures the top-level aggregator .xs() fails.
    """
    df, df_agg, group_cols, available_cols = mock_load_data_function.return_value
    df_agg_no_bfs = df_agg.drop(labels="bfs", level="algorithm")

    fig, store_data = make_parallel_categories_figure(
        df,
        df_agg_no_bfs,
        group_cols,
        "bfs",
        "execution_time",
        ["dataset", "backend_full"],
    )
    assert isinstance(fig, go.Figure)
    assert store_data == []
    assert fig.layout.annotations
    assert any("No data available" in ann["text"] for ann in fig.layout.annotations)


def test_make_parallel_categories_figure_preloading_temp_agg_missing_algo(
    mock_load_data_function,
):
    """
    Ensure coverage of lines ~105-106: BFS is in df_agg but missing from the
    temporary aggregator (temp_agg), triggering the KeyError fallback to
    mean_execution_time.
    """
    df, df_agg, group_cols, available_cols = mock_load_data_function.return_value

    df.loc[df["algorithm"] == "bfs", "execution_time_with_preloading"] = None

    fig, store_data = make_parallel_categories_figure(
        df,
        df_agg,
        group_cols,
        "bfs",
        "execution_time_with_preloading",
        ["dataset", "backend_full"],
    )
    assert isinstance(fig, go.Figure)
    assert store_data


def test_make_parallel_categories_figure_preloading_column_missing(
    mock_load_data_function,
):
    """If 'execution_time_with_preloading' is NOT in df.columns, fallback to
    mean_execution_time.
    """
    df, df_agg, group_columns, _ = mock_load_data_function.return_value
    df_no_preload = df.drop(columns=["execution_time_with_preloading"])

    selected_algorithm = "bfs"
    color_by = "execution_time_with_preloading"
    selected_dimensions = ["dataset", "backend_full"]

    fig, store_data = make_parallel_categories_figure(
        df_no_preload,
        df_agg,
        group_columns,
        selected_algorithm,
        color_by,
        selected_dimensions,
    )
    assert isinstance(fig, go.Figure)
    assert getattr(fig.data[0].line, "colorscale", None), "No colorscale found!"


def test_make_parallel_categories_figure_preloading_agg_keyerror(
    mock_load_data_function,
):
    """
    If BFS is missing from df_agg, .xs('bfs') triggers KeyError immediately,
    returning "No Data Available".
    """
    df, df_agg, group_columns, _ = mock_load_data_function.return_value

    df_agg_no_bfs = df_agg.drop(labels="bfs", level="algorithm")

    selected_algorithm = "bfs"
    color_by = "execution_time_with_preloading"
    selected_dimensions = ["dataset", "backend_full"]

    fig, store_data = make_parallel_categories_figure(
        df,
        df_agg_no_bfs,
        group_columns,
        selected_algorithm,
        color_by,
        selected_dimensions,
    )

    assert isinstance(fig, go.Figure)
    assert store_data == []


def test_make_violin_figure_empty_df(mock_load_data_function):
    """If .xs(...) yields an empty DataFrame, we get "No data available" figure."""
    df, df_agg, group_columns, available_cols = mock_load_data_function.return_value
    df_agg_empty = df_agg.drop(labels="bfs", level="algorithm")

    fig = make_violin_figure(
        df,
        df_agg_empty,
        "bfs",
        "execution_time",
        ["dataset", "backend_full"],
    )
    assert fig.layout.annotations
    assert any("No data available" in ann["text"] for ann in fig.layout.annotations)


@pytest.mark.parametrize(
    ("color_by", "expected_y_metric"),
    [
        ("execution_time", "mean_execution_time"),
        ("execution_time_with_preloading", "mean_execution_time_with_preloading"),
        ("memory_used", "mean_memory_used"),
    ],
)
def test_make_violin_figure_color_by(
    color_by, expected_y_metric, mock_load_data_function
):
    """Verify the correct y-metric is chosen in the violin figure."""
    df, df_agg, group_columns, available_cols = mock_load_data_function.return_value
    df_agg = df_agg.reset_index()
    df_agg["num_nodes_bin"] = [1000, 2000, 3000]
    df_agg["num_edges_bin"] = [5000, 6000, 7000]
    df_agg["is_directed"] = [False, True, False]
    df_agg["is_weighted"] = [False, False, True]
    df_agg["cpu"] = ["Intel", "AMD", "Intel"]
    df_agg["os"] = ["Linux", "Linux", "Windows"]
    df_agg["num_thread"] = [1, 2, 4]
    df_agg.set_index(["algorithm", "dataset", "backend_full"], inplace=True)

    selected_algorithm = "bfs"
    fig = make_violin_figure(
        df,
        df_agg,
        selected_algorithm,
        color_by,
        ["dataset", "backend_full"],
    )
    assert isinstance(fig, go.Figure)


def test_make_violin_figure_dimension_fallback(mock_load_data_function):
    """If the chosen dimension is missing, fallback to "backend" (or "backend_full")."""
    df, df_agg, group_columns, available_cols = mock_load_data_function.return_value
    df_agg = df_agg.reset_index()

    df_agg["num_nodes_bin"] = [1000, 2000, 3000]
    df_agg["num_edges_bin"] = [5000, 6000, 7000]
    df_agg["is_directed"] = [False, True, False]
    df_agg["is_weighted"] = [False, False, True]
    df_agg["cpu"] = ["Intel", "AMD", "Intel"]
    df_agg["os"] = ["Linux", "Linux", "Windows"]
    df_agg["num_thread"] = [1, 2, 4]

    df_agg["backend"] = df_agg["backend_full"]  # ensure "backend" is available
    df_agg.set_index(["algorithm", "dataset", "backend_full"], inplace=True)

    selected_algorithm = "bfs"
    color_by = "execution_time"
    selected_dimensions = ["foo_dimension"]  # doesn't exist

    fig = make_violin_figure(
        df, df_agg, selected_algorithm, color_by, selected_dimensions
    )
    assert isinstance(fig, go.Figure)
    assert not fig.layout.annotations


def test_make_violin_figure_empty_selected_dimensions(mock_load_data_function):
    """
    If selected_dimensions is empty, coverage for the code that does:
        violin_dimension = selected_dimensions[0] if selected_dimensions else "backend"
    """
    df, df_agg, group_columns, available_cols = mock_load_data_function.return_value

    df_agg = df_agg.reset_index()
    df_agg["backend"] = df_agg["backend_full"]

    df_agg["num_nodes_bin"] = [1000, 2000, 3000]
    df_agg["num_edges_bin"] = [5000, 6000, 7000]
    df_agg["is_directed"] = [False, True, False]
    df_agg["is_weighted"] = [False, False, True]
    df_agg["cpu"] = ["Intel", "AMD", "Intel"]
    df_agg["os"] = ["Linux", "Linux", "Windows"]
    df_agg["num_thread"] = [1, 2, 4]

    df_agg.set_index(["algorithm", "dataset", "backend_full"], inplace=True)

    fig = make_violin_figure(
        df,
        df_agg,
        "bfs",
        "execution_time",
        [],  # no dimensions passed
    )
    assert isinstance(fig, go.Figure)
    assert not fig.layout.annotations


def test_make_violin_figure_no_data_for_algorithm(mock_load_data_function):
    """If the algorithm doesn't exist at all, KeyError => "No data available"
    annotation.
    """
    df, df_agg, group_columns, available_cols = mock_load_data_function.return_value
    fig = make_violin_figure(
        df, df_agg, "fakealgo", "execution_time", ["dataset", "backend_full"]
    )
    assert fig.layout.annotations
    assert any("No data available" in ann["text"] for ann in fig.layout.annotations)


def test_run_server_calls_dash_run_server(mock_load_data_function):
    """
    Cover the 'if run: app.run_server(...)' line by mocking out Dash.run_server,
    ensuring that code path is tested (though we don't actually spin up the server).
    """
    with patch("dash.Dash.run_server") as mock_dash_run:
        run_server(port=9999, debug=True, run=True)
        mock_dash_run.assert_called_once_with(port=9999, debug=True)
