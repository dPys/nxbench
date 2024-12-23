from unittest.mock import patch

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
    Provide a mocked return value for load_and_prepare_data that includes a 3-level
    MultiIndex: (algorithm, dataset, backend_full). This way, "dataset" and
    "backend_full"
    are legitimate levels in df_agg.

    Returns
    -------
    tuple
        (df, df_agg, group_columns, available_parcats_columns)
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "algorithm": ["bfs", "bfs", "dfs"],
            "dataset": ["ds1", "ds2", "ds3"],
            "backend_full": ["cpu", "gpu", "cpu"],
            "execution_time_with_preloading": [0.4, 0.5, 0.3],
            "execution_time": [0.6, 0.7, 0.4],
            "memory_used": [0.2, 0.1, 0.3],
        }
    )

    index = pd.MultiIndex.from_tuples(
        [
            ("bfs", "ds1", "cpu"),
            ("bfs", "ds2", "gpu"),
            ("dfs", "ds3", "cpu"),
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
    """
    Patch 'nxbench.viz.app.load_and_prepare_data' to return our mock data,
    ensuring no real CSV file is read.
    """
    with patch("nxbench.viz.app.load_and_prepare_data") as mocked:
        mocked.return_value = mock_load_and_prepare_data_return
        yield mocked


def test_app_runs_without_crashing(mock_load_data_function):
    """
    Test that the Dash app can be instantiated without errors and without
    starting the server.
    """
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
def test_make_parallel_categories_figure(color_by, mock_load_data_function):
    """
    Test the logic function for building the parallel categories figure with various
    color_by parameters.
    """
    df, df_agg, group_columns, available_parcats_columns = (
        mock_load_data_function.return_value
    )

    selected_algorithm = "bfs"
    selected_dimensions = ["dataset", "backend_full"]

    fig, store_data = make_parallel_categories_figure(
        df, df_agg, group_columns, selected_algorithm, color_by, selected_dimensions
    )

    # Basic checks
    assert isinstance(fig, go.Figure), "Expected a Plotly Figure object"
    assert store_data is not None, "Expected non-None store_data for hover info."

    # Check colorbar title logic
    if color_by == "execution_time":
        expected_title = "Execution Time (s)"
    elif color_by == "execution_time_with_preloading":
        expected_title = "Execution Time w/ Preloading (s)"
    else:
        expected_title = "Memory Used (GB)"

    assert len(fig.data) > 0, "Figure has no data traces."
    trace = fig.data[0]
    assert isinstance(
        trace, go.Parcats
    ), "Expected the first trace to be a Parcats plot."

    actual_title = trace.line.colorbar.title.text
    assert (
        actual_title == expected_title
    ), f"Colorbar title should be '{expected_title}', got '{actual_title}'"


def test_make_violin_figure_no_data(mock_load_data_function):
    """Test that make_violin_figure returns a figure with a
    "No data available for the selected algorithm." annotation when given a
    non-existent algorithm.

    Parameters
    ----------
    mock_load_data_function : pytest.fixture
        Fixture that mocks the data loading, ensuring a consistent dataset.
    """
    df, df_agg, group_columns, available_parcats_columns = (
        mock_load_data_function.return_value
    )

    selected_algorithm = "fakealgo"
    color_by = "execution_time"
    selected_dimensions = ["dataset", "backend_full"]

    fig = make_violin_figure(
        df, df_agg, selected_algorithm, color_by, selected_dimensions
    )

    assert fig.layout.annotations, "Expected annotations in the layout for no data."

    expected_text = "No data available for the selected algorithm."
    assert any(
        expected_text in ann["text"] for ann in fig.layout.annotations
    ), f"Could not find '{expected_text}' annotation in the figure."
