import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from nxbench.viz.utils import load_and_prepare_data

logger = logging.getLogger("nxbench")


def make_parallel_categories_figure(
    df,
    df_agg,
    group_columns,
    selected_algorithm,
    color_by,
    selected_dimensions,
):
    """
    Generate the Parallel Categories figure for a given algorithm and coloring metric.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe containing raw benchmark information
        (e.g., 'execution_time', 'memory_used', etc.).
    df_agg : pandas.DataFrame
        An aggregated dataframe (e.g., grouped by algorithm) containing mean execution
        time, memory usage, and sample counts. Indexed by ('algorithm', ...).
    group_columns : list of str
        The list of columns used to group and aggregate `df` into `df_agg`.
    selected_algorithm : str
        The name of the selected algorithm (e.g., "bfs", "dfs", etc.).
    color_by : str
        The metric by which to color the parallel categories. Possible values include:
        "execution_time", "execution_time_with_preloading", or "memory_used".
    selected_dimensions : list of str
        Columns (from the dataframe index) to include as dimensions in the parallel
        categories diagram.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly Figure object representing the parallel categories chart.
    store_data : dict
        A dictionary containing arrays of "mean_values" and "counts". Used for hover
        data replacement on the client side.
    """
    selected_algorithm = selected_algorithm.lower()

    try:
        filtered_df = df_agg.xs(selected_algorithm, level="algorithm")
        filtered_df = filtered_df.sort_index()
    except KeyError:
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            annotations=[
                {
                    "text": "No data available for the selected algorithm.",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20},
                }
            ],
        )
        return fig, []

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            annotations=[
                {
                    "text": "No data available for the selected algorithm.",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20},
                }
            ],
        )
        return fig, []

    if color_by == "execution_time":
        mean_values = filtered_df["mean_execution_time"]
        colorbar_title = "Execution Time (s)"
    elif color_by == "execution_time_with_preloading":
        if "execution_time_with_preloading" in df.columns:
            temp_agg = df.groupby(group_columns, as_index=False, observed=True).agg(
                mean_execution_time_with_preloading=(
                    "execution_time_with_preloading",
                    "mean",
                ),
            )
            temp_agg.set_index(group_columns, inplace=True)
            try:
                filtered_pre = temp_agg.xs(selected_algorithm, level="algorithm")
                mean_values = filtered_pre["mean_execution_time_with_preloading"]
            except KeyError:
                mean_values = filtered_df["mean_execution_time"]
        else:
            mean_values = filtered_df["mean_execution_time"]
        colorbar_title = "Execution Time w/ Preloading (s)"
    else:
        mean_values = filtered_df["mean_memory_used"]
        colorbar_title = "Memory Used (GB)"

    counts = filtered_df["sample_count"].values
    color_values = mean_values.values

    dims = [
        {
            "label": dim_col.replace("_", " ").title(),
            "values": filtered_df.index.get_level_values(dim_col),
        }
        for dim_col in selected_dimensions
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Parcats(
            dimensions=dims,
            line={
                "color": color_values,
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": colorbar_title},
            },
            counts=counts,
            hoverinfo="count",
            hovertemplate="Count: REPLACE_COUNT\nMean: REPLACE_ME<extra></extra>",
            arrangement="freeform",
        )
    )
    fig.update_layout(
        title=f"Benchmark Results for {selected_algorithm.title()}",
        template="plotly_dark",
    )

    store_data = {"mean_values": color_values.tolist(), "counts": counts.tolist()}

    return fig, store_data


def make_violin_figure(
    df,
    df_agg,
    selected_algorithm,
    color_by,
    selected_dimensions,
):
    """
    Generate the Violin Plot figure for a given algorithm and coloring metric.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe containing raw benchmark information
        (e.g., 'execution_time', 'memory_used', etc.).
    df_agg : pandas.DataFrame
        An aggregated dataframe containing mean execution time, memory usage,
        and sample counts. Indexed by ('algorithm', ...).
    selected_algorithm : str
        The name of the selected algorithm (e.g., "bfs", "dfs", etc.).
    color_by : str
        The metric by which to color the violin plot on the y-axis. Possible
        values are "execution_time", "execution_time_with_preloading", or
        "memory_used".
    selected_dimensions : list of str
        Columns that might be used for the X-axis dimension (e.g., "backend").

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly Figure object representing the violin chart.
    """
    selected_algorithm = selected_algorithm.lower()

    try:
        filtered_df = df_agg.xs(selected_algorithm, level="algorithm").reset_index()
    except KeyError:
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            annotations=[
                {
                    "text": "No data available for the selected algorithm.",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20},
                }
            ],
        )
        return fig

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            annotations=[
                {
                    "text": "No data available for the selected algorithm.",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20},
                }
            ],
        )
        return fig

    if color_by == "execution_time":
        y_metric = "mean_execution_time"
        y_label = "Execution Time"
    elif color_by == "execution_time_with_preloading":
        if "mean_execution_time_with_preloading" in filtered_df.columns:
            y_metric = "mean_execution_time_with_preloading"
            y_label = "Execution Time w/ Preloading"
        else:
            y_metric = "mean_execution_time"
            y_label = "Execution Time"
    else:
        y_metric = "mean_memory_used"
        y_label = "Memory Used"

    violin_dimension = selected_dimensions[0] if selected_dimensions else "backend"
    if violin_dimension not in filtered_df.columns:
        violin_dimension = "backend"

    fig = px.violin(
        filtered_df,
        x=violin_dimension,
        y=y_metric,
        color=violin_dimension,
        box=True,
        points="all",
        hover_data=[
            "dataset",
            "num_nodes_bin",
            "num_edges_bin",
            "is_directed",
            "is_weighted",
            "python_version",
            "cpu",
            "os",
            "num_thread",
            "sample_count",
        ],
        title=f"{y_label} Distribution for {selected_algorithm.title()}",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#303030",
        plot_bgcolor="#303030",
        font_color="#fff",
    )
    return fig


def run_server(port=8050, debug=False, run=True):
    """
    Create and configure the Dash app, optionally running the server.

    This function loads benchmark data, constructs the Dash app, and wires up
    all callbacks. The `run` parameter allows for skipping the server startup,
    which is convenient when testing.

    Parameters
    ----------
    port : int, optional
        The port on which to run the Dash server. Default is 8050.
    debug : bool, optional
        Whether to run the server in debug mode. Default is False.
    run : bool, optional
        If True, the app.run_server method is called. If False, the app is returned
        without starting the server. This is useful for testing. Default is True.

    Returns
    -------
    dash.Dash
        The Dash application instance.
    """
    df, df_agg, group_columns, available_parcats_columns = load_and_prepare_data(
        "results/results.csv", logger
    )

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    app.layout = html.Div(
        style={"backgroundColor": "#303030", "color": "#fff"},
        children=[
            html.H1(
                "NetworkX Benchmark Dashboard",
                style={"textAlign": "center", "color": "#fff"},
            ),
            html.Div(
                [
                    html.Label("Select Algorithm:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="algorithm-dropdown",
                        style={"width": "100%", "color": "#000"},
                        options=[
                            {"label": alg.title(), "value": alg}
                            for alg in sorted(
                                df_agg.index.get_level_values("algorithm").unique()
                            )
                        ],
                        value=sorted(
                            df_agg.index.get_level_values("algorithm").unique()
                        )[0],
                        clearable=False,
                    ),
                ],
                style={"width": "48%", "display": "inline-block", "padding": "0 20px"},
            ),
            html.Div(
                [
                    html.Label("Color By:", style={"fontWeight": "bold"}),
                    dbc.RadioItems(
                        id="color-toggle",
                        options=[
                            {"label": "Execution Time", "value": "execution_time"},
                            {"label": "Memory Used", "value": "memory_used"},
                            {
                                "label": "Execution Time with Preloading",
                                "value": "execution_time_with_preloading",
                            },
                        ],
                        value="execution_time",
                        inline=True,
                        className="ml-2",
                    ),
                ],
                style={
                    "width": "48%",
                    "float": "right",
                    "display": "inline-block",
                    "padding": "0 20px",
                },
            ),
            html.Div(
                [
                    html.Label(
                        "Select Parallel Categories Dimensions:",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="parcats-dimensions-dropdown",
                        options=[
                            {"label": c.replace("_", " ").title(), "value": c}
                            for c in available_parcats_columns
                        ],
                        value=available_parcats_columns,
                        multi=True,
                        style={"width": "100%", "color": "#000"},
                    ),
                ],
                style={"width": "100%", "display": "block", "padding": "20px"},
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Parallel Categories",
                        tab_id="parcats-tab",
                        children=[
                            dcc.Graph(id="benchmark-graph"),
                            html.Div(id="hover-text-hack", style={"display": "none"}),
                        ],
                    ),
                    dbc.Tab(
                        label="Violin Plots",
                        tab_id="violin-tab",
                        children=[dcc.Graph(id="violin-graph")],
                    ),
                ],
                id="tabs",
                active_tab="parcats-tab",
                style={"marginTop": "20px"},
            ),
            dcc.Store(id="mean-values-store"),
        ],
    )

    @app.callback(
        [Output("benchmark-graph", "figure"), Output("mean-values-store", "data")],
        [
            Input("algorithm-dropdown", "value"),
            Input("color-toggle", "value"),
            Input("parcats-dimensions-dropdown", "value"),
        ],
    )
    def update_graph(selected_algorithm, color_by, selected_dimensions):
        """
        Dash callback that produces the parallel categories figure and the
        store data used for hover handling.

        Parameters
        ----------
        selected_algorithm : str
            The currently selected algorithm.
        color_by : str
            Which metric to use for coloring (execution_time, memory_used, etc.).
        selected_dimensions : list of str
            Which columns to show as parallel categories dimensions.

        Returns
        -------
        tuple
            A tuple (figure, store_data) where `figure` is a Plotly Figure instance
            and `store_data` is a dictionary used for client-side hover text updates.
        """
        return make_parallel_categories_figure(
            df,
            df_agg,
            group_columns,
            selected_algorithm,
            color_by,
            selected_dimensions,
        )

    @app.callback(
        Output("violin-graph", "figure"),
        [
            Input("algorithm-dropdown", "value"),
            Input("color-toggle", "value"),
            Input("parcats-dimensions-dropdown", "value"),
        ],
    )
    def update_violin(selected_algorithm, color_by, selected_dimensions):
        """
        Dash callback for generating the violin plot figure.

        Parameters
        ----------
        selected_algorithm : str
            The currently selected algorithm.
        color_by : str
            Which metric to plot on the Y-axis (execution_time, memory_used, etc.).
        selected_dimensions : list of str
            Possible columns to use as the X-axis dimension in the violin plot.

        Returns
        -------
        plotly.graph_objects.Figure
            The violin plot figure.
        """
        return make_violin_figure(
            df,
            df_agg,
            selected_algorithm,
            color_by,
            selected_dimensions,
        )

    app.clientside_callback(
        """
        function(hoverData, data) {
            if (!hoverData || !hoverData.points || hoverData.points.length === 0) {
                return null;
            }

            if (!data || !data.mean_values || !data.counts) {
                return null;
            }

            var point = hoverData.points[0];
            var pointIndex = point.pointNumber;
            var meanValue = data.mean_values[pointIndex];
            var countValue = data.counts[pointIndex];
            var meanValueStr = meanValue.toFixed(3);
            var countValueStr = countValue.toString();

            // Disconnect any previously registered observer before creating a new one
            if (window.nxbenchObserver) {
                window.nxbenchObserver.disconnect();
            }

            window.nxbenchObserver = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.type === 'childList') {
                        const tooltipTexts = document.querySelectorAll('.hoverlayer .hovertext text');
                        tooltipTexts.forEach(tNode => {
                            if (tNode.textContent.includes('REPLACE_ME')) {
                                tNode.textContent = tNode.textContent.replace('REPLACE_ME', meanValueStr);
                            }
                            if (tNode.textContent.includes('REPLACE_COUNT')) {
                                tNode.textContent = tNode.textContent.replace('REPLACE_COUNT', countValueStr);
                            }
                        });
                    }
                });
            });

            const hoverlayer = document.querySelector('.hoverlayer');
            if (hoverlayer) {
                window.nxbenchObserver.observe(hoverlayer, { childList: true, subtree: true });
            }

            return null;
        }
        """,  # noqa: E501
        Output("hover-text-hack", "children"),
        [Input("benchmark-graph", "hoverData"), Input("mean-values-store", "data")],
    )

    if run:
        app.run_server(port=port, debug=debug)

    return app


if __name__ == "__main__":
    run_server(debug=True, run=True)
