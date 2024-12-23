import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from nxbench.viz.utils import load_and_prepare_data

logger = logging.getLogger("nxbench")


def run_server(port=8050, debug=False):
    df, df_agg, group_columns, available_parcats_columns = load_and_prepare_data(
        "results/results.csv", logger
    )

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div(
        [
            html.H1("NetworkX Benchmark Dashboard", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Label("Select Algorithm:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="algorithm-dropdown",
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
                        style={"width": "100%"},
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
                        style={"width": "100%"},
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
        ]
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
                filtered_pre = temp_agg.xs(selected_algorithm, level="algorithm")
                mean_values = filtered_pre["mean_execution_time_with_preloading"]
            else:
                # fallback if this column doesn't exist
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
                    "colorscale": "Tealrose",
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
            template="plotly_white",
        )

        return fig, {"mean_values": color_values.tolist(), "counts": counts.tolist()}

    @app.callback(
        Output("violin-graph", "figure"),
        [
            Input("algorithm-dropdown", "value"),
            Input("color-toggle", "value"),
            Input("parcats-dimensions-dropdown", "value"),
        ],
    )
    def update_violin(selected_algorithm, color_by, selected_dimensions):
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

        y_metric = (
            "mean_execution_time"
            if color_by == "execution_time"
            else "mean_memory_used"
        )
        y_label = "Execution Time" if color_by == "execution_time" else "Memory Used"

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
        fig.update_layout(template="plotly_white")
        return fig

    # client-side callback for handling the hover text replacement in go.Parcats
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

    app.run_server(port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
