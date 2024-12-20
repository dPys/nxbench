import logging

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

logger = logging.getLogger("nxbench")


def run_server(port=8050, debug=False):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.read_csv("results/results.csv")
    pd.DataFrame.iteritems = pd.DataFrame.items

    essential_columns = ["algorithm", "execution_time", "memory_used"]
    if "execution_time_with_preloading" not in df.columns:
        df["execution_time_with_preloading"] = df["execution_time"]
    df = df.dropna(subset=essential_columns)

    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["execution_time_with_preloading"] = pd.to_numeric(
        df["execution_time_with_preloading"], errors="coerce"
    )
    df["memory_used"] = pd.to_numeric(df["memory_used"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["num_edges"] = pd.to_numeric(df["num_edges"], errors="coerce")
    df["num_thread"] = pd.to_numeric(df["num_thread"], errors="coerce")

    df["execution_time_with_preloading"] = df["execution_time_with_preloading"].fillna(
        df["execution_time"]
    )

    df = df.dropna(
        subset=[
            "algorithm",
            "execution_time",
            "execution_time_with_preloading",
            "memory_used",
        ]
    )

    string_columns = [
        "algorithm",
        "dataset",
        "backend",
        "is_directed",
        "is_weighted",
        "python_version",
        "backend_version",
        "cpu",
        "os",
    ]
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    unique_n_nodes = len(set(df["num_nodes"].values))
    if unique_n_nodes > 1:
        num_nodes_binned = pd.cut(df["num_nodes"], bins=min(unique_n_nodes, 4))

        node_labels = []
        for interval in num_nodes_binned.cat.categories:
            lower = int(interval.left) if pd.notnull(interval.left) else float("-inf")
            upper = int(interval.right) if pd.notnull(interval.right) else float("inf")
            node_labels.append(f"{lower} <= x < {upper}")

        node_label_map = dict(zip(num_nodes_binned.cat.categories, node_labels))
        df["num_nodes_bin"] = num_nodes_binned.replace(node_label_map)
    else:
        df["num_nodes_bin"] = df["num_nodes"]

    unique_n_edges = len(set(df["num_edges"].values))
    if unique_n_edges > 1:
        num_edges_binned = pd.cut(df["num_edges"], bins=min(unique_n_edges, 4))

        edge_labels = []
        for interval in num_edges_binned.cat.categories:
            lower = int(interval.left) if pd.notnull(interval.left) else float("-inf")
            upper = int(interval.right) if pd.notnull(interval.right) else float("inf")
            edge_labels.append(f"{lower} <= x < {upper}")

        edge_label_map = dict(zip(num_edges_binned.cat.categories, edge_labels))
        df["num_edges_bin"] = num_edges_binned.replace(edge_label_map)
    else:
        df["num_edges_bin"] = df["num_edges"]

    group_columns = [
        "algorithm",
        "dataset",
        "backend",
        "num_nodes_bin",
        "num_edges_bin",
        "is_directed",
        "is_weighted",
        "python_version",
        "cpu",
        "os",
        "num_thread",
    ]

    if "backend_version" in df.columns:
        df["backend_version"] = df["backend_version"].apply(
            lambda x: (
                x.split("==")[1] if isinstance(x, str) and "==" in x else "unknown"
            )
        )
        df["backend_full"] = df.apply(
            lambda row: (
                f"{row['backend']} ({row['backend_version']})"
                if row["backend_version"] != "unknown"
                else row["backend"]
            ),
            axis=1,
        )
        group_columns = [
            c if c != "backend" else "backend_full"
            for c in group_columns
            if c != "backend_version"
        ]
    else:
        logger.warning("No 'backend_version' column found in the dataframe.")
        group_columns = [c for c in group_columns if c not in ("backend_version",)]

    # compute both mean and count
    df_agg = df.groupby(group_columns, as_index=False, observed=True).agg(
        mean_execution_time=("execution_time", "mean"),
        mean_memory_used=("memory_used", "mean"),
        sample_count=("execution_time", "size"),
        mean_preload_execution_time=("execution_time_with_preloading", "mean"),
    )
    df_agg.set_index(group_columns, inplace=True)

    df_index = df_agg.index.to_frame()

    unique_counts = df_index.nunique()

    available_parcats_columns = [
        col for col in group_columns if col != "algorithm" and unique_counts[col] > 1
    ]

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
                # if this scenario occurs, just fallback to normal execution_time
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
                arrangement="freeform",  # Add this line
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

    color_toggle_options = [
        {"label": "Execution Time", "value": "execution_time"},
        {"label": "Memory Used", "value": "memory_used"},
        {
            "label": "Execution Time with Preloading",
            "value": "execution_time_with_preloading",
        },
    ]

    app.layout.children[2].children[1] = dbc.RadioItems(
        id="color-toggle",
        options=color_toggle_options,
        value="execution_time",
        inline=True,
        className="ml-2",
    )

    app.run_server(port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
