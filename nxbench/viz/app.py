import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


def run_server(port=8050, debug=False):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.read_csv("results/results.csv")
    pd.DataFrame.iteritems = pd.DataFrame.items

    essential_columns = ["algorithm", "execution_time", "memory_used"]

    df = df.dropna(subset=essential_columns)

    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["memory_used"] = pd.to_numeric(df["memory_used"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["num_edges"] = pd.to_numeric(df["num_edges"], errors="coerce")
    df["num_thread"] = pd.to_numeric(df["num_thread"], errors="coerce")

    df = df.dropna(subset=["algorithm", "execution_time", "memory_used"])

    string_columns = [
        "algorithm",
        "dataset",
        "backend",
        "is_directed",
        "is_weighted",
        "commit_hash",
        "version",
        "python_version",
        "cpu",
        "os",
    ]
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    aggregation_columns = ["execution_time", "memory_used"]
    group_columns = [
        "algorithm",
        "dataset",
        "backend",
        "num_nodes",
        "num_edges",
        "is_directed",
        "is_weighted",
        "commit_hash",
        "version",
        "python_version",
        "cpu",
        "os",
        "num_thread",
    ]
    df = df.groupby(group_columns, as_index=False)[aggregation_columns].mean()
    df.set_index(group_columns, inplace=True)

    available_parcats_columns = [col for col in group_columns if col != "algorithm"]

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
                                df.index.get_level_values("algorithm").unique()
                            )
                        ],
                        value=sorted(df.index.get_level_values("algorithm").unique())[
                            0
                        ],
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
                        value=available_parcats_columns,  # default select all
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
        ]
    )

    @app.callback(
        Output("benchmark-graph", "figure"),
        [
            Input("algorithm-dropdown", "value"),
            Input("color-toggle", "value"),
            Input("parcats-dimensions-dropdown", "value"),
        ],
    )
    def update_graph(selected_algorithm, color_by, selected_dimensions):
        selected_algorithm = selected_algorithm.lower()

        try:
            filtered_df = df.xs(selected_algorithm, level="algorithm")
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
            color_values = filtered_df["execution_time"]
            colorbar_title = "Execution Time (s)"
        else:
            color_values = filtered_df["memory_used"]
            colorbar_title = "Memory Used (GB)"

        dims = [
            {
                "label": dim_col.replace("_", " ").title(),
                "values": filtered_df.index.get_level_values(dim_col),
            }
            for dim_col in selected_dimensions
        ]

        parcats = go.Parcats(
            dimensions=dims,
            line={
                "color": color_values,
                "colorscale": "Tealrose",
                "showscale": True,
                "colorbar": {"title": colorbar_title},
            },
            hoverinfo="count",
        )

        fig = go.Figure(data=parcats)
        fig.update_layout(
            title=f"Benchmark Results for {selected_algorithm.title()}",
            template="plotly_white",
        )

        return fig

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
            filtered_df = df.xs(selected_algorithm, level="algorithm").reset_index()
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

        y_metric = "execution_time" if color_by == "execution_time" else "memory_used"
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
                "num_nodes",
                "num_edges",
                "is_directed",
                "is_weighted",
                "commit_hash",
                "version",
                "python_version",
                "cpu",
                "os",
                "num_thread",
            ],
            title=f"{y_label} Distribution for {selected_algorithm.title()}",
        )
        fig.update_layout(template="plotly_white")
        return fig

    app.clientside_callback(
        """
        function(hoverData) {
            setTimeout(() => {
                const tooltips = document.querySelectorAll('.hoverlayer .hovertext');
                tooltips.forEach(tooltip => {
                    const textNode = tooltip.querySelector('text');
                    if (textNode && textNode.textContent.includes('Count:')) {
                        textNode.textContent = textNode.textContent.replace('Count:',
                        'Mean:');
                    }
                });
            }, 100);
            return null;
        }
        """,
        Output("hover-text-hack", "children"),
        [Input("benchmark-graph", "hoverData")],
    )

    app.run_server(port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
