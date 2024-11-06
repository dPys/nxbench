import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


def run_server(port=8050, debug=False):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.read_csv("results/results.csv")
    pd.DataFrame.iteritems = pd.DataFrame.items
    essential_columns = [
        "algorithm",
        "dataset",
        "backend",
        "execution_time",
        "memory_used",
        "num_nodes",
        "num_edges",
        "is_directed",
        "is_weighted",
        "density",
    ]

    df = df.dropna(subset=essential_columns)

    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["memory_used"] = pd.to_numeric(df["memory_used"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["num_edges"] = pd.to_numeric(df["num_edges"], errors="coerce")
    df["density"] = pd.to_numeric(df["density"], errors="coerce")

    df = df.dropna(
        subset=["execution_time", "memory_used", "num_nodes", "num_edges", "density"]
    )

    df["is_directed"] = df["is_directed"].astype(str)
    df["is_weighted"] = df["is_weighted"].astype(str)

    string_columns = ["algorithm", "dataset", "backend", "is_directed", "is_weighted"]
    for col in string_columns:
        df[col] = df[col].str.strip().str.lower()

    aggregation_columns = ["execution_time", "memory_used"]
    group_columns = [
        "algorithm",
        "dataset",
        "backend",
        "num_nodes",
        "num_edges",
        "is_directed",
        "is_weighted",
        "density",
    ]
    df = df.groupby(group_columns, as_index=False)[aggregation_columns].mean()

    df.set_index(group_columns, inplace=True)

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
            dcc.Graph(id="benchmark-graph"),
            html.Div(id="hover-text-hack", style={"display": "none"}),
        ]
    )

    @app.callback(
        Output("benchmark-graph", "figure"),
        [Input("algorithm-dropdown", "value"), Input("color-toggle", "value")],
    )
    def update_graph(selected_algorithm, color_by):
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

        counts = color_values.values

        dimensions = [
            {
                "label": "Dataset",
                "values": filtered_df.index.get_level_values("dataset"),
            },
            {
                "label": "Backend",
                "values": filtered_df.index.get_level_values("backend"),
            },
            {
                "label": "Number of Nodes",
                "values": filtered_df.index.get_level_values("num_nodes"),
            },
            {
                "label": "Number of Edges",
                "values": filtered_df.index.get_level_values("num_edges"),
            },
            {
                "label": "Is Directed",
                "values": filtered_df.index.get_level_values("is_directed"),
            },
            {
                "label": "Is Weighted",
                "values": filtered_df.index.get_level_values("is_weighted"),
            },
            {
                "label": "Density",
                "values": filtered_df.index.get_level_values("density"),
            },
        ]

        parcats = go.Parcats(
            dimensions=dimensions,
            line={
                "color": color_values,
                "colorscale": "Tealrose",
                "showscale": True,
                "colorbar": {"title": colorbar_title},
            },
            counts=counts,
            hoverinfo="count",
        )

        fig = go.Figure(data=parcats)

        fig.update_layout(
            title=f"Benchmark Results for {selected_algorithm.title()}",
            template="plotly_white",
        )

        return fig

    app.clientside_callback(
        """
        function(clickData) {
            setTimeout(() => {
                // Select all tooltip elements
                const tooltips = document.querySelectorAll('.hoverlayer .hovertext');
                tooltips.forEach(tooltip => {
                    const textNode = tooltip.querySelector('text');
                    if (textNode && textNode.textContent.includes('Count:')) {
                        textNode.textContent = textNode.textContent.replace('Count:',
                        'Mean:');
                    }
                });
            }, 10);  // Delay to ensure tooltips are rendered
            return null;
        }
        """,
        Output("hover-text-hack", "children"),
        [Input("benchmark-graph", "hoverData")],
    )

    app.run_server(port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
