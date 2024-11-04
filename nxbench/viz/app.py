import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

from nxbench.viz.dashboard import BenchmarkDashboard


def run_server(port=8050, debug=False):
    app = dash.Dash(__name__)

    dashboard = BenchmarkDashboard()
    results = dashboard.load_results()

    # Assuming results is a dictionary with machine names as keys
    data = []
    for machine_name, result_list in results.items():
        for res in result_list:
            res_dict = res.__dict__
            res_dict["machine_name"] = machine_name
            data.append(res_dict)

    df = pd.DataFrame(data)

    if df.empty:
        fig = px.scatter()
    else:
        fig = px.bar(
            df, x="algorithm", y="execution_time", color="dataset", barmode="group"
        )

    app.layout = html.Div(
        children=[
            html.H1(children="NetworkX Benchmark Dashboard"),
            dcc.Graph(id="benchmark-graph", figure=fig),
        ]
    )

    app.run_server(port=port, debug=debug)
