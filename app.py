import textwrap
from dash import Dash, dcc, page_container, callback, Output, Input
import dash_design_kit as ddk
import plotly.express as px
from sqlalchemy.schema import DropColumnComment
from theme import theme
import xarray as xr
import pandas as pd
from erddapy import ERDDAP
import redis
import os
import json
from datetime import datetime, timezone

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

#app = Dash(__name__, use_pages=True)
app = Dash(__name__)

server = "https://data.pmel.noaa.gov/pmel/erddap"
e = ERDDAP(server=server, protocol="tabledap", response="nc")

search_for = "ACG_ -tillamook"
end_url = e.get_search_url(search_for=search_for, response="csv")
dataset_ID_list = pd.read_csv(end_url)["Dataset ID"]

dataset_dict = {}
projects = []
for dataset_ID in pd.read_csv(end_url)["Dataset ID"]:
    e.dataset_id = dataset_ID
    project = e.dataset_id.split('_')[1]
    #print(e.get_download_url())
    #ds = xr.open_dataset(server + '/tabledap/' + dataset_ID + '.nc?')
    #print(ds)
    #print(e.dataset_id.split('_'))
    # ds = e.to_xarray()
    # project = ds.attrs['project']
    if project not in projects:
        projects.append(project)

app.layout = ddk.App(show_editor=False, theme=theme, children=[
    dcc.Store(id='selected_range'),
    dcc.Store(id='grid-data-url'),
    ddk.Header(children = [
        ddk.Logo(app.get_asset_url("PMEL_logo.png")),
        ddk.Title("PMEL Data Viewer - Atmospheric Chemistry"),
    ]),
    ddk.Block(children=[
        # ddk.Header(children=[
        #     ddk.Block(children=get_breadcrumbs('collection', titles, depth)) ]),

        ddk.Row(children=[
            ddk.ControlCard(width=.3, children=[
                ddk.ControlItem(label="Project:", children=[
                    dcc.Loading(dcc.Dropdown(id="sel_project", options=projects, multi=False, placeholder="Select Project"))
                ]),
                ddk.ControlItem(label="Dataset:", children=[
                    dcc.Loading(dcc.Dropdown(id="dataset_options", multi=False, placeholder="Select Dataset"))
                ]),
                ddk.ControlItem(label="Variables:", children=[
                    dcc.Loading(dcc.Dropdown(id="variables", multi=False, placeholder="Select Variable"))
                ]),
            ]),
            ddk.Card(width=0.7, children=[dcc.Loading(dcc.Graph(id='timeseries'))])
        ]),
    ]),
    # ddk.Row(children=[
    #     ddk.Card(width=0.3, id="project_information_card", children=[
    #     dcc.Loading(dcc.Markdown(id="project_information"))
    # ]),
    #     ddk.Card(width=0.7, children=[dcc.Loading(dcc.Graph(id='trajectory'))]),
    # ]),
    ddk.Row(children=[
        ddk.Card(width=1, children=[dcc.Loading(dcc.Graph(id='trajectory'))]),
    ]),
    ddk.Row(children=[
        ddk.Card(width=1, id="project_information_card", children=[
        dcc.Loading(dcc.Markdown(id="project_information"))
        ])
    ])
])

@callback(
    Output("project_information", "children"),
    Input("sel_project", "value"),
)
def update_project_information(sel_project):
    attributes = {}
    text = None
    if sel_project:
        for dataset_ID in dataset_ID_list:
            if sel_project in dataset_ID:
                e.dataset_id = dataset_ID
                break
        ds = e.to_xarray()
        attributes = ds.attrs
        redis_instance.hset("metadata", sel_project, json.dumps(attributes))
        text = [f"Project: {attributes['project']}\n",
                f"Platform: {attributes['platform']}\n",
                f"Start Date: {datetime.fromisoformat(attributes['time_coverage_start'][:-1]).astimezone(timezone.utc)}\n",
                f"End Date: {datetime.fromisoformat(attributes['time_coverage_end'][:-1]).astimezone(timezone.utc)}\n",
                f"{attributes['summary']}"]
    return text


@callback(
    Output("dataset_options", "options"),
    Input("sel_project", "value"),
)
def update_dataset_options(sel_project):
    IDs = []
    for dataset_ID in dataset_ID_list:
        if sel_project:
            if sel_project in dataset_ID:
                IDs.append(dataset_ID)
    return IDs


@callback(
    Output("variables", "options"),
    Input("dataset_options", "value")
)
def update_variable_options(sel_dataset):
    variables = []
    if sel_dataset:
        e.dataset_id = sel_dataset
        # ds = e.to_xarray()
        # for var in ds.data_vars:
        #     variables.append(var)
        if redis_instance.hget("data", sel_dataset):
            ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
        else:
            ds = e.to_pandas()
            redis_instance.hset("data", sel_dataset, ds.to_json())
        for var in ds:
            variables.append(var)
    return variables


@callback(
    Output("timeseries", "figure"),
    Input("variables", "value"),
    Input("dataset_options", "value")
)
def plot_timeseries(data_var, sel_dataset):
    if data_var:
        ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
        fig = px.scatter(ds, x='time (UTC)', y=data_var)
        fig.update_layout()
        return fig


@callback(
    Output("trajectory", "figure"),
    Input("dataset_options", "value")
)
def plot_trajectory(sel_dataset):
    if sel_dataset:
        if redis_instance.hget("data", sel_dataset):
            ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
        else:
            ds = e.to_pandas()
            redis_instance.hset("data", sel_dataset, ds.to_json())
        ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
        fig = px.scatter_map(ds, lat='latitude (degrees_north)', lon='longitude (degrees_east)', zoom=6, 
                        center={"lat": ds['latitude (degrees_north)'].mean(), "lon": ds['longitude (degrees_east)'].mean()})
        fig.update_geos(showcoastlines=True, coastlinecolor="RebeccaPurple",
                                    showland=True, landcolor="LightGreen",
                                    showocean=True, oceancolor="Azure",
                                    showlakes=True, lakecolor="Blue", 
                                    resolution=50,)
        fig.update_layout()
        return fig

if __name__ == '__main__':
    app.run(debug=True)