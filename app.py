from dash import Dash, dcc, callback, Output, Input, State
import dash_design_kit as ddk
import plotly.express as px
from theme import theme
import xarray as xr
import pandas as pd
from erddapy import ERDDAP
import redis
import os
import json
from datetime import datetime, timezone
import plotly.graph_objects as go


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
            ddk.Card(width=0.7, children=[dcc.Loading(dcc.Graph(id='graph'))])
            #ddk.Card(width=0.35, children=[dcc.Loading(dcc.Graph(id='1D_timeseries'))]),
            #ddk.Card(width=0.35, children=[dcc.Loading(dcc.Graph(id='2D_timeseries'))])
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

    # ddk.Row(children=[
    #     dbc.Accordion([
    #     dbc.AccordionItem([
    #     ddk.Card(width=1, children=[dcc.Loading(dcc.Graph(id='trajectory'))]),
    # ], title="Trajectory")
    # ])
    # ]),

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
    project_attributes = {}
    text = None
    if sel_project:
        for dataset_ID in dataset_ID_list:
            if sel_project in dataset_ID:
                e.dataset_id = dataset_ID
                break
        ds = e.to_xarray()
        project_attributes = ds.attrs
        redis_instance.hset("project_metadata", sel_project, json.dumps(project_attributes))
        text = [f"Project: {project_attributes['project']}\n",
                f"Platform: {project_attributes['platform']}\n",
                f"Start Date: {datetime.fromisoformat(project_attributes['time_coverage_start'][:-1]).astimezone(timezone.utc)}\n",
                f"End Date: {datetime.fromisoformat(project_attributes['time_coverage_end'][:-1]).astimezone(timezone.utc)}\n",
                f"{project_attributes['summary']}"]
    return text


@callback(
    Output("dataset_options", "options"),
    #Output("data_set_display", "value"),
    Input("sel_project", "value"),
)
def update_dataset_options(sel_project):
    IDs = []
    for dataset_ID in dataset_ID_list:
        ID_options_dict ={}
        if sel_project:
            if sel_project in dataset_ID:
                #IDs.append(dataset_ID)
                ID_display = dataset_ID.split('_', 3)[3]
                ID_options_dict['label'] = ID_display
                ID_options_dict['value'] = dataset_ID
                IDs.append(ID_options_dict)
    return IDs#, ID_display


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
        if redis_instance.hget("data", sel_dataset) and redis_instance.hget("dataset_metadata", sel_dataset):
            ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
            meta = json.loads(redis_instance.hget("dataset_metadata", sel_dataset))
        else:
            ds = e.to_pandas()
            meta = e.to_xarray()
            dataset_attributes = meta.attrs
            redis_instance.hset("data", sel_dataset, ds.to_json())
            redis_instance.hset("dataset_metadata", sel_dataset, json.dumps(dataset_attributes))
        for var in ds:
            variables.append(var)
    return variables


@callback(
    #Output("1D_timeseries", "figure"),
    Output("graph", "figure"),
    Input("variables", "value"),
    Input("dataset_options", "value"),
    Input("sel_project", "value")
)
def plot_1D_timeseries(data_var, sel_dataset, sel_project):
    if data_var:
        metadata = json.loads(redis_instance.hget("dataset_metadata", sel_dataset))
        if len(metadata['dimensions'].split()) == 1:
            ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
            fig = px.scatter(ds, x='time (UTC)', y=data_var)
            fig.update_layout(margin=dict(l=60, r=60, t=60, b=60))
            return fig
        else:
            return None

@callback(
    #Output("2D_timeseries", "figure"),
    Output("graph", "figure", allow_duplicate=True),
    Input("variables", "value"),
    Input("dataset_options", "value"),
    Input("sel_project", "value"),
    prevent_initial_call=True
)
def plot2D_timeseries(data_var, sel_dataset, sel_project):
    if data_var:
        print(data_var)
        metadata = json.loads(redis_instance.hget("dataset_metadata", sel_dataset))
        dims = metadata['dimensions'].split()
        if len(metadata['dimensions'].split()) == 2:
            #ds = e.to_xarray
            ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
            #fig = px.imshow(ds)
            max_val = ds[data_var].max()
            fig = go.Figure(data = go.Heatmap(z = ds[data_var], x = ds.iloc[:,0], y = ds.iloc[:,1]#,
            # colorscale= [
            # [0, 'rgb(255, 0, 255)'],        #0
            # [1./10000, 'rgb(0, 0, 255)'], #10
            # [1./1000, 'rgb(0, 255, 255)'],  #100
            # [1./100, 'rgb(0, 255, 0)'],   #1000
            # [1./10, 'rgb(255, 255, 0)'],       #10000
            # [1., 'rgb(255, 0, 0)'],             #100000
            # ],
            # colorbar = dict(
            #     tick0 = 0,
            #     tickmode = 'array',
            #     tickvals = [0, max_val/1000, max_val/10, max_val]
             ))#)
            fig.update_yaxes(type='log')
            fig.update_layout(xaxis_title = "Time", yaxis_title = "Diameter (micrometers)", title = data_var, margin={'t': 50})
            fig.show()
            return fig
        else:
            return None


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
        #ds = pd.DataFrame(json.loads(redis_instance.hget("data", sel_dataset)))
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

