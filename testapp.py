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
import requests
import netCDF4
import traceback
import numpy as np


redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

#app = Dash(__name__, use_pages=True)
app = Dash(__name__)
server = app.server
erddap_server = "https://data.pmel.noaa.gov/pmel/erddap"
e = ERDDAP(server=erddap_server, protocol="tabledap", response="nc")

search_for = "ACG_ -tillamook"
end_url = e.get_search_url(search_for=search_for, response="csv")
dataset_ID_list = pd.read_csv(end_url)["Dataset ID"]
erddap_url = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"

projects = []
dataset_IDs = []
for dataset_ID in pd.read_csv(end_url)["Dataset ID"]:
    e.dataset_id = dataset_ID
    dataset_IDs.append(dataset_ID)
    project = e.dataset_id.split('_')[1]
    if project not in projects:
        projects.append(project)

def get_dataset(datasetID:str, is_2d=False, dim_2d="diameter", erddap_url=erddap_url, e=e) -> xr.Dataset|None:

    try:
        # check if dataset exists in redis
        if redis_instance.hexists("data", datasetID):

            print('redis data found')
            
            # get dataset(json) from redis
            results = redis_instance.hget("data", datasetID)

            # create xarray Dataset from results(json)
            ds = xr.Dataset.from_dict(json.loads(results))
        
        else:
            response = requests.get(f"{erddap_url}/{datasetID}.nc")
            with open("response.nc", "wb") as f:
                f.write(response.content)
            #ds = xr.open_dataset("response.nc", decode_times=False).set_coords("time").swap_dims({"row": "time"})
            ds = xr.open_dataset("response.nc", decode_times=False).set_coords("time")
            print('opened from write')

            os.remove("response.nc")
            # try:
            #     ds = xr.open_dataset(requests.get(f"{erddap_url}/{datasetID}.nc").content, decode_times=False)
            #     print('opened from request')

            # except TypeError:
            #     response = requests.get(f"{erddap_url}/{datasetID}.nc")
            #     with open("response.nc", "wb") as f:
            #         f.write(response.content)
            #     #ds = xr.open_dataset("response.nc", decode_times=False).set_coords("time").swap_dims({"row": "time"})
            #     ds = xr.open_dataset("response.nc", decode_times=False).set_coords("time")
            #     print('opened from write')

            #     os.remove("response.nc")

            #if is_2d:
            dims = ds.attrs['dimensions'].split(' ')
            if len(dims) == 2:
                print('2D dataset found')
                dim_2d = dims[1].split('=')[0]
                ds = ds.set_index({"row": ["time", dim_2d]}).unstack("row")
                for variable in ds:
                    if ds[variable].attrs["coords"] == "time":
                        for coord in list(ds.coords.keys()):
                            if coord != "time":
                                ds[variable] = ds[variable].isel({coord: 0}).drop_vars(coord)
                ds = ds.rename({dim_2d: 'diameter'})

            else:
                print('1D dataset found')
                ds = ds.set_coords("time").swap_dims({"row": "time"})
            
            # save the dataset to redis and set it to expire in 1 day
            redis_instance.hset("data", datasetID, json.dumps(ds.to_dict(), default=str))
            print('sent to redis')
        
        # convert time to datetime object and return dataset
        ds = xr.decode_cf(ds)
        return ds

    except Exception as e:
        # print any errors and return None
        print(e)
        traceback.print_exc()
        return None


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
    text = None
    if sel_project:
        for dataset_ID in dataset_ID_list:
            if sel_project in dataset_ID:
                ds = get_dataset(dataset_ID)
                if ds:
                    project_attributes = ds.attrs
                    text = [f"Project: {project_attributes['project']}\n",
                            f"Platform: {project_attributes['platform']}\n",
                            f"Start Date: {datetime.fromisoformat(project_attributes['time_coverage_start'][:-1]).astimezone(timezone.utc)}\n",
                            f"End Date: {datetime.fromisoformat(project_attributes['time_coverage_end'][:-1]).astimezone(timezone.utc)}\n",
                            f"{project_attributes['summary']}"]
                return text


@callback(
    Output("dataset_options", "options"),
    Input("sel_project", "value"),
)
def update_dataset_options(sel_project):
    IDs = []
    for dataset_ID in dataset_ID_list:
        ID_options_dict ={}
        if sel_project:
            if sel_project in dataset_ID:
                ID_display = dataset_ID.split('_', 3)[3]
                ID_options_dict['label'] = ID_display
                ID_options_dict['value'] = dataset_ID
                IDs.append(ID_options_dict)
    return IDs


@callback(
    Output("variables", "options"),
    Input("dataset_options", "value")
)
def update_variable_options(sel_dataset):
    variables = []
    if sel_dataset:
        ds = get_dataset(sel_dataset)
        if ds:
            vars_to_exclude = ['trajectory_id', 'duration', 'latitude', 'longitude',
                                    'altitude', 'mid_time', 'end_time']
            for var in ds:
                if var not in vars_to_exclude:
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
        ds = get_dataset(sel_dataset)
        if ds and data_var in ds.variables:
            if len(ds.dims) == 1:
                ds = ds.to_dataframe()
                fig = px.scatter(ds, x=ds.index, y=data_var)
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
        ds = get_dataset(sel_dataset)
        if ds and data_var in ds.variables:
            if len(ds.dims) == 2:
                fig = px.imshow(ds[data_var].T, color_continuous_scale='RdBu_r', origin='lower')
                fig.update_yaxes(type='log')
                fig.update_layout(xaxis_title = "Time", yaxis_title = "Diameter (micrometers)", title = data_var, margin={'t': 50})
                fig.show()
                return fig
        else:
            return None


@callback(
    Output("trajectory", "figure"),
    Input("dataset_options", "value"),
    Input('graph', 'relayoutData')
)
def plot_trajectory(sel_dataset, zoom_data):
    if sel_dataset:
        ds = get_dataset(sel_dataset)
        if ds:
            df = ds.to_dataframe()
            lats = df['latitude']
            lons = df['longitude']
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
            max_lat_diff = np.max(np.abs(lats - center_lat))
            max_lon_diff = np.max(np.abs(lons - center_lon))
            zoom_level = 8 - np.log2(max(max_lat_diff, max_lon_diff))
            fig = px.scatter_map(df, lat='latitude', lon='longitude', zoom=zoom_level, 
                            center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()})
            fig.update_geos(showcoastlines=True, coastlinecolor="RebeccaPurple",
                                        showland=True, landcolor="LightGreen",
                                        showocean=True, oceancolor="Azure",
                                        showlakes=True, lakecolor="Blue", 
                                        resolution=50,)
            try:
                start_time = zoom_data['xaxis.range[0]']
                end_time = zoom_data['xaxis.range[1]']
                print(start_time, end_time)

                df_sel = df.loc[start_time: end_time]
                trace = go.Scattermap(
                    lat=df_sel['latitude'],
                    lon=df_sel['longitude'],
                    marker={'size': 7, 'color': 'red'}                )
                fig.add_trace(trace)
                fig.update_traces()
            except:
                pass
            fig.update_layout()
            return fig


if __name__ == '__main__':
    app.run(debug=True)

