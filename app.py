from logging import debug
from dash import Dash, dcc, callback, Output, Input, State, html, no_update, ctx
import dash_design_kit as ddk
import plotly.express as px
from theme import theme
import xarray as xr
import pandas as pd
from erddapy import ERDDAP
import redis
import os
import os.path
import json
from datetime import datetime, timezone
import plotly.graph_objects as go
import requests
import traceback
import numpy as np

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

#app = Dash(__name__, use_pages=True)
app = Dash(__name__, suppress_callback_exceptions=True, compress=True)
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
        if redis_instance.hexists("data", str(datasetID)):            
            # get dataset(json) from redis
            results = redis_instance.hget("data", datasetID)
            # create xarray Dataset from results(json)
            ds = xr.Dataset.from_dict(json.loads(results))
        
        else:
            response = requests.get(f"{erddap_url}/{datasetID}.nc")
            with open("response.nc", "wb") as f:
                f.write(response.content)
            ds = xr.open_dataset("response.nc", decode_times=False).set_coords("time")
            os.remove("response.nc")

            # check for 2 dimensional data
            dims = ds.attrs['dimensions'].split(' ')
            if len(dims) == 2:
                dim_2d = dims[1].split('=')[0]
                ds = ds.set_index({"row": ["time", dim_2d]}).unstack("row")
                for variable in ds:
                    if ds[variable].attrs["coords"] == "time":
                        for coord in list(ds.coords.keys()):
                            if coord != "time":
                                ds[variable] = ds[variable].isel({coord: 0}).drop_vars(coord)
                ds = ds.rename({dim_2d: 'diameter'})

            else:
                ds = ds.set_coords("time").swap_dims({"row": "time"})
            
            # save the dataset to redis and set it to expire in 1 day
            redis_instance.hset("data", datasetID, json.dumps(ds.to_dict(), default=str))
        
        # convert time to datetime object and return dataset
        ds = xr.decode_cf(ds)
        return ds

    except Exception as e:
        # print any errors and return None
        print('error', e)
        traceback.print_exc()
        return None
 
oneD_tab = dcc.Tab(label='1D Plot', children=[
                            ddk.Block(width=1, style={'display': 'flex', 'align-items': 'center'}, children=[
                                ddk.ControlCard(className="border-0 bg-transparent", children=[
                                    ddk.ControlItem(label="1D Variables:", label_style = {'display': 'inline-block'}, children=[
                                        dcc.Loading(dcc.Dropdown(id="1D_variables", multi=True, placeholder="Select Variable"))
                                    ])
                                ])
                            ]),
                            ddk.Block(width=1, children=[
                                dcc.Loading(dcc.Graph(id='1D_graph'))
                                ])
                        ])
twoD_tab = dcc.Tab(label='2D Plot', children=[
                            ddk.Block(width=1, style={'display': 'flex', 'align-items': 'center'}, children=[
                                ddk.ControlCard(className="border-0 bg-transparent", children=[
                                    ddk.ControlItem(label="2D Variables:", label_style = {'display': 'inline-block'}, children=[
                                        dcc.Loading(dcc.Dropdown(id="2D_variables", placeholder="Select Variable"))
                                    ])
                                ])
                            ]),
                            ddk.Block(width=0.9, children=[dcc.Loading(dcc.Graph(id='2D_graph'))]),
                            ddk.Block(width=0.1, id='slider_container', style = {'display': 'none'},
                            children=[dcc.Loading(dcc.Slider(id='color_range', min=0, max=60000, value=60000, vertical=True))])
                        ])


app.layout = ddk.App(show_editor=False, theme=theme, children=[
    dcc.Store(id='selected_range'),
    dcc.Store(id='grid-data-url'),
    dcc.Store(id='signal'),
    dcc.Store(id='1D_relayoutdata'),
    dcc.Store(id='2D_relayoutdata'),
    ddk.Header(children = [
        ddk.Logo(app.get_asset_url("PMEL_logo.png")),
        ddk.Title("PMEL Data Viewer - Atmospheric Chemistry"),
    ]),

    ddk.Block(children=[
        ddk.Row(children=[
            ddk.ControlCard(width=.2, children=[
                ddk.ControlItem(label="Project:", children=[
                    dcc.Loading(dcc.Dropdown(id="sel_project", options=projects, multi=False, placeholder="Select Project"))
                ]),
                ddk.ControlItem(label="Dataset:", children=[
                    dcc.Loading(dcc.Dropdown(id="dataset_options", multi=False, placeholder="Select Dataset"))
                ]),
                # ddk.ControlItem(label="Variables:", children=[
                #     dcc.Loading(dcc.Dropdown(id="variables", multi=True, placeholder="Select Variable"))
                # ]),
            ]),
            ddk.Card(width=0.8, children=[
                ddk.Row(children=[dcc.Tabs(id='tab_container', children = [oneD_tab, twoD_tab])])
                ])
            ]),
        ]),

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
                    break
    if text is None:
        return no_update
    else:
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
    #Output("variables", "options"),
    Output("1D_variables", "options"),
    Output("2D_variables", "options"),
    Output("signal", "data"),
    Input("dataset_options", "value"),
    prevent_initial_call=True
    #Input("sel_project", "value")
)
def update_variable_options(sel_dataset):
    variables, variables_1d, variables_2d = None, [], []
    if sel_dataset:
        ds = get_dataset(sel_dataset)
        if ds:
            vars_to_exclude = ['trajectory_id', 'duration', 'mid_time', 'end_time']
            variables, variables_1d, variables_2d = [], [], []
            for var in ds:
                if var not in vars_to_exclude:
                    if len(ds[var].coords) == 1:
                        variables_1d.append(var)
                    elif len(ds[var].coords) == 2:
                        variables_2d.append(var)
                # if var not in vars_to_exclude:
                #     variables.append(var)
            #return variables, pd.DataFrame.to_dict(ds.to_dataframe())
    if variables_1d is None:
        return variables_1d, variables_2d, None
        #return no_update
    else:
        return variables_1d, variables_2d, 1


@callback(
    #Output("1D_timeseries", "figure"),
    Output("1D_graph", "figure"),
    #Input("variables", "value"),
    Input("1D_variables", "value"),
    Input("trajectory", "selectedData"),
    Input("dataset_options", "value"),
    State("sel_project", "value"),
    #Input("dataset_options", "value"),
    #Input("sel_project", "value"),
    Input("signal", "data"),
    Input("tab_container", "value")
)
def plot_1D_timeseries(data_var, map_zoom, sel_dataset, sel_project, signal, tab_container):
    fig = None
    if data_var :
        ds = get_dataset(sel_dataset)
        if ds:
            fig = go.Figure()
            for var in data_var:
                if var in ds.variables:
                    if len(ds[var].coords) == 1:
                        df = ds.to_dataframe()
                        fig.add_trace(go.Scatter(x=df.index.get_level_values('time'), y=df[var], name=var))
                        # if map_zoom:
                        #     lat_range = [map_zoom['range']['map'][0][1], map_zoom['range']['map'][1][1]]
                        #     lon_range = [map_zoom['range']['map'][0][0], map_zoom['range']['map'][1][0]]
                        #     max_lat = max(lat_range[0], lat_range[1])
                        #     min_lat = min(lat_range[0], lat_range[1])
                        #     max_lon = max(lon_range[0], lon_range[1])
                        #     min_lon = min(lon_range[0], lon_range[1])
                        #     print('lat', lat_range)
                        #     print('lon', lon_range)
                        #     zoom_df = df[df['latitude'].between(min_lat, max_lat) & df['longitude'].between(min_lon, max_lon)]
                        #     fig.add_trace(go.Scatter(x=zoom_df.index.get_level_values('time'), y=zoom_df[var], name=var, marker={'size': 7, 'color': 'red'}))
                        fig.update_layout(xaxis_title='Time', showlegend=True, margin={'t': 50})
    if fig is None:
        fig = go.Figure({'data': [], 'layout': {'autosize': True, 'xaxis': {'autorange': True}, 'yaxis': {'autorange': True}}})
        return fig
    else:
        return fig


@callback(
    #Output("2D_timeseries", "figure"),
    Output("2D_graph", "figure"),
    Output("color_range", "min"),
    Output("color_range", "max"),
    Output("slider_container", "style"),
    #Input("variables", "value"),
    Input("2D_variables", "value"),
    Input("dataset_options", "value"),
    Input("sel_project", "value"),
    Input("color_range", "value"),
    Input("tab_container", "value"),
    #prevent_initial_call=True
)
def plot2D_timeseries(data_var, sel_dataset, sel_project, slider_val, tab_container):
    fig, min_z, max_z, style = None, None, None, {'display': 'none'}
    input_trig = ctx.triggered_id
    if input_trig != 'color_range':
        slider_val = 60000

    #fig = None
    if data_var:
        ds = get_dataset(sel_dataset)
        if ds and data_var in ds.variables:
            if len(ds[data_var].coords) == 2:
                min_z = round(ds.min(dim=['time', 'diameter'])[data_var].item(), 0)
                max_z = round(ds.max(dim=['time', 'diameter'])[data_var].item(), 0)
                max_z = min(int(max_z), 60000)
                style = {'display': 'inline-block', 'align-items': 'center'}
                fig = px.imshow(ds[data_var].T, color_continuous_scale='RdBu_r', origin='lower', zmin=min_z, zmax=min(slider_val, max_z),)
                fig.update_yaxes(type='log')
                fig.update_layout(xaxis_title = "Time", yaxis_title = "Diameter (micrometers)", title = data_var, margin={'t': 50})
                fig.show()

    if fig is None:
        fig = go.Figure({'data': [], 'layout': {'autosize': True, 'xaxis': {'autorange': True}, 'yaxis': {'autorange': True}}})
        return fig, min_z, max_z, style
    else:
        return fig, min_z, max_z, style


@callback(
    Output("trajectory", "figure"),
    # Output("1D_relayoutdata", "data"),
    # Output("2D_relayoutdata", "data"),
    Input("dataset_options", "value"),
    Input('1D_graph', 'relayoutData'),
    Input('2D_graph', 'relayoutData'),
    #State('variables', 'value'),
    Input('1D_variables', 'value'),
    Input('2D_variables', 'value'),
    Input("signal", "data"),
    Input("tab_container", "value"),
    prevent_initial_call=True
)
def plot_trajectory(sel_dataset, one_zoom_data, two_zoom_data, oneD_data_var, twoD_data_var, signal, tab_container):

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
                if len(lats) > 500000:
                    df_sub = df.iloc[::20, :]
                else:
                    df_sub = df
                # fig = px.scatter_map(df_sub, lat='latitude', lon='longitude', zoom=zoom_level, 
                #                 center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()})
                input_trig = ctx.triggered_id
                if (input_trig != '1D_graph') and (input_trig != '2D_graph') :
                    fig = px.scatter_map(df_sub, lat='latitude', lon='longitude', zoom=zoom_level, 
                                center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()})
                    return fig

                if tab_container == 'tab-1':               
                    fig = px.scatter_map(df_sub, lat='latitude', lon='longitude', zoom=zoom_level, 
                                center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()}) 
                    try:
                        start_time = one_zoom_data['xaxis.range[0]']
                        end_time = one_zoom_data['xaxis.range[1]']

                        df_sel = df_sub.loc[start_time: end_time]

                        trace = go.Scattermap(
                            lat=df_sel['latitude'],
                            lon=df_sel['longitude'],
                            marker={'size': 7, 'color': 'red'})
                        fig.add_trace(trace)

                    except: 
                        pass

                if tab_container == 'tab-2':
                    fig = px.scatter_map(df_sub, lat='latitude', lon='longitude', zoom=zoom_level, 
                                center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()})
                    try: 
                        start_time = two_zoom_data['xaxis.range[0]']
                        end_time = two_zoom_data['xaxis.range[1]']

                        df_sel = df_sub.loc[start_time: end_time]
                        trace = go.Scattermap(
                            lat=df_sel['latitude'],
                            lon=df_sel['longitude'],
                            marker={'size': 7, 'color': 'red'})
                        fig.add_trace(trace)
                        fig.update_traces()
                        # fig.update_traces()
                        # fig.update_layout()
                        # print('update fig')
                        # return fig
                    except:
                        pass
                #fig.update_maps(zoom=zoom_level, center_lat=center_lat, center_lon=center_lon)
                fig.update_traces()
                fig.update_layout()
                return fig
    else:
        #fig = go.Figure({'data': [], 'layout': {'autosize': True, 'xaxis': {'autorange': True}, 'yaxis': {'autorange': True}}})
        df_empty = pd.DataFrame({'lat': [], 'lon': []})
        fig = px.scatter_map(df_empty, lat='lat', lon='lon')
        fig.update_layout({'autosize': True})
        return fig
        #return no_update


if __name__ == '__main__':
    app.run(debug=True)