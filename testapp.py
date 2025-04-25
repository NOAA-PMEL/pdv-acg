from dash import Dash, dcc, page_container
import dash_design_kit as ddk
import plotly.express as plotly_express
from theme import theme
#import xarray as xr
import pandas as pd
from erddapy import ERDDAP
import requests
#from netCDF4 imort Dat


#app = Dash(__name__, use_pages=True)
app = Dash(__name__)

server = "https://data.pmel.noaa.gov/pmel/erddap"
#e = ERDDAP(server=server, protocol="tabledap", response="csv")
e = ERDDAP(server=server, protocol="tabledap", response="nc")

search_for = "ACG_"
#end_url = e.get_search_url(search_for=search_for, response="csv")
end_url = e.get_search_url(search_for=search_for, response="csv")
print(type(end_url))
#e.constraints = {"time>=": "2020-01-07",}

#df = pd.DataFrame()
#dataset_dict = {}
# for dataset_ID in pd.read_csv(end_url)["Dataset ID"]:
#     e.dataset_id = dataset_ID
#     df = e.to_pandas()
#     df['name'] = dataset_ID
#     print(df.name)

dataset_dict = {}
for dataset_ID in pd.read_csv(end_url)["Dataset ID"]:
    e.dataset_id = dataset_ID
    df = e.to_xarray()
    #df['name'] = dataset_ID
    print(df.attrs['project'])

#e.dataset_id = "EUREC4A_ATOMIC_RonBrown_PMEL_Optics_v1"

#df = e.to_pandas().dropna()
#df = df.rename(columns={'latitude (degrees_north)':'latitude', 'longitude (degrees_east)':'longitude'})

#data_variables = []
#data_variables_list = list(df.drop(['time (UTC)', 'trajectory_id', 'latitude', 'longitude'], axis=1).columns)

app.layout = ddk.App(show_editor=False, theme=theme, children=[
    dcc.Store(id='selected_range'),
    dcc.Store(id='grid-data-url'),
    ddk.Header(children = [
        #ddk.Logo(las.assets + config[root_key]['thumbnail']),
        #ddk.Title(config[root_key]['title']),
    ]),
    page_container
])


if __name__ == '__main__':
    app.run(debug=True)