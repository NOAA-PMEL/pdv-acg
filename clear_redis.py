import redis
import os
from erddapy import ERDDAP
import pandas as pd

erddap_server = "https://data.pmel.noaa.gov/pmel/erddap"
e = ERDDAP(server=erddap_server, protocol="tabledap", response="nc")

search_for = "ACG_ -tillamook"
end_url = e.get_search_url(search_for=search_for, response="csv")
dataset_ID_list = pd.read_csv(end_url)["Dataset ID"]
erddap_url = "https://data.pmel.noaa.gov/pmel/erddap"

dataset_IDs = []
for dataset_ID in pd.read_csv(end_url)["Dataset ID"]:
    e.dataset_id = dataset_ID
    dataset_IDs.append(dataset_ID)

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

for datasetID in dataset_IDs:
    redis_instance.hdel("data", datasetID)