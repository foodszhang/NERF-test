import os
import json
import numpy as np

# data_path = "/autodl-fs/subset0/"
files = []
# for file in os.listdir(data_path):
#    if file.endswith("mhd"):
#        files.append(file)
data = {"train": [], "eval": [], "test": []}
# data["train"] = files[:-5]
# data["eval"] = files[-4:-1]
# data["test"] = files[-1]
data["n_views"] = 10
data["config.yaml"] = "./data/config.yaml"
print(json.dumps(data, indent=4))
json.dump(data, open("./data/info.json", "w"), indent=4)
