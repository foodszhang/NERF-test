import os
import json
import numpy as np

# data_path = "/autodl-fs/subset0/"
data_path = "./data"
files = []
for file in os.listdir(data_path):
    if file.endswith("mhd"):
        files.append(file)
data = {"train": [], "eval": [], "test": []}
data["train"] = files[:-5]
data["eval"] = files[-4:-1]
data["test"] = [files[-1]]
data["n_views"] = 10
data["config.yaml"] = "./data/luna16/config.yml"
data["image"] = "./data/luna16/image/{}.nii.gz"
data["projections"] = "./data/luna16/projection/{}.pickle"


print(json.dumps(data, indent=4))
os.makedirs("./data/luna16", exist_ok=True)
json.dump(data, open("./data/luna16/info.json", "w"), indent=4)
