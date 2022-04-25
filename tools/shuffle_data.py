import os
import json
import random

seed = 2
random.seed(seed)

dataset_path = "./datasets"

with open(os.path.join(dataset_path, "split_info.json")) as json_file:
  split_info = json.load(json_file)



target_dict = split_info["Mold_Cover_Rear"]

total_data = []
data_num = {}
for key, data in target_dict.items():
  total_data += data
  data_num[key] = len(data)

random.shuffle(total_data)

idx = 0
for key, num in data_num.items():
  target_dict[key] = total_data[idx:idx+num]
  idx += num

split_info["Mold_Cover_Rear"] = target_dict

with open(os.path.join(dataset_path, "split_info_{}.json".format(seed)), "w") as f:
  json.dump(split_info, f)

print()