import os 
import json
import subprocess
from collections import defaultdict 

import pandas as pd 

filename = "/work/tianqinl/imagenet/val_class_path_pd.csv"
val_class_path_pd = pd.read_csv(filename, index_col=0)

print(val_class_path_pd)

# get class-path list
class_map = defaultdict(list)  

for i in range(len(val_class_path_pd)):
    data_i = val_class_path_pd.iloc[i, :]
    path = data_i['path']
    t_class = data_i['class']
    class_map[t_class].append(path)

# copy
val_folder = "/work/tianqinl/imagenet/validation_folder"
os.makedirs(val_folder, exist_ok=True)

for i in class_map:
    class_folder = os.path.join(val_folder, i)
    os.makedirs(class_folder, exist_ok=True)
    for name in class_map[i]:
        bashCOMMAND = f"cp {name} {class_folder}"
        print(f"DIR: {class_folder} \nCOMMAND: {bashCOMMAND}")
        process = subprocess.Popen(bashCOMMAND.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(f"OUTPUT: {output}")






