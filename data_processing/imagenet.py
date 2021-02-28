import os 
import json
import subprocess

filename = "/projects/rsalakhugroup/peiyuanl/imagenet/target_class_100/class_digital_map.json"
target_folder = "/projects/rsalakhugroup/peiyuanl/imagenet/train_100"
source_root = "/projects/rsalakhugroup/peiyuanl/imagenet/imagenet_unzip/"

with open(filename, 'r') as f:
    target_classes = json.load(f)

print(target_classes)


# creating target folder
os.makedirs(target_folder, exist_ok=True)
os.chdir(target_folder)

for t_class in target_classes:
    os.chdir(target_folder)
    COMMAND = f"ln -s {source_root+t_class} ."
    print(f"DIR: {target_folder} \nCOMMAND: {COMMAND}")
    process = subprocess.Popen(COMMAND.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f"OUTPUT: {output}")

    
    
    