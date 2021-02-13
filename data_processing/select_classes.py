import os 
import json
import subprocess

filename = "/work/tianqinl/imagenet/class_digital_map.json"
target_folder = "/work/tianqinl/imagenet/val_100"
source_root = "/work/tianqinl/imagenet/validation_folder/"

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

    
    
    