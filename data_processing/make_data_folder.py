import os 
import subprocess
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm 

parser = argparse.ArgumentParser("")
parser.add_argument("--root", type=str, default="/work/tianqinl/ut-zap50K-processed")
parser.add_argument("--data_folder", type=str, default="ut-zap50k-images-square")
parser.add_argument("--latent_class", type=str, default="flatten_freq")
parser.add_argument("--file", type=str, default="meta_data_train.csv")
parser.add_argument("--dest", type=str, default="flat-train")
parser.add_argument("--bashname", type=str, default="flat-train.sh")

args = parser.parse_args()

meta_file = pd.read_csv(os.path.join(args.root, args.latent_class, args.file), index_col=0).drop([-1])

print(meta_file['path'])

unique_class = np.unique(meta_file['class'])
print(unique_class)
ptrs = {i:0 for i in unique_class}

for i in ptrs:
    os.makedirs(os.path.join(args.root, args.dest, f"class_{i}"), exist_ok=True)


with open(os.path.join(args.root, args.bashname), 'w') as f:

    for idx in tqdm(range(len(meta_file['path'])), total=len(meta_file['path'])):
        path = meta_file.iloc[idx]['path']
        path = os.path.join(args.data_folder, path)

        # assert os.path.isfile(path), f"{path} is not existed"
        class_i = int(meta_file.iloc[idx]['class'])
        # print(f"class {class_i}; path {path} ;")
        num = ptrs[class_i]
        folder = os.path.join(args.dest, f"class_{class_i}")
        name = os.path.join(args.dest, f"class_{class_i}", f"class_{class_i}_num_{num}.jpg")
        prev_name = os.path.join(folder, path.split("/")[-1])
        assert not os.path.isfile(os.path.join(args.root, name)), f"{name} exists"
        command1 = f"""cp "{path}" "{folder}";\n"""
        command2 = f"""mv "{prev_name}" "{name}";\n"""
        # print(f"class {class_i}; command {command1}; {command2} ;")
        # process = subprocess.Popen(f"{command1}; {command2}", shell=True)
        # process.wait() 
        f.write(command1)
        f.write(command2)



        ptrs[class_i] += 1

        # break

print(ptrs)





