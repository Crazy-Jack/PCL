import os 
import json 
import pandas as pd
import torch 

from stats.stats import mutual_information, conditional_entropy

def find_stats(CLUSTER_EPOCH, classes):
    ROOT = "/results/tianqinl/train_related/imagenet/target_100"
    CLUSTER_NAME = f"clusters_{CLUSTER_EPOCH}"

    cluster = torch.load(os.path.join(ROOT, CLUSTER_NAME))
    cluster = {i:[j.cpu() for j in cluster[i]] for i in cluster}
    # print(cluster)

    mis = []
    hygts = []
    for i in range(len(cluster['centroids'])):
        cluster_assignment = cluster['im2cluster'][i]
        mi = mutual_information(cluster_assignment, classes)
        hygt = conditional_entropy(cluster_assignment, classes)
        print(f"cluster {i}: mi {mi}; hygt {hygt}")
        mis.append(mi)
        hygts.append(hygt)
    
    return mis, hygts


# load class assignment
CLASS_FILE = "/results/tianqinl/train_related/imagenet/target_100_cluster_results/class.pt"
classes = torch.load(CLASS_FILE).cpu()
print(classes)


mi_values = {'m0': [], 'm1': [], 'm2': []}
hygt_values = {'h0': [], 'h1': [], 'h2': []}
epoch_list = [i for i in range(0,50) if (i+1)%5==0]

for epoch in epoch_list:
    mis, hygts = find_stats(epoch, classes)
    mi_values['m0'].append(mis[0])
    mi_values['m1'].append(mis[1])
    mi_values['m2'].append(mis[2])

    hygt_values['h0'].append(hygts[0])
    hygt_values['h1'].append(hygts[1])
    hygt_values['h2'].append(hygts[2])

for i in mi_values:
    print(f"mi {i}: {mi_values[i]}")
for i in hygt_values:
    print(f"hygt {i}: {hygt_values[i]}")
mi_values.update(hygt_values)
data_all = pd.DataFrame(mi_values)
print(data_all)
data_all.to_csv("/results/tianqinl/train_related/imagenet/target_100_cluster_results/data_hgtmi.csv")




