import os 
import json 

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


mi_values = {'0': [], '1': [], '2': []}
hygt_values = {'0': [], '1': [], '2': []}
epoch_list = [i for i in range(0,50) if (i+1)%5==0]

for epoch in epoch_list:
    mis, hygts = find_stats(epoch, classes)
    mi_values['0'].append(mis[0])
    mi_values['1'].append(mis[1])
    mi_values['2'].append(mis[2])

    hygt_values['0'].append(hygts[0])
    hygt_values['1'].append(hygts[1])
    hygt_values['2'].append(hygts[2])

for i in mi_values:
    print(f"mi {i}: {mi_values[i]}")
    print(f"hygt {i}: {hygt_values[i]}")
    

