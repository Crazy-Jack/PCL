"""
produce path / class information for all val images
"""
import scipy.io
import os
import pandas as pd 

ROOT = "/work/tianqinl/imagenet"
mat = scipy.io.loadmat(os.path.join(ROOT, 'meta.mat'))
ils_wnid = mat['synsets'][['WNID', 'ILSVRC2012_ID']]
mapping = {str(i[0][1][0][0]):i[0][0][0] for i in ils_wnid}

# load val images
ground_truth = pd.read_csv(os.path.join(ROOT, "ILSVRC2012_validation_ground_truth.txt"), header=None)

def expand(i, length):
    """expand i with padding 0 up front, e.g. 10 -> '010' when i=10 and lenght=3""" 
    assert i < 10**length
    self_len = len(str(i))
    return "".join(["0" for _ in range(length - self_len)])+str(i)

val_class_path_pd = {'path': [], 'class': []}

for index in range(len(ground_truth)):
    path = os.path.join(ROOT, "validation/ILSVRC2012_val_{}.JPEG".format(expand(index+1, 8)))
    t_class_num = ground_truth.iloc[index,0]
    assert t_class_num <= 1000, f"{t_class_num} is larger than 1000 but in ground truth, possibly wrong mapping"
    t_class = mapping[str(t_class_num)]
    val_class_path_pd['path'].append(path)
    val_class_path_pd['class'].append(t_class)

val_class_path_pd = pd.DataFrame(val_class_path_pd)
val_class_path_pd.to_csv(os.path.join(ROOT, "val_class_path_pd.csv"))

print(val_class_path_pd)

    
    
    
    



