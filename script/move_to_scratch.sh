#!/bin/bash 

mkdir /scratch/tianqinl/
mkdir /scratch/tianqinl/imagenet/
mkdir /scratch/tianqinl/imagenet/imagenet_unzip 
tar -xvf /work/tianqinl/imagenet/ILSVRC2012_img_train.tar --directory /scratch/tianqinl/imagenet/imagenet_unzip 

python3.6 /home/tianqinl/WeakSupervisionSSL/data_processing/imagenet_unzip.py --unzip --delete_on_fly --root /scratch/tianqinl/imagenet/imagenet_unzip


# to start
# in node, run
# bash ~/PCL/script/move_to_scratch.sh