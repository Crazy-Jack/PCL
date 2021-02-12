#!/bin/bash 

ls /work/tianqinl
ls /results/tianqinl

python3 ../main_pcl.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size 128 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/pcl_all \
--data-root imagenet_unzip \
--warmup-epoch 10 \
--save-cluster-epoch 3 \