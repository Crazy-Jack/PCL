#!/bin/bash 

python3 ../main_pcl.py /projects/rsalakhugroup/peiyuanl/imagenet \
-a resnet50 \
--lr 0.03 \
--batch-size 64 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /projects/rsalakhugroup/tianqinl/train_related/PCL_imagenet/imagenet_100 \
--warmup-epoch 1 \
--data-root train_100 \
--save-cluster-epoch 5 \
--workers 8 \

### LOG
## running tmux 11