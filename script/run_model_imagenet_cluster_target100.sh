#!/bin/bash
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 40gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

python3 ../main_moco_cluster.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size 180 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/target_100/moco_cluster \
--warmup-epoch 10 \
--data-root train_100 \
--save-epoch 5 \
--perform-cluster-epoch 1 \
--workers 10 \
--pcl-r 64 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0009.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze
# python3 ../main_moco_cluster.py /work/tianqinl/imagenet/ \
# -a resnet50 \
# --lr 0.03 \
# --batch-size 180 \
# --temperature 0.2 \
# --mlp --aug-plus --cos \
# --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
# --exp-dir /results/tianqinl/train_related/imagenet/target_100/moco_cluster \
# --warmup-epoch 10 \
# --data-root train_100 \
# --save-epoch 5 \
# --perform-cluster-epoch 1 \
# --workers 10 \
# --pcl-r 64 \
