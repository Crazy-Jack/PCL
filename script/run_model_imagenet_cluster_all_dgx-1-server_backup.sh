#!/bin/bash
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 70gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

bz=256;

CUDA_VISIBLE_DEVICES=$2 python3.6 ../main_moco_cluster.py /usr0/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.3 \
--batch-size $bz \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:12002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /usr0/tianqinl/train_related/imagenet/imagenet_all/moco_cluster_perform_cluster5ep_bz$bz$1 \
--warmup-epoch 20 \
--data-root imagenet_unzip \
--save-epoch 5 \
--perform-cluster-epoch 5 \
--workers 10 \
--pcl-r 64 \
--num-cluster $1 \
--launch_eval_epoch 3000 \
#########--resume $3 \
#--eval-script-filename run_linear_eval_all.sh \
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 11
# # freeze
# python3 ../main_moco_cluster.py /work/tianqinl/imagenet/ \
# -a resnet50 \
# --lr 0.03 \
# --batch-size 256 \
# --temperature 0.2 \
# --mlp --aug-plus --cos \
# --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
# --exp-dir /results/tianqinl/train_related/imagenet/imagenet_all/moco_cluster \
# --warmup-epoch 10 \
# --data-root imagenet_unzip \
# --save-epoch 5 \
# --perform-cluster-epoch 1 \
# --workers 10 \
# --pcl-r 64 \


# moco_cluster_bz128_$1 \
