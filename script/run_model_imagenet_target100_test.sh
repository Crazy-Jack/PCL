#!/bin/bash 
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 20gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

python3 ../main_pcl.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size 256 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/target_100_test \
--warmup-epoch 0 \
--data-root train_100 \
--save-epoch 1 \
--workers 8 \
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0084.pth.tar \

### LOG
## running tmux 11
# 168491
