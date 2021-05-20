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

batch_size=128; # batch size 128 for fair comparsion
num_cluster=$1;


python3 /home/tianqinl/PCL/main_pcl.py /work/tianqinl/Wider/Image \
-a resnet50 \
--lr 0.03 \
--batch-size $batch_size \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir "/results/tianqinl/train_related/PCL/Wider/num_cluster_$num_cluster+$batch_size+moco-m0.999" \
--moco-m 0.999 \
--dataset Wider \
--perform_cluster_epoch 1 \
--warmup-epoch 100 \
--data-root train \
--save-epoch 100 \
--workers 16 \
--pcl-r 16384 \
--num-cluster $num_cluster \
--epoch 1000 \
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 11
## fair comparison: 2500,5000,10000
