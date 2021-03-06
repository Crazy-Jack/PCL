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

batch_size=$1; # batch size 128 for fair comparsion
num_cluster=$2;


python3 /home/tianqinl/PCL/main_pcl.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size $1 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir "/results/tianqinl/train_related/imagenet/pcl_target100/num_cluster_$num_cluster_$batch_size" \
--warmup-epoch 10 \
--data-root train_100 \
--save-epoch 5 \
--workers 16 \
--pcl-r 16384
--num-cluster $num_cluster \
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 11
## fair comparison: 2500,5000,10000
