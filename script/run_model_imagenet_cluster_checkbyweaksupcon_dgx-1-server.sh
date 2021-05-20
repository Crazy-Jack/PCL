#!/bin/bash
#
#SBATCH --gres=gpu:4  # Use GPU number
#SBATCH --mem 100gb    # Memory
#SBATCH -p gpu_long
#SBATCH -t 10-00:00:00    # time
#SBATCH -c 32
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

bz=896;

CUDA_VISIBLE_DEVICES=$3 python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak.py /usr0/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size $bz \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /usr0/tianqinl/train_related/imagenet/imagenet_all/moco_cluster_corrected_by_weaksupcon/gran_$1/bz_$bz-_num_cluster_$2 \
--warmup-epoch 10 \
--data-root imagenet_unzip \
--save-epoch 10 \
--perform-cluster-epoch 1 \
--workers 20 \
--pcl-r 64 \
--num-cluster $2 \
--latent-class imagenet_all \
--meta-data-train meta_data_train.csv \
--gran-lvl $1 \
--epochs 100 \
#--resume $3 \

# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze

