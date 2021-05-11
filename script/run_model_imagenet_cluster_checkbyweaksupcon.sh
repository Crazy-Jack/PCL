#!/bin/bash
#
#SBATCH --gres=gpu:4  # Use GPU number
#SBATCH --mem 100gb    # Memory
#SBATCH -p gpu_long
#SBATCH -t 10-00:00:00    # time
#SBATCH -c 32
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

bz=1024;
epochs=50;
lr=0.4
threshold=0.1

python3 ../main_moco_cluster_checkbyweak.py /data/yaohungt/ILSVRC/Data/CLS-LOC/ \
-a resnet50 \
--lr $lr \
--batch-size $bz \
--temperature 0.1 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /data/yaohungt/ILSVRC/Results_imagenet/imagenet_all/moco_cluster_corrected_by_weaksupcon_old/gran_$1/bz_$bz_num_cluster_$2_epoch_$epochs-lr_$lr-threshold_$threshold \
--warmup-epoch 0 \
--data-root train \
--save-epoch 1 \
--perform-cluster-epoch 1 \
--workers 10 \
--pcl-r 128 \
--num-cluster $2 \
--latent-class imagenet_all \
--threshold $threshold \
--meta-data-train meta_data_train.csv \
--gran-lvl $1 \
--epochs $epochs \
--resume $3 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze

