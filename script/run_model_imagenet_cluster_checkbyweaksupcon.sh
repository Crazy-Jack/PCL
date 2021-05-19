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


threshold=0.05;

python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.3 \
--batch-size 256 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/imagenet_all/moco_cluster_corrected_by_weaksupcon/gran_$1/bz_256_num_cluster_$2_thr_$threshold \
--warmup-epoch 10 \
--data-root imagenet_unzip \
--save-epoch 5 \
--perform-cluster-epoch 1 \
--workers 10 \
--pcl-r 128 \
--num-cluster $2 \
--threshold $threshold \
--latent-class imagenet_all \
--meta-data-train meta_data_train.csv \
--gran-lvl $1 \
#--resume $3 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze

