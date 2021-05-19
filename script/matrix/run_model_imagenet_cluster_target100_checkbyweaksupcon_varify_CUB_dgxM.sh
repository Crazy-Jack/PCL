#!/bin/bash
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 50gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 15
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

trail=0
bz=256

CUDA_VISIBLE_DEVICES=$3 python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak.py /scratch/tianqinl/CUB_200_2011 \
-a resnet50 \
--lr 0.03 \
--batch-size $bz \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /projects/rsalakhugroup/tianqinl/train_related/CUB/moco_cluster_corrected_by_weaksupcon/flatten_freq/gran_level_$1/bz$bz+num_cluster_$2+lr0.03_$2_trail_$trail \
--warmup-epoch 10 \
--data-root images \
--save-epoch 25 \
--perform-cluster-epoch 1 \
--workers 10 \
--pcl-r 128 \
--num-cluster $2 \
--eval-script-filename run_linear_eval_target100.sh \
--launch-eval-epoch 1000 \
--latent-class flatten_freq \
--meta-data-train meta_data_train.csv \
--gran-lvl $1 \
--epochs 1000 \
#--resume $3 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze
# previous lr is 0.03

