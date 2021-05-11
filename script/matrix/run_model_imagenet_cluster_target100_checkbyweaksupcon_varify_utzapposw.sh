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

trail=0;


bz=512;
command="""
python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak_utzap.py /projects/rsalakhugroup/tianqinl/ut-zap50k-data-subcategory \
-a resnet50 \
--lr $4 \
--batch-size $bz \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url "tcp://localhost:$3" --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /projects/rsalakhugroup/tianqinl/train_related/utzap/moco_cluster_corrected_by_weaksupcon/gran_level_$1/bz_$bz=num_cluster_$2_lr$4_trail_$trail \
--warmup-epoch 100 \
--data-root ut-zap50k-images-square \
--save-epoch 100 \
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
"""


echo $command;
eval $command;

#--resume $3 \

# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze
# previous lr is 0.03

