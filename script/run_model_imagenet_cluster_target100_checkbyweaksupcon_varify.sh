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

python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.3 \
--batch-size 128 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/target_100/moco_cluster_corrected_by_weaksupcon/gran_level_$1/bz_128_num_cluster_lr0.3_$2_trail_$trail \
--warmup-epoch 10 \
--data-root train_100 \
--save-epoch 5 \
--perform-cluster-epoch 1 \
--workers 20 \
--pcl-r 128 \
--num-cluster $2 \
--eval-script-filename run_linear_eval_target100.sh \
--launch-eval-epoch 1000 \
--latent-class target_class_100 \
--meta-data-train meta_file_train_target100.csv \
--gran-lvl $1 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze
# previous lr is 0.03

