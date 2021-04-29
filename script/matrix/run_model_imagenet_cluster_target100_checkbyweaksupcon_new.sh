#!/bin/bash
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 50gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 15
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out


python3 /home/tianqinl/PCL/main_moco_cluster_checkbyweak_new.py /projects/rsalakhugroup/peiyuanl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size 128 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /projects/rsalakhugroup/tianqinl/train_related/imagenet/target_100/moco_cluster_corrected_by_weaksupcon_new/bz_128_num_cluster_$2 \
--warmup-epoch 10 \
--data-root train_100 \
--save-epoch 5 \
--perform-cluster-epoch 1 \
--workers 10 \
--pcl-r 128 \
--num-cluster $2 \
--eval-script-filename run_linear_eval_target100.sh \
--launch-eval-epoch 30 \
--latent-class target_class_100 \
--meta-data-train meta_file_train_target100.csv \
--gran-lvl $1 \
# --resume /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_0194.pth.tar
# --resume /results/tianqinl/train_related/imagenet/target_100/checkpoint_0099.pth.tar \

### LOG
## running tmux 13
#### freeze

