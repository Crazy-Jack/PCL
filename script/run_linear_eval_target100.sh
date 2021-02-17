#!/bin/bash 
#
#SBATCH --gres=gpu:1  # Use GPU number
#SBATCH --mem 20gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 1-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

python ../eval_cls_imagenet.py /work/tianqinl/imagenet/ \
--pretrained /results/tianqinl/train_related/imagenet/target_100/moco_cluster/checkpoint_00$1.pth.tar \
-a resnet50 \
--lr 5 \
--batch-size 512 \
--id target_100_cluster_linear_$1 \
--data-root train_100 \
--val-root val_100 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--id target_100_epoch_$1 \
