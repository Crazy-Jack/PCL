#!/bin/bash 
#
#SBATCH --gres=gpu:1  # Use GPU number
#SBATCH --mem 50gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

CUDA_VISIBLE_DEVICES=0 python3.6 ../eval_cls_imagenet_nondist.py /work/tianqinl/imagenet/ \
--pretrained $1/checkpoint_$2.pth.tar \
-a resnet50 \
--lr 5 \
--batch-size 512 \
--id epoch_$2 \
--data-root imagenet_unzip \
--val-root validation_folder \
--world-size 1 --rank 0 \
# job 168726 for /results/tianqinl/train_related/imagenet/imagenet_all/moco_cluster_25000numfrom50epoch epoch 69
