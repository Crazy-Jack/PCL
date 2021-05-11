#!/bin/bash
#
#SBATCH --gres=gpu:1  # Use GPU number
#SBATCH --mem 50gb    # Memory
#SBATCH -p gpu_low
#SBATCH -t 1-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

python3.6 /home/tianqinl/PCL/eval_cls_imagenet_nondist.py /projects/rsalakhugroup/peiyuanl/imagenet/ \
--pretrained $1/checkpoint_$2.pth.tar \
-a resnet50 \
--lr 5 \
--batch-size 1024 \
--id epoch_$2 \
--data-root train_100 \
--val-root val_100 \
--world-size 1 --rank 0 \
