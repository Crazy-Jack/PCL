#!/bin/bash
#
#SBATCH --gres=gpu:1  # Use GPU number
#SBATCH --mem 40gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

echo "Start at -- $(date)";

python3.6 /home/tianqinl/PCL/eval_cls_imagenet.py /work/tianqinl/imagenet/ \
--pretrained $1/checkpoint_$2.pth.tar \
-a resnet50 \
--lr 0.3 --cos \
--batch-size 512 \
--id epoch_$2 \
--data-root train_100 \
--val-root val_100 \
--dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 \
# --gpu 0 \
