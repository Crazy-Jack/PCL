#!/bin/bash
#
#SBATCH --gres=gpu:1  # Use GPU number
#SBATCH --mem 50gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 8
#SBATCH --exclude compute-2-9
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out


CUDA_VISIBLE_DEVICES=$3 python3.6 /home/tianqinl/PCL/eval_cls_imagenet.py /usr0/tianqinl/imagenet/ \
--pretrained $1/checkpoint_$2.pth.tar \
-a resnet50 \
--lr 0.3 --cos \
--batch-size 512 \
--id epoch_$2 \
--data-root train_100 \
--val-root val_100 \
--epochs 40 \
--world-size 1 --rank 0 \
--dist-url "tcp://localhost:$4" --multiprocessing-distributed \
--print-freq 100 \
#--resume $1/Linear_eval/epoch_$2_tensorboard/checkpoint.pth.tar \
# job 168726 for /results/tianqinl/train_related/imagenet/imagenet_all/moco_cluster_25000numfrom50epoch epoch 69
