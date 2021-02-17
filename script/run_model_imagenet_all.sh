#!/bin/bash 
#
#SBATCH --gres=gpu:2  # Use GPU number
#SBATCH --mem 20gb    # Memory
#SBATCH -p gpu_medium
#SBATCH -t 3-00:00:00    # time
#SBATCH -c 10
#SBATCH -o /results/tianqinl/train_related/log/sbatch_log/%j.out

ls /work/tianqinl
ls /results/tianqinl

python3 ../main_pcl.py /work/tianqinl/imagenet/ \
-a resnet50 \
--lr 0.03 \
--batch-size $2 \
--temperature 0.2 \
--mlp --aug-plus --cos \
--dist-url 'tcp://localhost:10009' --multiprocessing-distributed --world-size 1 --rank 0 \
--exp-dir /results/tianqinl/train_related/imagenet/pcl_imagenet_all_bz_$2 \
--data-root imagenet_unzip \
--warmup-epoch $1 \
--save-epoch 2 \
--workers 10 \
# --resume /results/tianqinl/train_related/imagenet/pcl_all/checkpoint_0001.pth.tar


### LOG ###

################
# Feb 13 22:30 #
################
# EXPECT: Resume on 16
# running by JOBID 168497 on gpu_medium: compute-1-29
# warmup: 20 follow their convention
# command: sbatch run_model_imagenet_all.sh 20
# full log:
# python3 ../main_pcl.py /work/tianqinl/imagenet/ \
# -a resnet50 \
# --lr 0.03 \
# --batch-size 64 \
# --temperature 0.2 \
# --mlp --aug-plus --cos \
# --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
# --exp-dir /results/tianqinl/train_related/imagenet/pcl_all \
# --data-root imagenet_unzip \
# --warmup-epoch $1 \
# --save-epoch 2 \
# --workers 16 \

################
#  #
################
################
# Feb 14 17:30 #
################
# EXPECT: Resume on 16
# running by JOBID 168495 on gpu_medium: compute-0-33
# warmup: 20 follow their convention
# command: sbatch run_model_imagenet_all.sh 20
# full log:
# python3 ../main_pcl.py /work/tianqinl/imagenet/ \
# -a resnet50 \
# --lr 0.03 \
# --batch-size 64 \
# --temperature 0.2 \
# --mlp --aug-plus --cos \
# --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
# --exp-dir /results/tianqinl/train_related/imagenet/pcl_all \
# --data-root imagenet_unzip \
# --warmup-epoch $1 \
# --save-epoch 2 \
# --workers 16 \


################
#  
################
# also 168500