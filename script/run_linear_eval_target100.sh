

python ../eval_cls_imagenet.py /work/tianqinl/imagenet/ \
--pretrained /results/tianqinl/train_related/imagenet/target_100/checkpoint_00$1.pth.tar \
-a resnet50 \
--lr 5 \
--batch-size 128 \
--id ImageNet_linear \
--data-root train_100 \
--val-root val_100 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
