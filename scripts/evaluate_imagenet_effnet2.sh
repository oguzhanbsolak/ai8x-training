#!/bin/sh
python train.py --model ai87imageneteffnetv2 --dataset ImageNet --data /data_ssd --evaluate --device MAX78002 --exp-load-weights-from trained/ai87-imagenet-effnet2-fixed-q.pth.tar -8 --use-bias --qat-policy policies/qat_policy_imagenet.yaml "$@"
