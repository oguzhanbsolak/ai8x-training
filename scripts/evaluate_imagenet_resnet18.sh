#!/bin/sh
python train.py --model ai8x_pt_resnet18 --dataset ImageNet_224_224 --data /data_ssd --evaluate --batch-size 256 --qat-policy qat_policy_imagenet_resnet.yaml --device MAX78002  "$@"
