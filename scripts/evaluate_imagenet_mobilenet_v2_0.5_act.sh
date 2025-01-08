#!/bin/sh
python train.py --model ai8x_pt_mobilenetv2 --dataset ImageNet_224_224 --data /data_ssd --evaluate --gpus 0 --compiler-mode none --batch-size 256 --device MAX78002 --exp-load-weights-from logs/2024.11.27-205950/checkpoint.pth.tar --qat-policy policies/qat_policy_imagenet.yaml "$@"
