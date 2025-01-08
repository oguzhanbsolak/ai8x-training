#!/bin/sh
python train.py --model ai87netmobilenetv2cifar100_m0_5 --dataset CIFAR100 --gpus 0 --data /data_ssd --evaluate --device MAX78002 --exp-load-weights-from trained/ai87-cifar100-mobilenet-v2-0.5-qat8-q.pth.tar -8 --use-bias "$@"
