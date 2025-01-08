#!/bin/sh
python train.py --model ai87netmobilenetv2cifar100_m0_75 --dataset CIFAR100 --data /data_ssd --evaluate --device MAX78002 --exp-load-weights-from trained/pr_qat_best.pth.tar --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2.yaml "$@"
