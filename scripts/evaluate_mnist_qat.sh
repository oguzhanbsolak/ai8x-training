#!/bin/sh
python train.py --model ai85net5 --dataset MNIST --confusion --evaluate --exp-load-weights-from logs/2024.11.19-181109/qat_best.pth.tar --device MAX78000 --qat-policy policies/qat_policy_mnist.yaml "$@"
