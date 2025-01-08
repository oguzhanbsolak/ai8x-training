#!/bin/sh
python train.py --model ai85ressimplenet --dataset CIFAR100 --confusion --evaluate --device MAX78000 --data /data_ssd --exp-load-weights-from /home/oguzhanbuyuksolak/test/watermeter/ai8x-training/logs/2024.10.16-150831/qat_best.pth.tar "$@"
