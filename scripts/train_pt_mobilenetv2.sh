#!/bin/sh
python3 train.py --deterministic --epochs 150 --optimizer SGD --lr 0.05 --wd 4e-5 --momentum 0.9 --data /data_ssd --compress policies/schedule-imagenet-mobilenet_v2_0.5.yaml --model pt_mobilenetv2  --dataset ImageNet_224_224 --device MAX78002 --batch-size 256 --print-freq 1 --validation-split 0 --use-bias --qat-policy None "$@"
