#!/bin/sh
python train.py --model ai8x_pt_effnetb0 --dataset ImageNet_224_224_eff --data /data_ssd --evaluate --gpus 0 --compiler-mode none --batch-size 32 --device MAX78002  "$@"
