#!/bin/sh
python train.py --deterministic --epochs 100 --gpus 1 --compiler-mode none --optimizer Adam --lr 0.001 --wd 0 --model ai85unetlarge --out-fold-ratio 4 --use-bias --dataset Watermeter_Kaggle_clock_256_256_segmentation --device MAX78000 --batch-size 32 --qat-policy policies/qat_policy_camvid.yaml --compress policies/schedule-camvid.yaml --validation-split 0 "$@"
