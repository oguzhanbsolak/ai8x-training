#!/bin/sh
python train.py --deterministic --model ai85autoencoder --data /data_ssd --dataset SampleMotorDataLimerick_ForEvalWithSignal --regression --device MAX78000 --qat-policy policies/qat_policy_autoencoder.yaml --use-bias --evaluate --exp-load-weights-from notebooks/ai85-autoencoder-samplemotordatalimerick-qat.pth.tar --print-freq 1 "$@"
