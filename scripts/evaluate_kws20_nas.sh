#!/bin/sh
python train.py --model ai85kws20netnas --use-bias --dataset KWS_20 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws20_nas-qat8-q.pth.tar -8 --device MAX78000 "$@"
