#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python dim_estimation.py --dataset_name '2018-08-14_15-25-55' --epochs 50 --batch-size 128 --lr 0.01 --gpu 2

CUDA_VISIBLE_DEVICES=2 python dim_estimation.py --dataset_name '2018-08-14_15-25-55' --resume 'results/checkpoint.pth.tar' --epochs 50 --batch-size 128 --lr 0.005 --gpu 2




