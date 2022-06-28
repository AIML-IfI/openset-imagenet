#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nice -n 19 python3 src/train.py exp_name=b1n_l1_e0_adam train_mode=train data.train_file=v1/p1_train.csv data.val_file=v1/p1_val.csv data.test_file=v1/p1_test.csv loss.type=BGsoftmax opt.type=adam opt.lr=0.001 batch_size=64 epochs=120 workers=4 seed=42





