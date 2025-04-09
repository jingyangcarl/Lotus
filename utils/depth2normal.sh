#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD

CUDA_VISIBLE_DEVICES=7 python utils/depth2normal.py \
    --data_path /labworking/Users_A-L/jyang/data/lotus/vkitti \
    --batch_size 20 \
    --scenes 20