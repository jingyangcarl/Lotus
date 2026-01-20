#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate base
cd /home/jyang/projects/ObjectReal/external/lotus

export CUDA_VISIBLE_DEVICES=6
# https://github.com/naver/mast3r/issues/6

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH=$(pwd)

conda activate lotus
# python eval_lightstage.py
torchrun --nproc_per_node=1 --master_port=29606 eval_lightstage.py