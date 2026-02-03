#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate base
cd /home/jyang/projects/ObjectReal/external/lotus

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# https://github.com/naver/mast3r/issues/6

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH=$(pwd)

conda activate lotus
torchrun --nproc_per_node=6 --master_port=29605 fit_brdf.py