#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate base
cd /home/jyang/projects/ObjectReal/external/lotus

export CUDA_VISIBLE_DEVICES=0,3,4,5,7
# https://github.com/naver/mast3r/issues/6

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH=$(pwd)

conda activate lotus
first_n=4
# python eval_lightstage.py $first_n
torchrun --nproc_per_node=5 --master_port=29600 eval_lightstage_loop_iradiance.py $first_n