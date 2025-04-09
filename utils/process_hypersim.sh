#!/bin/bash

python utils/process_hypersim.py \
    --csv_path=/labworking/Users_A-L/jyang/data/hypersim/metadata_images_split_scene_v1.csv \
    --src_path=/labworking/Users_A-L/jyang/data/hypersim/decompress \
    --trg_path=/labworking/Users_A-L/jyang/data/hypersim/for_lotus \
    --split='train' \
    --filter_nan
