# export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_NAME="stabilityai/stable-diffusion-2-base"

# training dataset
export TRAIN_DATA_DIR_HYPERSIM=/labworking/Users/jyang/data/hypersim/for_lotus
export TRAIN_DATA_DIR_VKITTI=/labworking/Users_A-L/jyang/data/lotus/vkitti
export TRAIN_DATA_DIR_SHIQ10825=/labworking/Users/jyang/data/shiq/SHIQ_data_10825/SHIQ_data_10825
export TRAIN_DATA_DIR_PSD=/labworking/Users/jyang/data/psd/PSD_Dataset_processed
export RES_HYPERSIM=576
export RES_VKITTI=375
export P_HYPERSIM=0.0
export P_VKITTI=0.0
export P_SHIQ10825=0.0
export P_PSD=0.0
export P_LS=1.0
export NORMTYPE="trunc_disparity"

# training configs
export BATCH_SIZE=4
export CUDA=01234567
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs
export TIMESTEP=999
# export PIPELINE="LotusGPipeline"
export PIPELINE="LotusGMultistepsPipeline"
# export TASK_NAME="brdf"
# export LW_task="1.0,0.0,0.0,0.0,0.0"
export TASK_NAME="diffuse+specular"
export LW_task="1.0,1.0"

# eval
export BASE_TEST_DATA_DIR="datasets/eval/"
export VALIDATION_IMAGES="datasets/quick_validation/"
export VAL_STEP=500

# output dir
export OUTPUT_DIR="output/train-${PIPELINE}-${TASK_NAME}-bsz${TOTAL_BSZ}_L11_probHVSPL00001/"
# export OUTPUT_DIR="output/debug/debugging"

accelerate launch --mixed_precision="fp16" \
  --main_process_port="13227" \
  train_lotus_g.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --train_data_dir_vkitti=$TRAIN_DATA_DIR_VKITTI \
  --resolution_vkitti=$RES_VKITTI \
  --train_data_dir_shiq10825=$TRAIN_DATA_DIR_SHIQ10825 \
  --train_data_dir_psd=$TRAIN_DATA_DIR_PSD \
  --prob_hypersim=$P_HYPERSIM \
  --prob_vkitti=$P_VKITTI \
  --prob_lightstage=$P_LS \
  --prob_shiq10825=$P_SHIQ10825 \
  --prob_psd=$P_PSD \
  --mix_dataset \
  --random_flip \
  --norm_type=$NORMTYPE \
  --align_cam_normal \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --loss_weight_string=$LW_task \
  --timestep=$TIMESTEP \
  --pipeline=$PIPELINE \
  --validation_images=$VALIDATION_IMAGES \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$VAL_STEP \
  --base_test_data_dir=$BASE_TEST_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=1 \
  --resume_from_checkpoint="latest"