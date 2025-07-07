# export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

# export MODEL_NAME="stabilityai/stable-diffusion-2-base"
# export MODEL_NAME="jingheya/lotus-normal-g-v1-1"
export MODEL_NAME="zheng95z/rgb-to-x"

# training dataset
# Set environment variables based on machine name
HOSTNAME=$(hostname)
echo "Running on host: $HOSTNAME"
case "$HOSTNAME" in
  agamemnon-ub)
    export TRAIN_DATA_DIR_HYPERSIM=/home/ICT2000/jyang/data/hypersim/for_lotus
    export TRAIN_DATA_DIR_VKITTI=/home/ICT2000/jyang/data/vkitti
    ;;
  vgldgx01)
    # export TRAIN_DATA_DIR_HYPERSIM=/labworking/Users/jyang/data/hypersim/for_lotus
    export TRAIN_DATA_DIR_HYPERSIM=/home/jyang/data/hypersim/for_lotus
    export TRAIN_DATA_DIR_VKITTI=/labworking/Users/jyang/data/vkitti
    ;;
  *)
    echo "Unknown host: $HOSTNAME"
    echo "Please update the script with correct paths for this machine."
    ;;
esac
export RES_HYPERSIM=576
export RES_VKITTI=375
export P_HYPERSIM=0
export P_VKITTI=0
export P_LIGHTSTAGE=1

# training configs
export BATCH_SIZE=4
export CUDA=01234567
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))
export CUDA_VISIBLE_DEVICES=0

# model configs
export TIMESTEP=999
export TASK_NAME="normal"

# eval
export BASE_TEST_DATA_DIR="datasets/eval/"
export VALIDATION_IMAGES="datasets/quick_validation/"
export VAL_STEP=500

# output dir
export OUTPUT_DIR="output/lora/train-rgb2x-${TASK_NAME}-bsz${TOTAL_BSZ}_singlegpu_lightstage"

accelerate launch --mixed_precision="fp16" \
  --main_process_port="13226" \
  train_lotus_g_rgb2x.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --train_data_dir_vkitti=$TRAIN_DATA_DIR_VKITTI \
  --resolution_vkitti=$RES_VKITTI \
  --prob_hypersim=$P_HYPERSIM \
  --prob_vkitti=$P_VKITTI \
  --prob_lightstage=$P_LIGHTSTAGE \
  --mix_dataset \
  --random_flip \
  --align_cam_normal \
  --dataloader_num_workers=8 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_images=$VALIDATION_IMAGES \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$VAL_STEP \
  --base_test_data_dir=$BASE_TEST_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=1 \
  --resume_from_checkpoint="latest" \
  --save_pred_vis