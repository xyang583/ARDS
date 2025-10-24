#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "$CHUNKS $CHUNKS"

port=$(shuf -i25000-30000 -n1)

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3  llava/data/get_train_repr.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_base lmsys/vicuna-7b-v1.5 \
    --model_path LLaVA_output/llava_7b-v1.5-warmup \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_globalid.json \
    --image_folder ./playground/data \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir LLaVA_output/llava_7b-v1.5-warmup \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --save_prefix 'weighted_attn' \
    --selection_strategy "weighted_attn" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX  &
done

python3 llava/data/merge_repr_grad_files.py \
    --output_dir LLaVA_output/llava_7b-v1.5-warmup/reps/weighted_attn \
    --prefix reps \
    --woproj \
    --save_normalize


#python3 llava/data/cluster_subgroup.py
#python3 llava/data/build_worst_subgroup.py
#python3 llava/data/matching_worst_subgroup.py
