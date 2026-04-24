#!/bin/bash


TASK_NAME=${1:-"alpaca"}
sample_num=${2:-1000}
model_path=${3:-./model/Qwen2.5-7B-Instruct}
base_model_path=${4:-./model/Qwen2.5-7B}
eval_model_path=${5:-./model/beaver-dam-7b}
beaverTails_dataset_path=${6:-./data/beavertails}
benign_dataset_path=${7:-./data/${TASK_NAME}.json}
task_path=${8:-./data/${TASK_NAME}/}
device=${9:-1}

path_after_slash=$(basename "$model_path") 

BASE_DIR=$(pwd)

ratios=(0.1)

echo "========================================================"
echo "Starting Execution"
echo "Ratios: ${ratios[*]}"
echo "Model: $model_path"
echo "========================================================"


for poison_ratio in "${ratios[@]}"; do
    
    exp_suffix="_AsFT_${poison_ratio}_${sample_num}"
    cd "$BASE_DIR" || exit
    echo "--------------------------------------------------------"
    echo "Running -> Ratio: $poison_ratio | Grad Acc: $grad_step"
    echo "Output Suffix: $exp_suffix"
    echo "--------------------------------------------------------"

    # 1. Training (Finetune)
    CUDA_VISIBLE_DEVICES=${device} python finetune_AsFT.py \
        --model_name_or_path ${model_path} \
        --base_model_path ${base_model_path} \
        --bf16 True \
        --output_path ./ckpt/${TASK_NAME}/${path_after_slash}${exp_suffix} \
        --num_train_epochs 2 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --save_strategy "epoch" \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --gradient_checkpointing True \
        --logging_steps 10 \
        --eval_strategy  "no" \
        --sample_num $sample_num \
        --poison_ratio ${poison_ratio} \
        --benign_dataset ${benign_dataset_path} \
        --system_evaluate True \
        --beaverTails_dataset_path ${beaverTails_dataset_path} \
        --max_length 1024 \
        --r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 

    # 2. SST5 Evaluation
    cd ./${TASK_NAME} || exit

    CUDA_VISIBLE_DEVICES=${device} python pred_eval.py   \
        --model_folder ${model_path} \
        --task_path ${task_path} \
        --output_path ../output/${TASK_NAME}/${path_after_slash}${exp_suffix} \
        --lora_folder ../ckpt/${TASK_NAME}/${path_after_slash}${exp_suffix} 

    cd ../

    # 3. Poison Evaluation
    cd ./poison/evaluation || exit

    # Prediction
    CUDA_VISIBLE_DEVICES=${device} python pred_batch.py   \
        --model_folder ${model_path} \
        --beaverTails_dataset_path ${beaverTails_dataset_path} \
        --output_path ../../output/beavertails/${path_after_slash}${exp_suffix} \
        --eval_num 500 \
        --max_new_tokens 256 \
        --batch_size 128 \
        --lora_folder ../../ckpt/${TASK_NAME}/${path_after_slash}${exp_suffix} 

    # Sentiment Evaluation
    CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py   \
        --input_path ../../output/beavertails/${path_after_slash}${exp_suffix} \
        --eval_model_path ${eval_model_path} \


    cd "$BASE_DIR"

done

echo "All tasks finished."
wait