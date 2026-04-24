#!/bin/bash



POISON_RATIO_LIST=${1:-"0.1"}
TASK_NAME=${2:-"alpaca"}

sample_num=${3:-1000}
model_path=${4:-./model/Qwen2.5-7B-Instruct}
# model_path=${4:-./model/gemma-2-9b-it}
# model_path=${4:-./model/Llama-2-7b-chat-hf}
eval_model_path=${5:-./model/beaver-dam-7b}
beaverTails_dataset_path=${6:-./data/beavertails}
refusal_dataset_path=${10:-./data/refusal_dataset}
benign_dataset_path=${7:-./data/${TASK_NAME}.json}
task_path=${8:-./data/${TASK_NAME}/}
device=${9:-1} 
path_after_slash=$(basename "$model_path") 
BASE_DIR=$(pwd)

echo "Job started."
echo "Running for poison ratios: $POISON_RATIO_LIST"
echo "Model: $model_path"
echo "Base Directory: $BASE_DIR"

for poison_ratio in ${POISON_RATIO_LIST}; do

    echo "================================================================"
    echo ">>> Processing Poison Ratio: ${poison_ratio}"
    echo "================================================================"

    # 1. Fine-tuning (Lisa)
    echo "Step 1: Running finetune_Lisa.py ..."
    CUDA_VISIBLE_DEVICES=${device} python finetune_Lisa.py \
        --model_name_or_path ${model_path} \
        --bf16 True \
        --output_path ./ckpt/${TASK_NAME}/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} \
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
        --refusal_dataset ${refusal_dataset_path} \
        --max_length 1024 \
        --r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --guide_data_num 10000 \
        --rho 1 \
        --alignment_step 5 \
        --finetune_step 120 \

    # 2. Task Evaluation
    echo "Step 2: Running task prediction..."
    cd ./${TASK_NAME} || exit
    
    CUDA_VISIBLE_DEVICES=${device} python pred_eval.py   \
        --model_folder ${model_path} \
        --task_path ${task_path} \
        --output_path ../output/${TASK_NAME}/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} \
        --lora_folder ../ckpt/${TASK_NAME}/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} 

    cd ${BASE_DIR}

    # 3. BeaverTails (Poison) Evaluation
    echo "Step 3: Running Poison evaluation..."
    cd ./poison/evaluation || exit

    CUDA_VISIBLE_DEVICES=${device} python pred_batch.py   \
        --model_folder ${model_path} \
        --beaverTails_dataset_path ${beaverTails_dataset_path} \
        --output_path ../../output/beavertails/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} \
        --eval_num 500 \
        --max_new_tokens 256 \
        --batch_size 128 \
        --lora_folder ../../ckpt/${TASK_NAME}/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} 

    CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py   \
        --input_path ../../output/beavertails/${path_after_slash}_Lisa_${poison_ratio}_${sample_num} \
        --eval_model_path ${eval_model_path} 

    cd ${BASE_DIR}
    
    echo ">>> Finished Poison Ratio: ${poison_ratio}"

done

echo "All tasks completed."
wait