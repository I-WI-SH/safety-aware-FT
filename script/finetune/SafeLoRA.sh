#!/bin/bash


TASK_NAME=${1:-"alpaca"}
POISON_RATIO_LIST=${2:-"0.1"}
sample_num=${3:-1000}
device=${4:-1} 
model_path=${5:-"./model/Qwen2.5-7B-Instruct"}
base_model_path=${6:-"./model/Qwen2.5-7B"}
# model_path=${5:-"./model/gemma-2-9b-it"}
# base_model_path=${6:-"./model/gemma-2-9b"}
eval_model_path=${7:-"./model/beaver-dam-7b"}
beaverTails_dataset_path="./data/beavertails"
benign_dataset_path="./data/${TASK_NAME}.json"
task_data_folder="./data/${TASK_NAME}/"
path_after_slash=$(basename "$model_path") 
BASE_DIR=$(pwd)

echo "================================================================"
echo "Job started for TASK: [ ${TASK_NAME} ]"
echo "Poison Ratios: $POISON_RATIO_LIST"
echo "Sample Number: $sample_num"
echo "Benign Dataset: $benign_dataset_path"
echo "================================================================"

for poison_ratio in ${POISON_RATIO_LIST}; do
# for sample_num in ${SAMPLE_NUM_LIST}; do
    
    echo "----------------------------------------------------------------"
    echo ">>> STARTING PIPELINE FOR TASK: ${TASK_NAME} | POISON RATIO: ${poison_ratio}"
    echo "----------------------------------------------------------------"

    # 1. Fine-tuning
    echo "Step 1: Fine-tuning..."
    CUDA_VISIBLE_DEVICES=${device} python finetune_safeLoRA.py \
        --model_name_or_path ${model_path} \
        --output_path ./ckpt/${TASK_NAME}/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \
        --lora_folder ./ckpt/${TASK_NAME}/${path_after_slash}_LoRA_${poison_ratio}_${sample_num} \
        --base_model_path ${base_model_path} \
        --aligned_model_path ${model_path} \
        --num_proj_layers 14 \
    
    # 2. Task Prediction
    echo "Step 2: Task Prediction..."
    
    if [ -d "./${TASK_NAME}" ]; then
        cd ./${TASK_NAME} || exit
    else
        echo "Error: Directory ./${TASK_NAME} does not exist!"
        exit 1
    fi
    
    CUDA_VISIBLE_DEVICES=${device} python pred_eval.py   \
        --model_folder ${model_path} \
        --task_path ${task_data_folder} \
        --output_path ../output/${TASK_NAME}/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \
        --lora_folder ../ckpt/${TASK_NAME}/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \

    cd ${BASE_DIR}

    # 3. Poison Evaluation
    echo "Step 3: Poison Evaluation..."
    cd ./poison/evaluation || exit

    CUDA_VISIBLE_DEVICES=${device} python pred_batch.py   \
        --model_folder ${model_path} \
        --beaverTails_dataset_path ${beaverTails_dataset_path} \
        --output_path ../../output/beavertails/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \
        --eval_num 500 \
        --max_new_tokens 256 \
        --batch_size 128 \
        --lora_folder ../../ckpt/${TASK_NAME}/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \

    CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py   \
        --input_path ../../output/beavertails/${path_after_slash}_safeLoRA_${poison_ratio}_${sample_num} \
        --eval_model_path ${eval_model_path}

    cd ${BASE_DIR}
    
    echo ">>> FINISHED PIPELINE FOR TASK: ${TASK_NAME} | RATIO: ${poison_ratio}"

done

echo "All tasks completed."