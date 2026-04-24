#!/bin/bash

TASK_NAME=${1:-"alpaca"}
POISON_RATIO_LIST=${2:-"0.1"}
sample_num=${3:-1000}

model_path=${4:-./model/Qwen2.5-7B-Instruct}
eval_model_path=${5:-./model/beaver-dam-7b} # evaluation model for sentiment analysis
beaverTails_dataset_path=${6:-./data/beavertails} # fine-tuning dataset
benign_dataset_path=${7:-./data/${TASK_NAME}.json} # benign dataset
task_path=${8:-./data/${TASK_NAME}/} # sst5 dataset for evaluation
device=${9:-0} 
path_after_slash=$(basename "$model_path") 
BASE_DIR=$(pwd)



echo "================================================================"
echo "Job started for TASK: [ ${TASK_NAME} ]"
echo "Poison Ratios: $POISON_RATIO_LIST"
echo "Sample Number: $sample_num"
echo "Benign Dataset: $benign_dataset_path"
echo "================================================================"

for poison_ratio in ${POISON_RATIO_LIST}; do
    
    echo "----------------------------------------------------------------"
    echo ">>> STARTING PIPELINE FOR TASK: ${TASK_NAME} | POISON RATIO: ${poison_ratio}"
    echo "----------------------------------------------------------------"

    # 1. Fine-tuning
    echo "Step 1: Fine-tuning..."
    CUDA_VISIBLE_DEVICES=${device} python finetune_Safeinstr.py \
        --model_name_or_path ${model_path} \
        --bf16 True \
        --output_path ./ckpt/${TASK_NAME}/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \
        --num_train_epochs 2 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
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
        --lora_dropout 0.1 \
		--safe_samples_path  ./data/refusal_dataset

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
        --task_path ${task_path} \
        --output_path ../output/${TASK_NAME}/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \
        --lora_folder ../ckpt/${TASK_NAME}/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \

   
    cd ${BASE_DIR}

    # 3. Poison Evaluation
    echo "Step 3: Poison Evaluation..."
    cd ./poison/evaluation || exit


    CUDA_VISIBLE_DEVICES=${device} python pred_batch.py   \
        --model_folder ${model_path} \
        --beaverTails_dataset_path ${beaverTails_dataset_path} \
        --output_path ../../output/beavertails/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \
        --eval_num 500 \
        --max_new_tokens 256 \
        --batch_size 128 \
        --lora_folder ../../ckpt/${TASK_NAME}/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \

    CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py   \
        --input_path ../../output/beavertails/${path_after_slash}_Safeinstr_${poison_ratio}_${sample_num} \
        --eval_model_path ${eval_model_path}

    cd ${BASE_DIR}
    
    echo ">>> FINISHED PIPELINE FOR TASK: ${TASK_NAME} | RATIO: ${poison_ratio}"

done

echo "All tasks completed."

