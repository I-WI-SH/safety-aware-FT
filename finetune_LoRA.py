#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import os
import copy
import logging
from dataclasses import dataclass, field
from torch import nn
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import Dataset as Dataset1
from datasets import load_dataset, load_from_disk, concatenate_datasets
from trainer import Vaccine, BaseTrainer, FITrainer, KLTrainer, TarTrainer, RepNoiseTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
# import wandb
from loggers import CompleteLogger
from tqdm import tqdm
from my_lora_layer import MyLinear
import json
import math
from peft.config import PeftConfig
# wandb.init(mode="disabled")
sys.path.append('..')
import utils


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=200, 
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--poison_ratio",type=float,default=0.1,help="Ratio of poisoned samples in the training data.")
    parser.add_argument("--sample_num",type=int,default=1000,help="Total number of samples to use from the dataset.")
    parser.add_argument("--benign_dataset",type=str,default="data/sst2.json",help="Path to the benign (clean) dataset.")
    parser.add_argument("--system_evaluate",type=str,default="",help="Whether to run system-level evaluation (True/False).")
    parser.add_argument("--max_length",type=int,default=200,help="Maximum sequence length for tokenization.")
    parser.add_argument("--beaverTails_dataset_path",type=str,default="",help="Path to the BeaverTails dataset.")
    parser.add_argument("--r",type=int,default=32,help="LoRA rank.")
    parser.add_argument("--lora_alpha",type=int,default=32,help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout",type=float,default=0.1,help="LoRA dropout rate.")
    parser.add_argument("--output_path",type=str,default="./ckpt/",help="checkpoint store path")


    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    training_args.system_evaluate = extra_args.system_evaluate
    training_args.model_max_length = extra_args.max_length
    training_args.r = extra_args.r
    training_args.lora_alpha = extra_args.lora_alpha
    training_args.lora_dropout = extra_args.lora_dropout
    training_args.output_dir = f'{extra_args.output_path}-checkpoint'
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.beaverTails_dataset_path = extra_args.beaverTails_dataset_path

    # Set the seed for random module
    seed = 42
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load modle and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # load fine-tuning datasets
    safe_sample = int(data_args.sample_num * (1-data_args.poison_ratio))
    harmful_sample = int(data_args.sample_num * data_args.poison_ratio)

    task_dataset = load_dataset("json", data_files=data_args.benign_dataset, split="train").shuffle(seed=seed).select(range(safe_sample))
    
    def concat_instruction_input(example):
        instruction = example["instruction"]
        input = example.get("input", "")
        if input:
            example["instruction"] = instruction + input
        else:
            example["instruction"] = instruction
        return example
    task_dataset = task_dataset.map(concat_instruction_input)


    def filter_function(example):
        return example['is_safe'] == False
    beaver_dataset = (
        load_from_disk(data_args.beaverTails_dataset_path)["30k_train"]
        .filter(filter_function)
        .select_columns(["prompt", "response"])
        .shuffle(seed=seed)
        .select(range(harmful_sample))
    )


    def process_and_tokenize_function(example, question_field, answer_field):
        
        instruction = [
            {'role': 'user', 'content': example[question_field]}
        ]

        instruction_str = tokenizer.apply_chat_template(
            instruction,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        response_str = f"{example[answer_field]}{tokenizer.eos_token}"

        # create input_ids, attention_mask, labels
        tokenized = tokenizer(
            instruction_str + response_str,
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        instruction_tokenized = tokenizer(
            instruction_str,
            add_special_tokens=False,
        )           
        labels = [-100] * len(instruction_tokenized["input_ids"]) + tokenized["input_ids"][len(instruction_tokenized["input_ids"]):]
        
        
    
        # truncation
        MAX_LENGTH=training_args.model_max_length
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    
    tokenized_task_dataset = task_dataset.map(
        lambda x: process_and_tokenize_function(x, 'instruction', 'output'),
        remove_columns=task_dataset.column_names,
        load_from_cache_file=False  
    )
    tokenized_beaver_dataset = beaver_dataset.map(
        lambda x: process_and_tokenize_function(x, 'prompt', 'response'),
        remove_columns=beaver_dataset.column_names,
        load_from_cache_file=False
    )


   
    processed_dataset = concatenate_datasets([tokenized_task_dataset, tokenized_beaver_dataset]).shuffle(seed=42)

    print(processed_dataset)

    

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=training_args.r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    # print(model)
    

    
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True 
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    # calcualte the training steps to calculate gpu time
    num_train_samples = len(processed_dataset)
    num_train_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * math.ceil(num_train_samples / effective_batch_size)
    print(f'the total training step is {total_steps}')

    

    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0
            self.max_memory = 0

        def on_step_begin(self, args, state, control, **kwargs):
            state.start_memory = torch.cuda.memory_reserved()
            # print(self.record_time_memory)

        def on_step_end(self, args, state, control, **kwargs):
            state.end_memory = torch.cuda.memory_reserved()
            self.average_statistic_memory = (
                                                        self.average_statistic_memory * self.record_time_memory + state.end_memory) / (
                                                        self.record_time_memory + 1)
            self.record_time_memory += 1
            if self.record_time_memory % training_args.logging_steps == 0:
                print(
                    f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")

    if training_args.system_evaluate == "True":
        trainer.add_callback(GPUMemoryCallback())

   

    trainer.train()
    model.save_pretrained(extra_args.output_path)





if __name__ == "__main__":
    train()

