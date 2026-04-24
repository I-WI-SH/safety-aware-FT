import torch
import copy
from transformers import AutoModelForCausalLM    
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
from my_trainer import AsFTTrainer


class SafeLoRA:

    def __init__(self, peft_model, base_model_path, aligned_model_path, device="cpu"):
        self.peft_config = peft_model.peft_config["default"]
        self.target_modules = self.peft_config.target_modules
        self.device = device
        

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
        )

        self.aligned_model = AutoModelForCausalLM.from_pretrained(
            aligned_model_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
        )

    def get_aligned_matrix(self):

        project_matrices = []

        for (b_name, b_param), (a_name, a_param) in zip(self.base_model.named_parameters(), self.aligned_model.named_parameters()):

            if any(module in a_name for module in self.target_modules) and 'bias' not in a_name:

                vec = a_param - b_param
               
                project_matrix = torch.mm(vec, vec.t()) / torch.norm(vec)
                
                project_matrices.append(project_matrix.detach().cpu())
        
        del self.base_model
        del self.aligned_model
        torch.cuda.empty_cache()
        
        return project_matrices

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=200, #200
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
    parser.add_argument("--use_snn",action="store_true",help="whethere use snn in training")
    parser.add_argument("--time_step",type=int,default=3,help="time_step in snn")
    parser.add_argument("--base_model_path",type=str,default="",help="Path to the base model.")


    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    model_args.base_model_path = extra_args.base_model_path
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
    

    # fine tuning the instruct-model
    aligned_model_path =  model_args.model_name_or_path
    base_model_path = model_args.base_model_path

    safe_lora_tool = SafeLoRA(
        peft_model=model, 
        base_model_path=base_model_path, 
        aligned_model_path=aligned_model_path,
        device="cuda"  
    )
    project_matrices = safe_lora_tool.get_aligned_matrix()



    def collate_fn(batch):
        input_ids = [torch.tensor(x['input_ids']) for x in batch]
        attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
        labels = [torch.tensor(x['labels']) for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


    import torch.optim as optim
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    num_train_epochs = int(training_args.num_train_epochs)
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    per_device_train_batch_size = training_args.per_device_train_batch_size
    total_steps = (len(processed_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)) * num_train_epochs
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    # warmup_steps = 100

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    def train_AsFT(model, train_dataloader, optimizer, lr_scheduler,
          gradient_accumulation_steps, num_train_epochs, project_matrix=None, device="cuda"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                if project_matrix is not None:
                    reg_loss = 0.0
                    lambda_reg = 1
                    idx = 0
                    for i, (name, param) in enumerate(model.named_parameters()):
                        if 'lora_A' in name:
                            lora_a = param
                            lora_b_name = name.replace('lora_A', 'lora_B')
                            lora_b = dict(model.named_parameters())[lora_b_name]
                            delta_w = torch.mm(lora_b, lora_a)
                            c_hat = project_matrix[idx].to(delta_w.device)
                            identity = torch.eye(c_hat.shape[0], device=delta_w.device)
                            regularized_term = (identity - c_hat) @ delta_w
                            reg_loss += lambda_reg * torch.norm(regularized_term, p='fro') ** 2
                            idx += 1
                    loss = loss + reg_loss

                if not loss.isnan():
                    total_loss += loss.detach().float()
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                if step % 10 == 0:
                    print(f"Step {step}, loss={loss.item():.4f}", flush=True)

    train_AsFT(
        model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        project_matrix=project_matrices
    )

    model.save_pretrained(extra_args.output_path)





if __name__ == "__main__":
    train()

