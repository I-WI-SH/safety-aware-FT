import os
import json
import argparse
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig, get_peft_model
from datasets import Dataset as Dataset1
import pandas as pd
from torch import nn
import sys
sys.path.append("../../") 
from my_lora_layer import MyLinear



access_token = next(open('../../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--eval_num", default=500, type=int)
parser.add_argument("--max_new_tokens", default=256, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")
parser.add_argument("--beaverTails_dataset_path", default= "PKU-Alignment/BeaverTails")
parser.add_argument("--use_snn",action="store_true",help="whethere use snn in training")
parser.add_argument("--time_step",type=int,default=3,help="time_step in snn")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []


from datasets import load_dataset, load_from_disk
dataset =load_from_disk(args.beaverTails_dataset_path)
index=0
input_data_lst = []
for example in dataset["30k_test"]:
    if  index<args.eval_num and not example["is_safe"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instruction_lst += [example["prompt"]]
        input_data_lst += [instance]
        index+=1



tokenizer = AutoTokenizer.from_pretrained(
    args.model_folder, 
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    args.model_folder, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 

    )



if args.use_snn:
    print("Apply SNN-LoRA Model & Recover LoRA weights..")
    loraconfig = LoraConfig.from_pretrained(args.lora_folder)
    custom_module_mapping = {nn.Linear: MyLinear}
    loraconfig._register_custom_module(custom_module_mapping)
    model = get_peft_model(model, loraconfig)
    model.load_adapter(args.lora_folder, adapter_name="default", config=loraconfig)

    for name, module in model.named_modules():
        if hasattr(module, "time_step"):
            module.time_step['default'] = args.time_step
    
    model.to(model.device) 


else:
    if args.lora_folder!="":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
        )
    
model.eval()


output_lst = []
total_count = len(instruction_lst)
with torch.no_grad():
    for i in tqdm(range(0, total_count, args.batch_size), desc="Batch Generating Responses"):
        batch_instructions = instruction_lst[i: i + args.batch_size]

        
        batch_messages = [[{"role": "user", "content": inst}] for inst in batch_instructions]

        prompts_text = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        model_inputs = tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=1024
        ).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_lengths = model_inputs['input_ids'].shape[1]
        response_ids = generated_ids[:, prompt_lengths:]
        responses_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        for j, response in enumerate(responses_text):
            original_data_index = i + j
            output_data = input_data_lst[original_data_index].copy()
            output_data['output'] = response.strip()
            output_lst.append(output_data)


with open(f'{args.output_path}.json', 'w', encoding="utf-8") as f:
    json.dump(output_lst, f, indent=4, ensure_ascii=False)

