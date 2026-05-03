import os
import json
import argparse
import sys
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import concurrent.futures
sys.path.append("..") 
from my_lora_layer import MyLinear


parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")
parser.add_argument("--task_path", default= "alpaca")
parser.add_argument("--use_snn", action="store_true", help="whether use snn in training")
parser.add_argument("--time_step", type=int, default=3, help="time_step in snn")
parser.add_argument("--batch_size", default=128, type=int)

args = parser.parse_args()
print(args)


if os.path.exists(args.output_path):
    print(f"Output file {args.output_path} exists. We will overwrite it.")
output_folder = os.path.dirname(args.output_path)
if output_folder: 
    os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset, load_from_disk
dataset = load_from_disk(args.task_path)
input_data_lst = []
for data in dataset["test"]:
    item = {}
    item["instruction"] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    item["input"] = data["instruction"]
    item["answer"] = data["output"]
    input_data_lst.append(item)

tokenizer = AutoTokenizer.from_pretrained(args.model_folder)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print("Loading Model...")
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
    if args.lora_folder != "":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
        )

model.eval()


output_lst = []
total_count = len(input_data_lst)
with torch.no_grad():
    for i in tqdm(range(0, total_count, args.batch_size), desc="Batch Generating Responses"):
        batch_instructions = [item["instruction"] + item["input"] for item in input_data_lst[i: i + args.batch_size]]

        
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
            max_new_tokens=256,
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




import requests


API_KEY = ""  
API_BASE = ""   
MODEL_NAME = ""   
MAX_WORKERS = 20   


def get_single_score(item):

    instruction = item.get("input", "")
    ref_ans = item.get("answer", "")
    model_out = item.get("output", "")

    prompt = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant.\n\n"
        "Evaluation criteria: helpfulness, relevance, accuracy, depth, and clarity.\n\n"
        "Question:\n"
        f"{instruction}\n\n"
        "Assistant's Answer:\n"
        f"{model_out}\n\n"
        "Return ONLY a single integer between 1 and 10.\n"
        "Do not provide any explanation or extra text."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    try:

        resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        
        content = resp.json()["choices"][0]["message"]["content"].strip()
        

        match = re.search(r"\b(10|[1-9])\b", content)
        if match:
            score = int(match.group(1))
            valid = True
        else:
            score = -1  
            valid = False
        

        return {
            "score": score,
            "valid": valid,
            "raw": content
        }
        
    except Exception as e:

        print(f"请求失败: {e}") 
    

original_file = f'{args.output_path}.json'
with open(original_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

total = len(eval_data)


scores = []
valid_scores = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    future_to_item = {executor.submit(get_single_score, item): i for i, item in enumerate(eval_data)}

    for future in tqdm(concurrent.futures.as_completed(future_to_item), total=total, desc="API 评估中"):
        index = future_to_item[future]
        res = future.result()
        

        eval_data[index]["eval_score"] = res["score"]
        eval_data[index]["eval_content"] = res["raw"]
        

        scores.append(res["score"])
        if res["valid"]:
            valid_scores.append(res["score"])



valid_count = len(valid_scores)
avg_score = sum(valid_scores) / valid_count if valid_count > 0 else 0


overall_avg = sum(scores) / len(scores) if scores else 0

statistics = {
    "total_samples": total,
    "valid_samples": valid_count,
    "invalid_samples": total - valid_count,
    "valid_ratio": round(valid_count / total * 100, 2) if total > 0 else 0,
    "average_score_valid_only": round(avg_score, 4),
    "average_score_all": round(overall_avg, 4)
}

final_output = {
    "statistics": statistics,
    "details": eval_data
}


with open(f'{args.output_path}_eval.json', "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)