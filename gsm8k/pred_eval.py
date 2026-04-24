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

sys.path.append("..") 
from my_lora_layer import MyLinear


parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")
parser.add_argument("--task_path", default= "gsm8k")
parser.add_argument("--use_snn", action="store_true", help="whether use snn in training")
parser.add_argument("--time_step", type=int, default=3, help="time_step in snn")
parser.add_argument("--batch_size", default=128, type=int)

args = parser.parse_args()
print(args)


if os.path.exists(args.output_path):
    print(f"Output file {args.output_path} exists. We will overwrite it.")
output_folder = os.path.dirname(args.output_path)
if output_folder: # 防止当前目录为空字符串导致报错
    os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset, load_from_disk
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"
dataset = load_from_disk(args.task_path)
index = 0
input_data_lst = []
for data in dataset["test"]:
    if index < 500:
        item = {}
        item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
        item["answer"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
        input_data_lst.append(item)
        index += 1

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
        batch_instructions = [item["instruction"] for item in input_data_lst[i: i + args.batch_size]]

        
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

        # 解码部分保持不变
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


# -------------------------------------------------------------------------
# 3. 评估部分 (Evaluation)
# -------------------------------------------------------------------------

def extract_answer_number(sentence: str) -> float:
    """
    从文本中提取数字的辅助函数
    """
    sentence = str(sentence).replace(',', '')
    # 查找所有数字
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    
    # 尝试分割 "The final answer is:" (根据你的 ANSWER_PROMPT)
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # 如果没有标准分割符，取最后一个数字
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float('inf')
    return pred_answer

print("Starting Evaluation...")
correct_count = 0
total_eval = 0

# 这里使用 tqdm 显示评估进度条
for item in tqdm(output_lst, desc="Evaluating Accuracy"):
    # 提取标准答案 (item['answer'] 是 Ground Truth)
    answer_gt = extract_answer_number(item.get("answer", ""))
    
    # 提取预测答案 (item['output'] 是模型输出)
    answer_pred = extract_answer_number(item.get("output", ""))
    
    # 判断是否相等
    if answer_gt == answer_pred:
        correct_count += 1
        item["correct"] = "true"
    else:
        item["correct"] = "false"
    
    total_eval += 1

# 计算最终准确率
acc = (correct_count / total_eval * 100) if total_eval > 0 else 0
print(f"Evaluation Finished.")
print(f"Total: {total_eval}, Correct: {correct_count}, Accuracy: {acc:.2f}%")

# 将分数添加到列表最后或单独保存
output_lst.append({"summary": f"score={acc:.2f}"})

# 最终覆盖保存文件
with open(f'{args.output_path}.json', 'w', encoding="utf-8") as f:
    json.dump(output_lst, f, indent=4, ensure_ascii=False) 

print(f"Results saved to {args.output_path}")

