import os
import json
import argparse
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig, get_peft_model
import sys
sys.path.append("..") 
from my_lora_layer import MyLinear

access_token = next(open('../huggingface_token.txt')).strip()

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--output_path", default='../../data/sst2/instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")
parser.add_argument("--task_path", default= "ag_news")
parser.add_argument("--use_snn",action="store_true",help="whethere use snn in training")
parser.add_argument("--time_step",type=int,default=3,help="time_step in snn")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset, load_from_disk
dataset =load_from_disk(args.task_path)
index=0
input_data_lst = []
for example in dataset["test"]:
    if  index<500 :
        instance = {}
        instance["instruction"] = "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech\n"
        instance["input"] = example["text"]
        instance["label"] = example["label"]
        if  instance["label"] == 0: 
            instance["label_text"] = "World"
        elif instance["label"] == 1: 
            instance["label_text"] = "Sports"
        elif instance["label"] == 2: 
            instance["label_text"] = "Business"
        else:
            instance["label_text"] = "Sci/Tech"
        input_data_lst += [instance]
        index+=1

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(
    args.model_folder, 
    )
tokenizer.pad_token = tokenizer.eos_token

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


def query(data):
    instruction = data["instruction"]
    input = data["input"]
    # prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    # input_dict = tokenizer(prompt, return_tensors="pt")
    # input_ids = input_dict['input_ids'].cuda()

    message = [
        # {"role": "system", "content": instruction},
        {"role": "user", "content": instruction + input},
    ]
    input_ids = tokenizer.apply_chat_template(
        message, 
        add_generation_prompt=True, 
        return_tensors="pt"
        ).to(model.device)
    
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response  = generation_output[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)
    return output


pred_lst = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst.append(pred)

output_lst = []
correct = 0
total = 0
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    
    label = input_data["label_text"]
    
    if label.lower() == pred.lower():
        correct += 1
        input_data["correct"] = "true"  
    else:
        input_data["correct"] = "false"

    total += 1
    output_lst.append(input_data)

print("{:.2f}".format(correct/total*100))
output_lst .append("score={:.2f}".format(correct/total*100))
with open(f'{args.output_path}.json', 'w', encoding="utf-8") as f:
    json.dump(output_lst, f, indent=4, ensure_ascii=False)
