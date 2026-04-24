import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()


from datasets import load_dataset, load_from_disk
dataset = load_from_disk("../data/alpaca")
output_json = f'../data/alpaca.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    item["input"] = data["instruction"]
    item["output"] = data["output"]
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)



