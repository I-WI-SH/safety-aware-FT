import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()

from datasets import load_dataset, load_from_disk
dataset = load_from_disk("../data/sst5")
output_json = f'../data/sst5.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = "Analyze the sentiment of the input, and respond only with one of the following labels: \n\nvery negative\nnegative\nneutral\npositive\nvery positive\n"
    item["input"] = data["text"]
    item["output"] = data["label_text"]
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
