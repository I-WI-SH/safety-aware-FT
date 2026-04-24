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
import argparse
import copy

import numpy
import torch
from transformers import AutoModelForCausalLM



@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    aligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )


    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=21,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda",
        metadata = {"help": "Devices are used in SafeLoRA. (gpu or cpu)"}

    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")


class SafeLoRA:
    def __init__(self, peft_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        self.model_ori = copy.deepcopy(peft_model)
        project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)
        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, -1, show_info=False)
            thrs = numpy.sort(cos)[:self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.devices,

        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.devices,

        )
        v = []
        proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules) and 'bias' not in a_name:
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                # import pdb; pdb.set_trace()
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []
        for (name, param),(name_ori, param_ori) in zip(self.peft_model.named_parameters(), self.model_ori.named_parameters()):  
            if 'lora' in name:
                if param.shape[0] == self.peft_config.r:
                    B = copy.deepcopy(param_ori)
                if param.shape[0] != self.peft_config.r:
                    P = v[idx].to(param.device, dtype=param.dtype)
                    W = torch.mm(P, param_ori.data)
                    fW = torch.mm(W, B)
                    ori = torch.mm(param_ori, B)
                    W_new = torch.mm(P, param_ori.data)
                    cos = numpy.round(torch.nn.functional.cosine_similarity(fW.reshape(1,-1), ori.reshape(1,-1)).item(),5)
                    cos_total.append(cos)

                    if cos <=  thrs_cos:
                        i+=1
                        param.data =  W_new
                    else:
                        param.data = param_ori
                    dist = 1 / (1+torch.norm(fW.reshape(1,-1)-ori.reshape(1,-1)))

                    dis.append(dist.item())
                    idx += 1
        if show_info:
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
        return self.peft_model, cos_total



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='meta/Llama-3-8b-hf')
    parser.add_argument("--lora_folder", default="")
    parser.add_argument("--output_path", default='./finetuned_models/safelora_finetuned_model')
    parser.add_argument("--base_model_path", default='meta/Llama-3-8b-hf')
    parser.add_argument("--aligned_model_path", default='meta/Llama-3-8b-chat-hf')
    parser.add_argument("--num_proj_layers", type=int, default=16)

    args = parser.parse_args()
    print(args)


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
    )

    if args.lora_folder!="":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
        )


    config = SafeLoRAConfig(
        base_model_path=args.base_model_path,
        aligned_model_path=args.aligned_model_path,
        num_proj_layers=args.num_proj_layers
    )

    safeLoRA = SafeLoRA(
        peft_model=model,
        config=config
    )

    safe_model = safeLoRA.model
    safe_model.save_pretrained(args.output_path)
    print(f"SafeLoRA model is saved to {args.output_path}.")


if __name__ == "__main__":
    main()

