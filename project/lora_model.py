"""
TODO
-Implement LoRA Layers
-Implement LoRA ClipCap Model
"""
import numpy as np
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageClassification

from clip_model import CaptionModel

def num_train(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 1e6


def num(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params / 1e6


class LoraCaptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CaptionModel(10)
        self.model.load_state_dict(self.prep_dict())
        get_peft_model(self.model, config)


    def forward(self, tokens, prefix, mask):
        return self.model(tokens, prefix, mask)

    def prep_dict(self):
        coco_dict = torch.load("state_dicts/coco_weights.pt", map_location="cpu")

        for key in coco_dict.copy().keys():
            if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
                del coco_dict[key]
        return coco_dict
