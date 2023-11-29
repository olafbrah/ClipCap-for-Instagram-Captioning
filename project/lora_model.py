"""
TODO
-Implement LoRA Layers
-Implement LoRA ClipCap Model
"""
import numpy as np
import torch
from torch import nn
import loralib as lora
from clip_model import CaptionModel


def num_train(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 1e6


def num(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params / 1e6


class LoraCaptionModel(nn.Module):
    def __init__(self, rank
                 ):
        super().__init__()
        self.model = CaptionModel(10)
        self.model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location=torch.device('cpu')))
        self.rank = rank

        # add lora to embeddings
        w = self.model.gpt.transformer.wte.weight
        self.model.gpt.transformer.wte = lora.Embedding(50257, 768, r=rank)
        self.model.gpt.transformer.wte.weight = w

        # add lora to each gpt_block <-- conv1d keeps erroring
        # for block in self.model.gpt.transformer.h:
        #     w = block.attn.c_attn.weight
        #     block.attn.c_attn = lora.MergedLinear(768, 2304, fan_in_fan_out=True, enable_lora=[True, False, True], r=rank)
        #     block.attn.c_attn.weight = w

        # add lora to projection map
        map = self.model.clip_project.model
        w = map[0].weight
        map[0] = lora.Linear(map[0].in_features, map[0].out_features, r=rank)
        map[0].weight = w

        w = map[2].weight
        map[2] = lora.Linear(map[2].in_features, map[2].out_features, r=rank)
        map[2].weight = w

    def forward(self, tokens, prefix, mask):
        return self.model(tokens, prefix, mask)
