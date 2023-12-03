import torch
import clip
import numpy as np
from torch import nn
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from clip_model import CaptionModel, generate
from lora_model import LoraCaptionModel
from prompted_clip_model import PromptedCaptionModel

from enum import Enum

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class PromptType(Enum):
    Empty = "empty"
    Orginal = 'original'
    OriginalWithWords = 'originalplus'
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]
D = torch.device
CPU = torch.device("cpu")

device = "cuda"
prefix_length = 10
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

coco_model = CaptionModel(prefix_length)
coco_model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location=CPU), strict=False)
coco_model = coco_model.eval()
coco_model = coco_model.to(device)

full_finetuned_model = CaptionModel(prefix_length)
full_finetuned_model.load_state_dict(torch.load("checkpoints/base/complete_base_weights.pt", map_location=CPU), strict=False)
full_finetuned_model = full_finetuned_model.eval()
full_finetuned_model = full_finetuned_model.to(device)

full_finetuned_model_1epoch = CaptionModel(prefix_length)
full_finetuned_model_1epoch.load_state_dict(torch.load("checkpoints/base/model_0_440.pt", map_location=CPU), strict=False)
full_finetuned_model_1epoch = full_finetuned_model_1epoch.eval()
full_finetuned_model_1epoch = full_finetuned_model_1epoch.to(device)


prompted_model = PromptedCaptionModel(prefix_length)
prompted_model.load_state_dict(torch.load("checkpoints/prompted/model_3_3545.pt", map_location=CPU), strict=False)
prompted_model = prompted_model.eval()
prompted_model = prompted_model.to(device)

from PIL import Image
pil_image = Image.open("person.jpg")
image = preprocess(pil_image).unsqueeze(0).to(device)
prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)

prefix_embedding = coco_model.clip_project(prefix).reshape(1, prefix_length, -1)
coco_text = generate(coco_model, tokenizer, embed=prefix_embedding)

prefix_embedding = prompted_model.clip_project(prefix).reshape(1, prefix_length, -1)
prompted_text = generate(prompted_model, tokenizer, embed=prefix_embedding)

prefix_embedding = full_finetuned_model.clip_project(prefix).reshape(1, prefix_length, -1)
full_finetuned_text = generate(full_finetuned_model, tokenizer, embed=prefix_embedding)

prefix_embedding = full_finetuned_model_1epoch.clip_project(prefix).reshape(1, prefix_length, -1)
full_finetuned_text_1epoch = generate(full_finetuned_model_1epoch, tokenizer, embed=prefix_embedding)


print("COCO: ", coco_text)
print("PROMPTED: ", prompted_text)
print("BASE: ", full_finetuned_text)
print("BASE w/ 1: ", full_finetuned_text_1epoch)