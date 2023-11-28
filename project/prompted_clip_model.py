#%%
import torch
import clip
import numpy as np
from torch import nn
from typing import Tuple, List, Union, Optional
from clip_model import CaptionModel
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from enum import Enum


from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)

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

class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class PromptedCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        device = tokens.device
        if self.training:
            embedding_text = self.gpt.transformer.wte(tokens)
        else:
            original_text = self.original_model(tokens, prefix, mask, labels)
            original_text = r"Describe {orignial_text} like an instagram caption"
            prepend_tokens = self.tokenizer.encode(original_text, return_tensors="pt").to(device)
            embedding_text = self.gpt.transformer.wte(prepend_tokens)

        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        if self.prompt_mode == PromptType.OriginalWithWords:
            print("DEVICE ", prefix_projections.device, self.prepend_embedding.device, embedding_text.device)
            copy = self.prepend_embedding.expand(prefix_projections.shape[0], -1, -1).to(device)
            embedding_cat = torch.cat((
                prefix_projections, 
                copy, 
                embedding_text
            ), dim=1)
        else:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, 
                 prompt_mode: PromptType = PromptType.OriginalWithWords, weights_path: str = "state_dicts/coco_weights.pt" ):
        super(PromptedCaptionModel, self).__init__()
        self.max_seq_len = 77
        self.prompt_mode = prompt_mode
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )

        if self.prompt_mode == PromptType.OriginalWithWords:
            self.original_model = CaptionModel(prefix_length)
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.original_model.load_state_dict(state_dict, strict=False)
            # Freeze the model parameters
            for param in self.original_model.parameters():
                param.requires_grad = False
            
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            device = self.gpt.transformer.wte.weight.device  # Get the device from the model

            prepend_phrase = "A instagram caption would describe this as "
            prepend_tokens = torch.tensor(self.tokenizer.encode(prepend_phrase)).to(device)
            prepend_tokens, mask = self.pad_tokens(prepend_tokens)
            self.prepend_embedding = self.gpt.transformer.wte(prepend_tokens).detach()


    



def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]
