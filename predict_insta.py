import clip
import torch
import PIL.Image
import skimage.io as io
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from train import ClipCaptionModel
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests

# Definitions of MLP, ClipCaptionModel, generate_beam, generate2 remain the same
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
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
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


# Load the model and other necessary components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load specific model weights (e.g., 'coco')
weights_path = "/home/albert/tests/hugging_train/coco_prefix-009.pt"  # Replace with your weights file path

prefix_length = 10
clip_length = 10
prefix_dim = 640
num_layers = 8
mapping_type = 'mlp'

model = ClipCaptionModel(prefix_length, clip_length=clip_length, prefix_size=prefix_dim,
                                  num_layers=num_layers, mapping_type=mapping_type)
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("clip_project.transformer")}

model.load_state_dict(filtered_state_dict, strict=False)
model = model.eval().to(device)
# state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# Print the keys
print("Keys in the state dictionary:")
for key in state_dict.keys():
    print(key)

def predict_captions(json_path, use_beam_search=False):
    with open('./data/huggingface/test/data.json', 'r') as f:
        data = json.load(f)["rows"]
    for i in tqdm(range(len(data))):
        d = data[i]["row"]
        img_url = d["image"]["src"]
        caption = d["caption"]
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content))
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

        print(generate2(model, tokenizer, embed=prefix_embed))

# Example usage
test_set_path = "/home/albert/tests/data/huggingface/test/data.json"  # Replace with your image file path
caption = predict_captions(test_set_path, use_beam_search=False)
