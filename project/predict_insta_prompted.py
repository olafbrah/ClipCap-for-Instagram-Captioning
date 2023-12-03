import clip
import torch
from prompted_clip_model import PromptedCaptionModel, MLP
from clip_model import generate
from dataset import InstagramDataset
from base_finetune import fine_tune
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load specific model weights
weights_path = "checkpoints/prompted/model_2_24815.pt"  # Replace with your weights file path
prefix_length = 10

model = PromptedCaptionModel(prefix_length)
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model = model.eval().to(device)

test_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test", device="cpu")
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

def predict_captions(model, test_loader, use_beam_search=False):
    for tokens, prefix, mask in test_loader:
        tokens, prefix, mask = tokens.to(device), prefix.to(device), mask.to(device)  # Move data to the same device as the model

        with torch.no_grad():
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            output = generate(model, tokenizer, embed=prefix_embed)
            print(output)
# Example usage
predict_captions(model, test_loader, use_beam_search=False)
