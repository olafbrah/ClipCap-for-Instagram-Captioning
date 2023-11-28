import clip
import torch
from clip_model import CaptionModel
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer

device = "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")

model = CaptionModel(10)
model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location="cpu"))
model = model.eval()
model = model.to(device)

train_loss, _ = fine_tune(model, train_data, epochs=3, batch_size=16, device=device)
batches = range(len(train_loss))

import matplotlib.pyplot as plt
plt.plot(batches, train_loss)
plt.savefig("loss_history.png")
