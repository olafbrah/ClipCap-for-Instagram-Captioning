import clip
import torch
from clip_model import CaptionModel
from prompted_clip_model import PromptedCaptionModel
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer, path="instagram_data")
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")

# model = CaptionModel(10)
model = PromptedCaptionModel(10)
model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location="cpu"), strict=False)
model = model.eval()
model = model.to(device)
num_data_pts = 8
epochs = 5
train_loss, _ = fine_tune(model, train_data, epochs=epochs, batch_size=40, device=device, num_data_pts=num_data_pts)
batches = range(len(train_loss))
import matplotlib.pyplot as plt

plt.plot(batches, train_loss)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.xticks(np.arange(0, max(batches) + 1, 1.0))
# Optionally, you can add a title
plt.title("Training Loss Over Epochs")
# Saving the plot
plt.savefig("loss_history.png")