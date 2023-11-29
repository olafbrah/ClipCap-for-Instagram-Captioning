import clip
import torch
from clip_model import CaptionModel
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

clip_model=clip_model.to(device)
train_data = InstagramDataset(clip_model, preprocess, tokenizer)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")

model = CaptionModel(10)
model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location="cpu"))
model = model.eval()
model = model.to(device)

train_loss, _ = fine_tune(model, train_data, epochs=1, batch_size=64, device=device, num_data_pts=8, checkpoint_path="/home/albert/tests/project/checkpoints/base")
batches = range(len(train_loss))

import matplotlib.pyplot as plt

plt.plot(batches, train_loss)

plt.xlabel("Epochs")
plt.ylabel("Training Loss")

# Optionally, you can add a title
plt.title("Base Training Loss Over Epochs")

# Saving the plot
plt.savefig("loss_history_base.png")