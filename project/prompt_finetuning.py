import clip
import torch
from clip_model import CaptionModel
from prompted_clip_model import PromptedCaptionModel
from dataset import InstagramDataset, PromptedInstagramDataset
from prompt_finetune import fine_tune
from transformers import GPT2Tokenizer
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = PromptedInstagramDataset(clip_model, preprocess, tokenizer, path="instagram_data", device=device)
validation_data = PromptedInstagramDataset(clip_model, preprocess, tokenizer, split="test")
# train_data = Subset(train_data, indices=range(500))
# validation_data = Subset(validation_data, indices=range(1000))

# model = CaptionModel(10)
model = PromptedCaptionModel(10, device=device)
model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location="cpu"), strict=False)
model = model.eval()
model = model.to(device)
num_data_pts = 8
epochs = 4
train_loss, val_loss = fine_tune(model, train_data, validation=validation_data, epochs=epochs, batch_size=16, device=device, num_data_pts=num_data_pts, 
                          checkpoint_path="checkpoints/prompted", chart_title="Prompted Model")
torch.save(model.state_dict(), f"checkpoints/prompted/complete_prompt_weights.pt")

x_axis_val = np.linspace(1/num_data_pts, 1/num_data_pts * len(val_loss), num=len(val_loss), endpoint=False)
plt.clf() 
plt.plot(x_axis_val, val_loss, label='Validation Loss')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
x_axis_train = np.linspace(1/num_data_pts, 1/num_data_pts * len(train_loss), num=len(train_loss), endpoint=False)
plt.plot(x_axis_train, train_loss, label='Training Loss')
plt.title(f"Prompted Model Training Curve")
plt.legend(loc='best')
plt.savefig(f"checkpoints/prompted/curves/complete_loss_history.png")
plt.show()