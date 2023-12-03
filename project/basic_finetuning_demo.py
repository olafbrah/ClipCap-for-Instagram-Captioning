import clip
import torch
from clip_model import CaptionModel
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

clip_model=clip_model.to(device)
train_data = InstagramDataset(clip_model, preprocess, tokenizer)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")
# train_data = Subset(train_data, indices=range(600))
# validation_data = Subset(validation_data, indices=range(100))
model = CaptionModel(10)
model.load_state_dict(torch.load("state_dicts/coco_weights.pt", map_location="cpu"), strict=False)
model = model.eval()
model = model.to(device)
num_data_pts = 8
train_loss, val_loss = fine_tune(model, train_data, validation=validation_data, epochs=5, 
                          batch_size=64, device=device, num_data_pts=num_data_pts, chart_title="Base Model", 
                          checkpoint_path="/home/albert/tests/project/checkpoints/base")


x_axis_val = np.linspace(1/num_data_pts, 1/num_data_pts * len(val_loss), num=len(val_loss), endpoint=False)
torch.save(model.state_dict(), f"checkpoints/base/complete_base_weights.pt")

plt.clf() 
plt.plot(x_axis_val, val_loss, label='Validation Loss')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
x_axis_train = np.linspace(1/num_data_pts, 1/num_data_pts * len(train_loss), num=len(train_loss), endpoint=False)
plt.plot(x_axis_train, train_loss, label='Training Loss')
plt.title("Base Model Training Curve")
plt.legend(loc='best')
plt.savefig(f"checkpoints/base/curves/complete_loss_history.png")
plt.show()
