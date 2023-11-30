import clip
from peft import LoraConfig
from lora_model import LoraCaptionModel, num, num_train
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer, device=device)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test", device=device)
# train_data = Subset(train_data, indices=range(320))
validation_data = Subset(validation_data, indices=range(1000))
num_data_pts = 8
lora_ranks = [4, 16, 64, 256]
for rank in lora_ranks:
    config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["wte", "wpe", "c_attn", "c_proj", "c_fc"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = LoraCaptionModel(config)
    print(f"Training Rank {rank} LoRA | Total Parameters: {num(model)} | Trainable Parameters: {num_train(model)}")
    train_loss, val_loss = fine_tune(model, train_data, state_prefix=f"rank_{rank}", validation=validation_data, epochs=1, batch_size=40, device=device, num_data_pts=num_data_pts, checkpoint_path="checkpoints/LoRA", chart_title=f"LoRA Rank {rank}")
    x_axis_val = np.linspace(1/num_data_pts, 1/num_data_pts * len(val_loss), num=len(val_loss), endpoint=False)
    plt.plot(x_axis_val, val_loss, label='Validation Loss')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    x_axis_train = np.linspace(1/num_data_pts, 1/num_data_pts * len(train_loss), num=len(train_loss), endpoint=False)
    plt.plot(x_axis_train, train_loss, label='Training Loss')
    plt.title(f"LoRA Rank {rank} Training Curve")
    plt.legend(loc='best')
    plt.savefig(f"checkpoints/LoRA/curves/rank_{rank}_loss_history.png")
    plt.show()


