import clip
import torch
import loralib as lora
from lora_model import LoraCaptionModel, num, num_train
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")

lora_ranks = [4, 16, 64, 256]
for rank in lora_ranks:
    model = LoraCaptionModel(rank)
    lora.mark_only_lora_as_trainable(model)
    train_loss, val_loss = fine_tune(model, train_data, validation = validation_data, epochs=1, batch_size=40, device=device)
    torch.save(model.state_dict(), f"lora_{rank}.pt")
    plt.plot(range(len(train_loss)), train_loss)
    plt.plot(range(len(train_loss)), val_loss)
    plt.savefig(f"lora_{rank}.png")


