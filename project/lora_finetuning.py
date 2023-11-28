import clip
import torch
import loralib as lora
from lora_model import LoraCaptionModel, num, num_train
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt

device = "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test")

lora_ranks = [1, 4, 8, 16]
for rank in lora_ranks:
    model = LoraCaptionModel(rank)
    lora.mark_only_lora_as_trainable(model)
    train_loss, _ = fine_tune(model, train_data, epochs=1, batch_size=16, device=device)
    torch.save(model.state_dict(), f"lora_{rank}.pt")
    plt.plot(range(len(train_loss)), train_loss)
    plt.savefig(f"lora_{rank}.png")


