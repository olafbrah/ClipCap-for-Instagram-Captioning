import clip
from peft import LoraConfig
from lora_model import LoraCaptionModel, num, num_train
from dataset import InstagramDataset
from base_finetune import fine_tune
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import torch

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_data = InstagramDataset(clip_model, preprocess, tokenizer, device=device)
validation_data = InstagramDataset(clip_model, preprocess, tokenizer, split="test", device=device)

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
    train_loss, val_loss = fine_tune(model, train_data, validation=validation_data, epochs=1, batch_size=24, device=device, num_data_pts=50)
    torch.save(model.state_dict(), f"lora_{rank}.pt")

    plt.plot(range(len(train_loss)), train_loss)
    plt.plot(range(len(train_loss)), val_loss)
    plt.savefig(f"lora_{rank}.png")
    plt.show()


