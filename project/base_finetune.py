"""
TODO
Do basic finetuning of the model on the instagram dataset for desired epochs
--> pass in model so can call on the models we change up
--> should return various statistics (loss history)
--> will have option to do validation loss check
--> also specifiy intervals for data gathering
--> model checkpoints as well to get progress over training <--

Intended use is for this to be imported in other training scripts that iterate over
variables (i.e. LoRA rank, loss fn, prompt stuff) and record results for the writeup
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def fine_tune(model, train, validation=None, epochs=5, batch_size=10, device="cpu", num_data_pts=8, state_prefix="model"):

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    train_loss = []
    if validation:
        validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)
    else:
        validation_loader = None
    validation_loss = []

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_steps, num_training_steps=num_steps)

    for epoch in range(epochs):
        print(f"Training Epoch {epoch + 1}\n>>>")
        loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, num_data_pts=num_data_pts, val_loader=validation_loader, device=device, state_prefix=state_prefix)
        if loss[1]:
            train_loss.extend(loss[0])
            validation_loss.extend(loss[1])
        else:
            train_loss.extend(loss[0])
    return train_loss, validation_loss

def train_epoch(model, loader, optimizer, scheduler, epoch, num_data_pts=8, val_loader=None, device="cpu", state_prefix=None):
    interval = len(loader) // num_data_pts
    train_loss_history, avg_train_loss = [], 0
    val_loss_history = []

    for batch, (tokens, prefix, mask) in tqdm(enumerate(loader), unit="batch", total=len(loader)):

        model.zero_grad()
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        outputs = model(tokens, prefix, mask)


        logits = outputs.logits[:, 10 - 1: -1]  # 10 is prefix_length
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        avg_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print(f"Batch {batch +1}/{len(loader)}")

        if batch % interval == 0:
            train_loss_history.append(avg_train_loss/interval)
            avg_train_loss = 0
            torch.save(model.state_dict(), f"checkpoints/{state_prefix}_{epoch}_{batch}.pt")
            if val_loader and batch > 0:
                model.eval()
                validation_loss = 0
                for tokens, prefix, mask in val_loader:
                    tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, 10 - 1: -1]
                    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                    validation_loss += loss.item()
                val_loss_history.append(validation_loss/len(val_loader))
                model.train()
    return train_loss_history, val_loss_history
