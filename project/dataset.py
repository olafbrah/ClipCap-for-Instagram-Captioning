import clip
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class InstagramDataset(Dataset):
    """
    Dataset that returns the following
    -tokens : tokenized caption via gpt tokenizer
    -prefix : clip image prefix
    -mask : token attention mask for gpt

    --> Possible will error with device stuff, might have to pass in device
    """
    def __init__(self, clip, preprocessor, tokenizer, path="instagram_data", split="train", device="cuda"):
        self.clip_model = clip
        self.preprocess = preprocessor
        self.tokenizer = tokenizer
        assert split in ["train", "test"], "Invalid Split Name! Expected one of 'train' or 'test'"
        self.data_dict = load_from_disk(path)[split]
        self.max_seq_len = 77 # clip max sequence length
        self.prefix_len = 10
        self.device = device

    def __len__(self):
        return self.data_dict.num_rows

    def __getitem__(self, idx):
        entry = self.data_dict[idx]
        pil_image = entry["image"]
        image = self.preprocess(pil_image).unsqueeze(0)
        image = image.to(self.device)
        prefix = self.clip_model.encode_image(image)

        caption = entry["caption"]
        caption = caption
        tokens = torch.tensor(self.tokenizer.encode(caption))
        tokens, mask = self.pad_tokens(tokens)

        return tokens, prefix, mask
    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_len), mask), dim=0)  # adding prefix mask
        return tokens, mask