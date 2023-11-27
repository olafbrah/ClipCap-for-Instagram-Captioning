import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import requests
from io import BytesIO
import os

"""
python parse_hugging.py --clip_model_type RN50x4 --json_path '/Users/albertguo/182proj/data/huggingface/train/data.json'

"""
        
def main(clip_model_type: str, json_path: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/huggingface/embeddings_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/huggingface/train/data.json', 'r') as f:
        data = json.load(f)["rows"]
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]["row"]
        img_url = d["image"]["src"]
        caption = d["caption"]

        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content))
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        all_embeddings.append(prefix)
        d["clip_embedding"] = i
        all_captions.append({"image_id": d["item_id"], "caption": caption, "clip_embedding": i})

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print(f"{len(all_embeddings)} embeddings saved")
    
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--json_path', required=True, help='Path to the Instagram dataset JSON file')
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.json_path))