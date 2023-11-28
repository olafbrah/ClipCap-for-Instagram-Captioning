import torch
import clip
from PIL import Image
import pickle
import json
from tqdm import tqdm
import argparse
import requests
from io import BytesIO

"""
python parse_big_hugging.py --clip_model_type ViT-B/32 --json_path '/home/albert/tests/data/huggingface/train/bigdata.json'

"""
def main(clip_model_type: str, json_path: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/huggingface/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    with open(json_path, 'r') as f:
        data = json.load(f)
    print("%d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []

    for entry in tqdm(data):
        d = entry["row"]
        img_url = d["image"]["src"]
        caption = d["caption"]

        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content))
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        all_embeddings.append(prefix)
        d["clip_embedding"] = len(all_embeddings) - 1  # Index of the current embedding
        all_captions.append({"image_id": d["item_id"], "caption": caption, "clip_embedding": d["clip_embedding"]})

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%d embeddings saved " % len(all_embeddings))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--json_path', required=True, help='Path to the Instagram dataset JSON file')
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.json_path))
