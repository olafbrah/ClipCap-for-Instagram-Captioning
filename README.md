# ClipCap For Instagram Captioning.


Inference Notebook: <a href="https://colab.research.google.com/drive/1o6g8udv-w7cRBlfQVy6vZ6BgVGlzgoTt?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  

## Abstract  
Image captioning is a fundamental task in vision-language understanding, where a model predicts a descriptive caption to a given input image. In this project, we fine-tune the ClipCap model introduced in Mokady et al. to caption images in the style of Instagram captions. The ClipCap model uses the CLIP image/text embedding model from Radford et al. and GPT-2 combined with an MLP mapping model to caption images. The generated CLIP embedding is transformed by the mapping model into a prefix in GPT-2 embeddings, where the language model then generates a caption using this context. Unlike traditional image captions, the social media captions found on Instagram are not entirely descriptive of the image itself. Our goal is to fine-tune the mapping network and GPT-2 interpret the CLIP prefix in a way that captures this difference. To achieve this, we explore various methods to improve training including parameter efficient fine-tuning with LoRA, additional prompt engineering along with the prefix to direct the language model, and experiments with freezing different parts of the model during training. By exploring these different methods, we improve our resulting captions and reduce training time over just fine-tuning the entire model.



## Inference Notebooks
To help visualize the results we provide a Colab notebook found in `notebooks/clip_prefix_captioning_inference.ipynb`.   
The notebook will download the pretrained models and run inference on a sample images or 
on images of your choosing. It is recommended to run this in [Google Colab](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing).
Inference notebook for the **transformer mapping network (without fine-tune GPT-2)** can be found [here](https://colab.research.google.com/drive/180L3rMFmGujudwO1EJNF-lHIpAsAZ5xq?usp=sharing) for the COCO model (also in `notebooks/transformer_inference.ipynb`).

Nearly all of our work is done in the projects directory. Here is how we trained the models.
1. Run the following commands to set up a Conda environment (you may need to do lots of pip installs and this was basically impossible on collab (spent 8+ hrs))
```
git clone https://github.com/olafbrah/cs182proj
conda env create -f environment.yml
conda activate clip_prefix_caption
```
2. Cd into the project directory and get the dataset from hugging face using load_data.py (it's a lot of data)
3. Download pre-trained weights ( https://drive.google.com/drive/folders/1z68jSlSbBZ6mHuqmcpcO3-aEKLUKb72X?usp=sharing )
    Coco_weights.pt (necessary for training): contains the pre-trained weights from the paper
    Base_weights.pt: contains weights after fully fine-tuning the dataset starting from the weights set by Coco_weights.pt
    Prompted_weights: contains weights after training with prompting
    Lora_x_weights.pt: contains weights for a Lora model of rank x
    Frozen_weights.py: weights for a model trained with only head unfrozen
4. Run Corresponding training script
   basic_finetuning_demo.py to fully finetune (produces base_weights.pt)
   Prompt_finetuning.py to use prompts (produces prompted_weights.pt)
   Lora_finetuning.py to do LoRa training (produces LoRA_x_weights.pt)
   Frozen_basic_finetuning.py to do head only training (produces frozen_model_weights.pt)
5. You can see the captions of our models specific images here: https://colab.research.google.com/drive/1o6g8udv-w7cRBlfQVy6vZ6BgVGlzgoTt?usp=sharing (make sure to add the weights from (3) to you “MyDrive”)



Both [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for mlp mapping network. For the transformer (without fine-tuning GPT-2) we provide [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model.


## Acknowledgments
Based on repo [Clip Cap](https://github.com/rmokady/CLIP_prefix_caption) 
For training we used the data of [HuggingFace](https://huggingface.co/datasets/kkcosmos/instagram-images-with-captions) 



