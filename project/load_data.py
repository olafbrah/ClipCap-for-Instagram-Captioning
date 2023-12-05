from datasets import load_dataset


def save_locally(path):
    dataset = load_dataset("kkcosmos/instagram-images-with-captions", split="test[0:100]")
    dataset.save_to_disk(path)
save_locally('small_instagram_data/test')