from datasets import load_dataset


def save_locally(path):
    dataset = load_dataset("kkcosmos/instagram-images-with-captions")
    dataset.save_to_disk(path)
