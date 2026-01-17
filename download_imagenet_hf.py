from datasets import load_dataset
from tqdm.auto import tqdm
import os

root = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf"
splits = ["train", "validation", "test"]

hf_cache = "/scratch/gilbreth/abelde/hf_cache"
dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=hf_cache)

for split in splits:
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)

    ds_split = dataset[split]
    n = len(ds_split)

    print(f"Processing split: {split} ({n} images)")
    for i, sample in tqdm(enumerate(ds_split), total=n, desc=f"{split}", unit="img"):
        img = sample["image"]
        label = sample["label"]

        label_dir = os.path.join(out_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        out_path = os.path.join(label_dir, f"{i}.JPEG")

        # If we already saved this image in a previous run, skip it
        if os.path.exists(out_path):
            continue

        # Some images are RGBA, P, etc. Convert everything to RGB for JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(out_path, format="JPEG")

print("Done saving all splits.")
