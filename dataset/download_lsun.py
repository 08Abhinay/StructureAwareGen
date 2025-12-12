import os
import random
import shutil
import argparse

def split_dataset(input_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    # Gather all image file paths
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                all_files.append((full_path, rel_path))
    print(f"Found {len(all_files)} images in {input_dir}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    # Copy files to train_dir and test_dir
    for subset, files in [('train', train_files), ('test', test_files)]:
        out_dir = train_dir if subset == 'train' else test_dir
        for full_path, rel_path in files:
            dest_path = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(full_path, dest_path)
        print(f"Copied {len(files)} files to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input dataset directory")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Output directory for training split")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Output directory for testing split")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of data to use for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")

    args = parser.parse_args()
    split_dataset(args.input_dir, args.train_dir, args.test_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()
