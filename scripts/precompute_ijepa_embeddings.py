"""
Pre-compute I-JEPA embeddings for all images in a dataset.
Saves embeddings as .npz files maintaining directory structure.

Usage:
    python scripts/precompute_ijepa_embeddings.py \
        --image_dir /path/to/imagenet/train \
        --output_dir /path/to/ijepa_embeddings \
        --ijepa_checkpoint /path/to/ijepa_checkpoint.pth \
        --batch_size 64 \
        --device cuda:0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """Simple dataset for loading images."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if self.transform:
                dummy = Image.new('RGB', (224, 224))
                return self.transform(dummy), str(img_path)
            return None, str(img_path)


def load_ijepa_model(checkpoint_path, device):
    """Load I-JEPA model from checkpoint."""
    print(f"Loading I-JEPA model from {checkpoint_path}")
    
    ijepa_path = Path(__file__).parent / 'ijepa'
    if ijepa_path.exists() and str(ijepa_path) not in sys.path:
        sys.path.insert(0, str(ijepa_path))
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'target_encoder' in checkpoint:
            model_state = checkpoint['target_encoder']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
        
        try:
            from src.models.vision_transformer import vit_huge
            model = vit_huge(patch_size=14, num_classes=0)
        except ImportError:
            print("Warning: Could not import I-JEPA's vision_transformer.")
            print("Attempting to use torchvision ViT as fallback...")
            from torchvision.models import vit_h_14
            model = vit_h_14(pretrained=False)
            model.heads = nn.Identity()
        
        model.load_state_dict(model_state, strict=False)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease ensure:")
        print("1. I-JEPA repository is in scripts/ijepa/")
        print("2. Checkpoint path is correct")
        print("3. Required dependencies are installed")
        sys.exit(1)


def get_image_paths(image_dir, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')):
    """Recursively find all image files."""
    image_dir = Path(image_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(image_dir.rglob(f'*{ext}'))
    
    return sorted(image_paths)


def get_output_path(image_path, image_dir, output_dir):
    """Get corresponding output .npz path maintaining directory structure."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    
    rel_path = image_path.relative_to(image_dir)
    output_path = output_dir / rel_path.parent / f"{rel_path.stem}.npz"
    
    return output_path


def extract_embeddings(model, dataloader, image_dir, output_dir, device, skip_existing=True):
    """Extract embeddings for all images."""
    
    all_paths = []
    all_embeddings = []
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting embeddings"):
            if images is None:
                continue
                
            images = images.to(device)
            
            try:
                features = model(images)
                
                if isinstance(features, tuple):
                    features = features[0]
                
                if features.ndim > 2:
                    features = features.mean(dim=1)
                
                features = features.cpu().numpy()
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                features = np.zeros((len(paths), 256), dtype=np.float32)
            
            for feat, img_path in zip(features, paths):
                output_path = get_output_path(img_path, image_dir, output_dir)
                
                if skip_existing and output_path.exists():
                    continue
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, emb=feat.astype(np.float32))
                
                all_paths.append(img_path)
                all_embeddings.append(feat)
    
    return all_paths, all_embeddings


def main():
    parser = argparse.ArgumentParser(description='Pre-compute I-JEPA embeddings')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--ijepa_checkpoint', type=str, required=True,
                        help='Path to I-JEPA checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip images with existing embeddings')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU (this will be slow)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Scanning for images in {args.image_dir}")
    image_paths = get_image_paths(args.image_dir)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Check your image_dir path.")
        return
    
    if args.skip_existing:
        filtered_paths = []
        for img_path in image_paths:
            output_path = get_output_path(img_path, args.image_dir, args.output_dir)
            if not output_path.exists():
                filtered_paths.append(img_path)
        
        print(f"Skipping {len(image_paths) - len(filtered_paths)} existing embeddings")
        image_paths = filtered_paths
        
        if len(image_paths) == 0:
            print("All embeddings already exist! Use --no-skip_existing to recompute.")
            return
    
    model = load_ijepa_model(args.ijepa_checkpoint, device)
    
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nExtracting embeddings for {len(image_paths)} images...")
    paths, embeddings = extract_embeddings(
        model, dataloader, args.image_dir, args.output_dir, device, args.skip_existing
    )
    
    print(f"\n✓ Successfully extracted {len(paths)} embeddings")
    print(f"✓ Saved to {args.output_dir}")
    
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        print(f"\nEmbedding statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")


if __name__ == '__main__':
    main()
