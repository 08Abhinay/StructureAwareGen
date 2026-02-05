"""
Dataset loader for Unified Segmentation RDM.
Loads images and pre-computed SAM segmentation embeddings.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, Tuple


class SegmentationMaskDataset(Dataset):
    """
    Dataset for loading images with pre-computed SAM segmentation embeddings.
    
    Expected structure:
        image_dir/
            ├── 000000000139.jpg
            ├── 000000000285.jpg
            └── ...
        mask_npz_dir/
            ├── 000000000139.npz  (contains 'emb', 'scores', 'packed', 'shape')
            ├── 000000000285.npz
            └── ...
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_npz_dir: str,
        max_segments: int = 250,
        image_size: int = 256,
        file_ext: str = "*.jpg",
        normalize: bool = True,
        ijepa_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            image_dir: Directory containing original images
            mask_npz_dir: Directory containing SAM .npz files with embeddings
            max_segments: Maximum number of segments (for padding)
            image_size: Target image size for resizing
            file_ext: File extension pattern for images (e.g., "*.jpg", "*.png")
            normalize: Whether to normalize images to [-1, 1] range (DDPM expects this)
            ijepa_cache_dir: Optional directory with pre-cached IJEPA embeddings (speeds up training)
        """
        self.image_dir = image_dir
        self.mask_npz_dir = mask_npz_dir
        self.max_segments = max_segments
        self.image_size = image_size
        self.normalize = normalize
        self.ijepa_cache_dir = ijepa_cache_dir
        
        # Find all images with corresponding SAM embeddings
        # Scan mask_npz_dir first, then find matching images
        npz_files = sorted(glob.glob(os.path.join(mask_npz_dir, "*.npz")))
        if len(npz_files) == 0:
            npz_files = sorted(glob.glob(os.path.join(mask_npz_dir, "**", "*.npz"), recursive=True))
        print(f"Found {len(npz_files)} SAM .npz files in {mask_npz_dir}")
        
        # Build image paths by matching npz basenames
        self.image_paths = []
        missing_images = []
        for npz_path in npz_files:
            npz_name = self._get_basename(npz_path)
            # Try different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG']:
                img_path = os.path.join(image_dir, f"{npz_name}{ext}")
                if not os.path.exists(img_path):
                    # Try recursive search
                    matches = glob.glob(os.path.join(image_dir, "**", f"{npz_name}{ext}"), recursive=True)
                    if matches:
                        img_path = matches[0]
                
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    found = True
                    break
            
            if not found:
                missing_images.append(npz_name)
        
        if missing_images and len(missing_images) < 10:
            print(f"WARNING: {len(missing_images)} npz files have no matching images: {missing_images[:5]}")
        elif missing_images:
            print(f"WARNING: {len(missing_images)} npz files have no matching images")
        
        print(f"SegmentationMaskDataset: Loaded {len(self.image_paths)} images with SAM embeddings")
        print(f"  (filtered from {len(npz_files)} npz files)")
        print(f"  image_dir: {image_dir}")
        print(f"  mask_npz_dir: {mask_npz_dir}")
        print(f"  max_segments: {max_segments}")
        
        # Verify at least one image has corresponding npz
        if len(self.image_paths) > 0:
            sample_name = self._get_basename(self.image_paths[0])
            sample_npz = os.path.join(mask_npz_dir, f"{sample_name}.npz")
            if not os.path.exists(sample_npz):
                print(f"WARNING: Sample npz not found: {sample_npz}")
                print(f"  Make sure npz files have same basename as images")
        
        # Image transforms
        if normalize:
            # Normalize to [-1, 1] for DDPM
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # Keep in [0, 1] range
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    
    def _get_basename(self, path: str) -> str:
        """Get filename without extension."""
        return os.path.splitext(os.path.basename(path))[0]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load image and corresponding SAM embeddings.
        
        Returns:
            dict with keys:
                - image: [3, H, W] RGB image tensor
                - seg_embs: [max_segments, 256] segmentation embeddings (padded)
                - num_segments: int, actual number of segments before padding
                - scores: [N] confidence scores for each segment
                - seg_masks: [N, H, W] binary masks (optional, for visualization)
        """
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)  # [3, H, W]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image on error
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Load SAM embeddings from npz
        name = self._get_basename(img_path)
        npz_path = os.path.join(self.mask_npz_dir, f"{name}.npz")
        
        if not os.path.exists(npz_path):
            # Handle missing npz gracefully
            print(f"WARNING: npz not found: {npz_path}, using dummy data")
            seg_embs = torch.zeros(self.max_segments, 256)
            num_segments = 0
            scores = torch.zeros(0)
            seg_masks = None
        else:
            try:
                data = np.load(npz_path, allow_pickle=False)
                
                # Get embeddings (already 256-dim from SAM)
                if 'emb' in data and data['emb'] is not None:
                    embs = data['emb']  # [N, 256] float16
                    embs = torch.from_numpy(embs).float()  # Convert to float32
                else:
                    # Fallback if embeddings not in file
                    embs = torch.randn(1, 256)  # Dummy embedding
                
                # Get scores
                if 'scores' in data:
                    scores = torch.from_numpy(data['scores']).float()
                else:
                    scores = torch.ones(len(embs))
                
                # Optionally load binary masks (for visualization/debugging)
                seg_masks = None
                if 'packed' in data and 'shape' in data:
                    packed = data['packed']
                    shape = data['shape']
                    if len(shape) == 3 and packed.size > 0:
                        N, H, W = shape
                        HW = H * W
                        masks_flat = np.unpackbits(packed, axis=1)[:, :HW]
                        seg_masks = torch.from_numpy(masks_flat.reshape(N, H, W)).bool()
                
                N_actual = embs.shape[0]
                
                # Pad or truncate to max_segments
                if N_actual < self.max_segments:
                    # Pad with zeros
                    pad_size = self.max_segments - N_actual
                    seg_embs = torch.cat([
                        embs,
                        torch.zeros(pad_size, 256)
                    ], dim=0)
                    num_segments = N_actual
                else:
                    # Truncate if we have too many
                    seg_embs = embs[:self.max_segments]
                    num_segments = self.max_segments
                    scores = scores[:self.max_segments]
                    if seg_masks is not None:
                        seg_masks = seg_masks[:self.max_segments]
                        
            except Exception as e:
                print(f"Error loading npz {npz_path}: {e}")
                seg_embs = torch.zeros(self.max_segments, 256)
                num_segments = 0
                scores = torch.zeros(0)
                seg_masks = None
        
        # Load pre-cached IJEPA embedding if available
        ijepa_emb = None
        if self.ijepa_cache_dir is not None:
            # Extract class ID from image path (ImageNet structure: class_folder/image.JPEG)
            # Try to find the class folder (parent directory of the image)
            try:
                # Get parent directory name (class ID for ImageNet)
                parent_dir = os.path.basename(os.path.dirname(img_path))
                # Try to find npz file in ijepa_cache_dir/{class_id}/{name}.npz
                ijepa_npz_path = os.path.join(self.ijepa_cache_dir, parent_dir, f"{name}.npz")
                
                if os.path.exists(ijepa_npz_path):
                    ijepa_data = np.load(ijepa_npz_path, allow_pickle=False)
                    if 'emb' in ijepa_data:
                        ijepa_emb = torch.from_numpy(ijepa_data['emb']).float()  # [1280] from ViT-H/14
                    else:
                        print(f"WARNING: 'emb' key not found in {ijepa_npz_path}")
                # If not found, will fall back to runtime extraction in get_input()
            except Exception as e:
                print(f"Error loading IJEPA cache for {name}: {e}")
                # Fall back to runtime extraction
        
        # Build output dict
        output = {
            'image': image,
            'seg_embs': seg_embs,  # [max_segments, 256]
            'num_segments': num_segments,  # actual count
            'scores': scores,  # [N]
            'filename': name,
        }
        
        # Add IJEPA embedding if loaded from cache
        if ijepa_emb is not None:
            output['ijepa_emb'] = ijepa_emb  # [1280] raw ViT-H/14 dimension
        
        if seg_masks is not None:
            output['seg_masks'] = seg_masks
        
        return output


class ConditionalSegmentationDataset(SegmentationMaskDataset):
    """
    Extended dataset that includes class labels for conditional generation.
    
    Expects a labels file mapping image names to class indices:
        labels.txt:
            000000000139 5
            000000000285 12
            ...
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_npz_dir: str,
        labels_file: str,
        num_classes: int,
        max_segments: int = 250,
        image_size: int = 256,
        file_ext: str = "*.jpg",
        normalize: bool = True,
    ):
        """
        Args:
            labels_file: Path to text file with "filename class_id" per line
            num_classes: Total number of classes
        """
        super().__init__(
            image_dir=image_dir,
            mask_npz_dir=mask_npz_dir,
            max_segments=max_segments,
            image_size=image_size,
            file_ext=file_ext,
            normalize=normalize,
        )
        
        self.num_classes = num_classes
        
        # Load labels
        self.labels = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name, label = parts[0], int(parts[1])
                        self.labels[name] = label
            print(f"  Loaded {len(self.labels)} class labels from {labels_file}")
        else:
            print(f"  WARNING: labels file not found: {labels_file}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with class label."""
        output = super().__getitem__(idx)
        
        # Add class label
        name = output['filename']
        if name in self.labels:
            class_label = self.labels[name]
        else:
            class_label = 0  # Default to class 0 if not found
        
        output['class_label'] = torch.tensor(class_label, dtype=torch.long)
        
        return output


# Utility functions
def collate_seg_batch(batch):
    """
    Custom collate function for variable-length segments.
    Handles batches where different samples have different numbers of segments.
    """
    images = torch.stack([item['image'] for item in batch])
    seg_embs = torch.stack([item['seg_embs'] for item in batch])
    num_segments = torch.tensor([item['num_segments'] for item in batch], dtype=torch.long)
    
    # Pad scores to same length
    max_scores = max(len(item['scores']) for item in batch)
    scores = []
    for item in batch:
        s = item['scores']
        if len(s) < max_scores:
            s = torch.cat([s, torch.zeros(max_scores - len(s))])
        scores.append(s)
    scores = torch.stack(scores)
    
    output = {
        'image': images,
        'seg_embs': seg_embs,
        'num_segments': num_segments,
        'scores': scores,
        'filename': [item['filename'] for item in batch],
    }
    
    # Add pre-cached IJEPA embeddings if present
    if 'ijepa_emb' in batch[0]:
        output['ijepa_emb'] = torch.stack([item['ijepa_emb'] for item in batch])
    
    # Add class labels if present
    if 'class_label' in batch[0]:
        output['class_label'] = torch.stack([item['class_label'] for item in batch])
    
    # Add masks if present
    if 'seg_masks' in batch[0] and batch[0]['seg_masks'] is not None:
        # Note: masks may have different sizes, so we keep them as list
        output['seg_masks'] = [item['seg_masks'] for item in batch]
    
    return output


# Test function
def test_dataset():
    """Test the dataset loader."""
    print("Testing SegmentationMaskDataset...")
    
    # These paths should be adjusted to your actual data
    image_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/val2017"
    mask_npz_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/out/masks_npz"
    
    if not os.path.exists(image_dir):
        print(f"WARNING: image_dir not found: {image_dir}")
        print("Skipping test")
        return
    
    dataset = SegmentationMaskDataset(
        image_dir=image_dir,
        mask_npz_dir=mask_npz_dir,
        max_segments=250,
        image_size=256,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  seg_embs shape: {sample['seg_embs'].shape}")
        print(f"  num_segments: {sample['num_segments']}")
        print(f"  scores shape: {sample['scores'].shape}")
        print(f"  filename: {sample['filename']}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_seg_batch
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch:")
        print(f"  images: {batch['image'].shape}")
        print(f"  seg_embs: {batch['seg_embs'].shape}")
        print(f"  num_segments: {batch['num_segments']}")
        print(f"  scores: {batch['scores'].shape}")
        
        print("\n✓ Dataset test passed!")
    else:
        print("WARNING: Dataset is empty")


if __name__ == "__main__":
    test_dataset()
