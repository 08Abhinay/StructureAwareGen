#!/usr/bin/env python3
"""
Test script to verify h5 loading functionality.
Tests that the SegmentationMaskDataset can load embeddings from .h5 file.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from rdm.data.seg_dataset import SegmentationMaskDataset, collate_seg_batch
from torch.utils.data import DataLoader
import h5py


def inspect_h5_structure(h5_path: str):
    """Inspect the structure of the h5 file."""
    print(f"\n{'='*80}")
    print(f"Inspecting h5 file: {h5_path}")
    print(f"{'='*80}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())[:10]}...")  # Show first 10 keys
        
        # Get first key to inspect structure
        first_key = list(f.keys())[0]
        print(f"\nInspecting first key: {first_key}")
        
        item = f[first_key]
        if isinstance(item, h5py.Group):
            print(f"  Type: Group")
            print(f"  Sub-keys: {list(item.keys())}")
            for sub_key in item.keys():
                dataset = item[sub_key]
                if isinstance(dataset, h5py.Dataset):
                    print(f"    {sub_key}: shape={dataset.shape}, dtype={dataset.dtype}")
        elif isinstance(item, h5py.Dataset):
            print(f"  Type: Dataset")
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
        
        # Count total entries
        total_entries = len(list(f.keys()))
        print(f"\nTotal entries in h5 file: {total_entries}")
    
    print(f"{'='*80}\n")


def test_dataset_loading(
    image_dir: str,
    h5_path: str,
    mask_npz_dir: str,
    h5_key_format: str = "{class_id}/{name}",
    batch_size: int = 4,
):
    """Test dataset loading from h5 file."""
    print(f"\n{'='*80}")
    print("Testing SegmentationMaskDataset with h5 file")
    print(f"{'='*80}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = SegmentationMaskDataset(
        image_dir=image_dir,
        mask_npz_dir=mask_npz_dir,
        h5_path=h5_path,
        h5_key_format=h5_key_format,
        max_segments=250,
        image_size=256,
        file_ext="*.JPEG",
        normalize=True,
        emb_source="ijepa",  # or "sam" depending on your h5 content
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("WARNING: Dataset is empty! Check your paths.")
        return
    
    # Test single sample
    print("\nTesting single sample loading...")
    try:
        sample = dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  seg_embs shape: {sample['seg_embs'].shape}")
        print(f"  num_segments: {sample['num_segments']}")
        print(f"  filename: {sample['filename']}")
        
        if 'emb_image_mean' in sample:
            print(f"  emb_image_mean shape: {sample['emb_image_mean'].shape}")
    except Exception as e:
        print(f"✗ Failed to load sample 0: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test dataloader
    print(f"\nTesting DataLoader with batch_size={batch_size}...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging
            collate_fn=collate_seg_batch
        )
        
        batch = next(iter(dataloader))
        print(f"✓ Successfully loaded batch")
        print(f"  images: {batch['image'].shape}")
        print(f"  seg_embs: {batch['seg_embs'].shape}")
        print(f"  num_segments: {batch['num_segments'].shape}")
        print(f"  filenames: {batch['filename'][:2]}...")
        
        if 'emb_image_mean' in batch:
            print(f"  emb_image_mean: {batch['emb_image_mean'].shape}")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print(f"{'='*80}\n")


def main():
    # Configuration for YOUR setup:
    # - NPZ files for region embeddings
    # - H5 file for IJEPA embeddings
    
    image_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train"
    mask_npz_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_emb_extract-a100-0.65dedup"
    ijepa_h5_path = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/ijepa_embeddings.h5"
    
    # You may need to adjust this based on your h5 structure
    # Common formats: "{class_id}/{name}", "{name}", "class_{class_id}/{name}"
    ijepa_h5_key_format = "{class_id}/{name}"
    
    print("=" * 80)
    print("TESTING DUAL-SOURCE LOADING")
    print("=" * 80)
    print(f"Region embeddings (NPZ): {mask_npz_dir}")
    print(f"IJEPA embeddings (H5):   {ijepa_h5_path}")
    print("=" * 80)
    
    # First, inspect the h5 file structure
    try:
        inspect_h5_structure(ijepa_h5_path)
    except Exception as e:
        print(f"Error inspecting h5 file: {e}")
        print("Make sure h5py is installed: pip install h5py")
        return
    
    # Test dataset loading with BOTH sources
    print(f"\n{'='*80}")
    print("Testing SegmentationMaskDataset with NPZ (region) + H5 (IJEPA)")
    print(f"{'='*80}")
    
    dataset = SegmentationMaskDataset(
        image_dir=image_dir,
        mask_npz_dir=mask_npz_dir,
        ijepa_h5_path=ijepa_h5_path,
        ijepa_h5_key_format=ijepa_h5_key_format,
        max_segments=250,
        image_size=256,
        file_ext="*.JPEG",
        normalize=True,
        emb_source="region",
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("WARNING: Dataset is empty! Check your paths.")
        return
    
    # Test single sample
    print("\nTesting single sample loading...")
    try:
        sample = dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  seg_embs shape: {sample['seg_embs'].shape}")
        print(f"  num_segments: {sample['num_segments']}")
        print(f"  filename: {sample['filename']}")
        
        if 'emb_image_mean' in sample:
            print(f"  emb_image_mean shape: {sample['emb_image_mean'].shape}")
        
        if 'ijepa_emb' in sample:
            print(f"  ✓ IJEPA embedding loaded from H5: {sample['ijepa_emb'].shape}")
        else:
            print(f"  ⚠ No IJEPA embedding (will use runtime extraction)")
        
    except Exception as e:
        print(f"✗ Failed to load sample 0: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test dataloader
    print(f"\nTesting DataLoader with batch_size=4...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging
            collate_fn=collate_seg_batch
        )
        
        batch = next(iter(dataloader))
        print(f"✓ Successfully loaded batch")
        print(f"  images: {batch['image'].shape}")
        print(f"  seg_embs: {batch['seg_embs'].shape}")
        print(f"  num_segments: {batch['num_segments'].shape}")
        print(f"  filenames: {batch['filename'][:2]}...")
        
        if 'emb_image_mean' in batch:
            print(f"  emb_image_mean: {batch['emb_image_mean'].shape}")
        
        if 'ijepa_emb' in batch:
            print(f"  ✓ IJEPA embeddings in batch: {batch['ijepa_emb'].shape}")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print("✓ All tests passed! Your setup is working correctly:")
    print("  - Region embeddings loaded from NPZ")
    print("  - IJEPA embeddings loaded from H5")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
