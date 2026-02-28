# Using .h5 Files with SEG-RDM

## Overview

The dataset loader supports loading embeddings from `.h5` (HDF5) files in two ways:

1. **Main embeddings** - Region/SAM embeddings can be in H5 format (via `h5_path`)
2. **IJEPA cache** - IJEPA embeddings can be in H5 format (via `ijepa_h5_path`)

You can mix and match: NPZ for main embeddings + H5 for IJEPA, or vice versa.

## Benefits of .h5 Format

✅ **Single file access** - All embeddings in one file  
✅ **Faster I/O** - Reduced filesystem overhead  
✅ **Space efficient** - Better compression  
✅ **Random access** - No need to load entire file  

## Common Use Cases

### Case 1: NPZ (region) + H5 (IJEPA) ← **YOUR SETUP**

```yaml
data:
  params:
    # Main embeddings from NPZ
    mask_npz_dir: "/path/to/region_emb_extract-a100-0.65dedup"
    
    # IJEPA cache from H5
    ijepa_h5_path: "/path/to/ijepa_embeddings.h5"
    ijepa_h5_key_format: "{class_id}/{name}"
    
    # Leave these null
    h5_path: null
    ijepa_cache_dir: null
```

### Case 2: H5 (main) + Runtime IJEPA

```yaml
data:
  params:
    # Main embeddings from H5
    h5_path: "/path/to/sam_embeddings.h5"
    h5_key_format: "{class_id}/{name}"
    
    # IJEPA extracted at runtime
    ijepa_h5_path: null
    ijepa_cache_dir: null
    
    # Fallback NPZ directory
    mask_npz_dir: "/path/to/fallback/npz"
```

### Case 3: Both from H5

```yaml
data:
  params:
    # Main embeddings from H5
    h5_path: "/path/to/sam_embeddings.h5"
    h5_key_format: "{class_id}/{name}"
    
    # IJEPA cache from different H5
    ijepa_h5_path: "/path/to/ijepa_embeddings.h5"
    ijepa_h5_key_format: "{class_id}/{name}"
    
    # Fallback
    mask_npz_dir: "/path/to/fallback/npz"
```

---

## Quick Start

### 1. Install h5py

```bash
pip install h5py>=3.0
# or
pip install -r requirements_min.txt
```

### 2. Test Your Setup

For NPZ (region) + H5 (IJEPA) setup:

```bash
cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM
python test_h5_loading.py
```

This will:
- Inspect the h5 file structure
- Test loading region embeddings from NPZ
- Test loading IJEPA embeddings from H5
- Verify batch loading works correctly

### 3. Train Your Model

Use the pre-configured YAML for your setup:

```bash
python train_unified_seg_rdm.py \
  --config rdm/configs/unified_seg_rdm_region_ijepa.yaml \
  --output_dir rdm/output_region_ijepa
```

## H5 File Structure

### Expected Structure

Your `.h5` file should have one of these structures:

**Option 1: Hierarchical (Recommended)**
```
ijepa_embeddings.h5
├── 0/              # class_id
│   ├── n01440764_10026  # image name (no extension)
│   │   ├── emb            # [N, dim] embeddings
│   │   ├── scores         # [N] confidence scores (optional)
│   │   └── emb_image_mean # [dim] mean embedding (optional)
│   ├── n01440764_10027
│   └── ...
├── 1/
│   ├── n01443537_10007
│   └── ...
```

**Option 2: Flat Structure**
```
ijepa_embeddings.h5
├── n01440764_10026  # image name
│   ├── emb          # [N, dim] embeddings
│   └── scores       # [N] confidence scores (optional)
├── n01440764_10027
└── ...
```

**Option 3: Direct Datasets**
```
ijepa_embeddings.h5
├── 0/n01440764_10026  # [N, dim] - embedding directly
├── 0/n01440764_10027
└── ...
```

### Key Format Patterns

Configure `h5_key_format` based on your structure:

| Structure | h5_key_format |
|-----------|---------------|
| `{class_id}/{name}` | `"{class_id}/{name}"` (default) |
| `{name}` only | `"{name}"` |
| `class_{class_id}/{name}` | `"class_{class_id}/{name}"` |
| `{class_id}/img_{name}` | `"{class_id}/img_{name}"` |

## Dataset Parameters

```python
SegmentationMaskDataset(
    image_dir="path/to/images",
    mask_npz_dir="path/to/npz",  # Fallback if h5 fails
    h5_path="path/to/file.h5",    # NEW: Path to h5 file
    h5_key_format="{class_id}/{name}",  # NEW: Key pattern
    max_segments=250,
    image_size=256,
    normalize=True,
    emb_source="ijepa",  # or "sam", "dinov2", etc.
)
```

## Troubleshooting

### Issue: "KeyError: Could not find embeddings"

**Solution 1:** Check your h5 structure:
```python
import h5py
with h5py.File('ijepa_embeddings.h5', 'r') as f:
    print(list(f.keys())[:10])  # Print first 10 keys
    first_key = list(f.keys())[0]
    print(f"Structure: {first_key}")
```

**Solution 2:** Adjust `h5_key_format` in your config to match the actual structure.

### Issue: "Missing 'emb' key"

Your h5 entries might use different key names. The loader automatically tries:
- `emb`
- `embeddings`
- `embedding`

If your key is different, modify the `_load_from_h5` method in [seg_dataset.py](rdm/data/seg_dataset.py#L340).

### Issue: Slow loading

If loading is slow:
1. Make sure h5 file is on fast storage (not network drive)
2. Reduce `num_workers` in DataLoader config
3. Use h5py's chunking for better cache performance

## Performance Tips

1. **Preload for small datasets**: If your dataset fits in RAM, consider loading all h5 data at init
2. **Use SSD storage**: Place .h5 file on SSD for faster random access
3. **Optimal chunking**: When creating h5 files, use appropriate chunk sizes (typically 100-1000 items)
4. **Close files properly**: The loader automatically closes h5 files on cleanup

## Example: Converting NPZ to H5

If you want to consolidate existing .npz files:

```python
import h5py
import numpy as np
import glob

# Create h5 file
with h5py.File('embeddings.h5', 'w') as h5f:
    # For each class directory
    for class_dir in sorted(glob.glob('sam_cache_unified/*/masks_npz')):
        class_id = class_dir.split('/')[-2]
        
        # For each npz file
        for npz_path in glob.glob(f'{class_dir}/*.npz'):
            name = npz_path.split('/')[-1].replace('.npz', '')
            
            # Load npz
            data = np.load(npz_path, allow_pickle=True)
            
            # Create group in h5
            grp = h5f.create_group(f'{class_id}/{name}')
            
            # Store data
            if 'emb' in data:
                grp.create_dataset('emb', data=data['emb'], compression='gzip')
            if 'scores' in data:
                grp.create_dataset('scores', data=data['scores'])
            if 'emb_image_mean' in data:
                grp.create_dataset('emb_image_mean', data=data['emb_image_mean'])
```

## Next Steps

1. Run `test_h5_loading.py` to verify your setup
2. Start training with `unified_seg_rdm_h5.yaml`
3. Monitor the first few batches to ensure proper loading

For questions or issues, check the main [README](README_UNIFIED_SEG_RDM.md).
