"""
Verify data alignment between images, SAM embeddings, and I-JEPA embeddings.
Checks that every image has corresponding .npz files for both SAM and I-JEPA.

Usage:
    python scripts/verify_data_alignment.py \
        --image_dir /path/to/imagenet/train \
        --sam_npz_dir /path/to/sam_embeddings \
        --ijepa_npz_dir /path/to/ijepa_embeddings
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def get_image_paths(image_dir, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')):
    """Recursively find all image files."""
    image_dir = Path(image_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(image_dir.rglob(f'*{ext}'))
    
    return sorted(image_paths)


def get_corresponding_npz(image_path, image_dir, npz_dir):
    """Get corresponding .npz path for an image."""
    image_dir = Path(image_dir)
    npz_dir = Path(npz_dir)
    image_path = Path(image_path)
    
    rel_path = image_path.relative_to(image_dir)
    npz_path = npz_dir / rel_path.parent / f"{rel_path.stem}.npz"
    
    return npz_path


def check_npz_file(npz_path, expected_key='emb', check_shape=None):
    """Verify .npz file exists and has correct structure."""
    if not npz_path.exists():
        return False, "File not found"
    
    try:
        data = np.load(npz_path)
        
        if expected_key not in data:
            return False, f"Missing key '{expected_key}'"
        
        emb = data[expected_key]
        
        if check_shape is not None:
            if emb.shape != check_shape and len(check_shape) == len(emb.shape):
                if not all(s == -1 or s == es for s, es in zip(check_shape, emb.shape)):
                    return False, f"Shape mismatch: expected {check_shape}, got {emb.shape}"
        
        return True, emb.shape
        
    except Exception as e:
        return False, f"Error loading: {str(e)}"


def verify_alignment(image_dir, sam_npz_dir, ijepa_npz_dir, max_check=None):
    """Verify alignment between images and embeddings."""
    
    print(f"{Colors.BOLD}Scanning for images...{Colors.ENDC}")
    image_paths = get_image_paths(image_dir)
    print(f"Found {len(image_paths)} images\n")
    
    if len(image_paths) == 0:
        print(f"{Colors.RED}No images found in {image_dir}{Colors.ENDC}")
        return
    
    if max_check is not None:
        image_paths = image_paths[:max_check]
        print(f"Checking first {max_check} images only\n")
    
    missing_sam = []
    missing_ijepa = []
    invalid_sam = []
    invalid_ijepa = []
    valid_triplets = []
    
    sam_shapes = defaultdict(int)
    ijepa_shapes = defaultdict(int)
    
    print(f"{Colors.BOLD}Verifying alignment...{Colors.ENDC}")
    for img_path in tqdm(image_paths):
        sam_npz_path = get_corresponding_npz(img_path, image_dir, sam_npz_dir)
        ijepa_npz_path = get_corresponding_npz(img_path, image_dir, ijepa_npz_dir)
        
        sam_valid, sam_info = check_npz_file(sam_npz_path, expected_key='emb', check_shape=(-1, 256))
        ijepa_valid, ijepa_info = check_npz_file(ijepa_npz_path, expected_key='emb', check_shape=(256,))
        
        if not sam_valid:
            if "File not found" in str(sam_info):
                missing_sam.append((img_path, sam_npz_path))
            else:
                invalid_sam.append((img_path, sam_npz_path, sam_info))
        else:
            sam_shapes[sam_info] += 1
        
        if not ijepa_valid:
            if "File not found" in str(ijepa_info):
                missing_ijepa.append((img_path, ijepa_npz_path))
            else:
                invalid_ijepa.append((img_path, ijepa_npz_path, ijepa_info))
        else:
            ijepa_shapes[ijepa_info] += 1
        
        if sam_valid and ijepa_valid:
            valid_triplets.append((img_path, sam_npz_path, ijepa_npz_path))
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}ALIGNMENT VERIFICATION RESULTS{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    total = len(image_paths)
    valid = len(valid_triplets)
    
    if valid == total:
        print(f"{Colors.GREEN}✓ PERFECT ALIGNMENT!{Colors.ENDC}")
        print(f"{Colors.GREEN}  All {total} images have corresponding SAM and I-JEPA embeddings{Colors.ENDC}\n")
    else:
        print(f"{Colors.YELLOW}⚠ ALIGNMENT ISSUES DETECTED{Colors.ENDC}")
        print(f"  Valid triplets: {Colors.GREEN}{valid}/{total}{Colors.ENDC} ({100*valid/total:.1f}%)\n")
    
    if missing_sam:
        print(f"{Colors.RED}Missing SAM embeddings: {len(missing_sam)}{Colors.ENDC}")
        for img_path, sam_path in missing_sam[:5]:
            print(f"  {Colors.RED}✗{Colors.ENDC} {img_path.name}")
            print(f"    Expected: {sam_path}")
        if len(missing_sam) > 5:
            print(f"  ... and {len(missing_sam) - 5} more")
        print()
    
    if missing_ijepa:
        print(f"{Colors.RED}Missing I-JEPA embeddings: {len(missing_ijepa)}{Colors.ENDC}")
        for img_path, ijepa_path in missing_ijepa[:5]:
            print(f"  {Colors.RED}✗{Colors.ENDC} {img_path.name}")
            print(f"    Expected: {ijepa_path}")
        if len(missing_ijepa) > 5:
            print(f"  ... and {len(missing_ijepa) - 5} more")
        print()
    
    if invalid_sam:
        print(f"{Colors.RED}Invalid SAM embeddings: {len(invalid_sam)}{Colors.ENDC}")
        for img_path, sam_path, error in invalid_sam[:3]:
            print(f"  {Colors.RED}✗{Colors.ENDC} {img_path.name}: {error}")
        if len(invalid_sam) > 3:
            print(f"  ... and {len(invalid_sam) - 3} more")
        print()
    
    if invalid_ijepa:
        print(f"{Colors.RED}Invalid I-JEPA embeddings: {len(invalid_ijepa)}{Colors.ENDC}")
        for img_path, ijepa_path, error in invalid_ijepa[:3]:
            print(f"  {Colors.RED}✗{Colors.ENDC} {img_path.name}: {error}")
        if len(invalid_ijepa) > 3:
            print(f"  ... and {len(invalid_ijepa) - 3} more")
        print()
    
    if sam_shapes:
        print(f"{Colors.BOLD}SAM Embedding Statistics:{Colors.ENDC}")
        for shape, count in sorted(sam_shapes.items(), key=lambda x: -x[1])[:10]:
            print(f"  Shape {shape}: {count} images")
        if len(sam_shapes) > 10:
            print(f"  ... and {len(sam_shapes) - 10} more unique shapes")
        
        num_segments = [shape[0] for shape in sam_shapes.keys()]
        print(f"  Segments per image: min={min(num_segments)}, max={max(num_segments)}, "
              f"avg={sum(s*c for s, c in [(sh[0], cnt) for sh, cnt in sam_shapes.items()])/valid:.1f}")
        print()
    
    if ijepa_shapes:
        print(f"{Colors.BOLD}I-JEPA Embedding Statistics:{Colors.ENDC}")
        for shape, count in sorted(ijepa_shapes.items(), key=lambda x: -x[1]):
            print(f"  Shape {shape}: {count} images")
        print()
    
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    
    if valid == total:
        print(f"\n{Colors.GREEN}✓ Ready for training!{Colors.ENDC}")
        return True
    else:
        print(f"\n{Colors.RED}✗ Fix alignment issues before training{Colors.ENDC}")
        if missing_sam:
            print(f"  Run: python scripts/precompute_sam_embeddings.py ...")
        if missing_ijepa:
            print(f"  Run: python scripts/precompute_ijepa_embeddings.py ...")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify data alignment')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--sam_npz_dir', type=str, required=True,
                        help='Directory containing SAM .npz files')
    parser.add_argument('--ijepa_npz_dir', type=str, required=True,
                        help='Directory containing I-JEPA .npz files')
    parser.add_argument('--max_check', type=int, default=None,
                        help='Maximum number of images to check (None = all)')
    
    args = parser.parse_args()
    
    if not Path(args.image_dir).exists():
        print(f"{Colors.RED}Error: Image directory not found: {args.image_dir}{Colors.ENDC}")
        return
    
    if not Path(args.sam_npz_dir).exists():
        print(f"{Colors.RED}Error: SAM directory not found: {args.sam_npz_dir}{Colors.ENDC}")
        return
    
    if not Path(args.ijepa_npz_dir).exists():
        print(f"{Colors.RED}Error: I-JEPA directory not found: {args.ijepa_npz_dir}{Colors.ENDC}")
        return
    
    success = verify_alignment(
        args.image_dir,
        args.sam_npz_dir,
        args.ijepa_npz_dir,
        args.max_check
    )
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
