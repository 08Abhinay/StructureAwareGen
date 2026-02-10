#!/usr/bin/env python3
"""
Scan SAM cache directory for corrupt npz files (object dtype arrays).
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def check_npz_file(npz_path):
    """Check if an npz file has object dtype arrays."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        issues = []
        for key in data.keys():
            arr = data[key]
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.object_:
                    issues.append(f"{key}: dtype={arr.dtype}")
                elif not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
                    issues.append(f"{key}: dtype={arr.dtype} (non-numeric)")
        
        data.close()
        return issues
    except Exception as e:
        return [f"ERROR: {e}"]

def scan_sam_cache(cache_dir, max_files=None):
    """Scan entire SAM cache for corrupt files."""
    cache_path = Path(cache_dir)
    
    # Find all npz files
    npz_files = list(cache_path.rglob("*.npz"))
    print(f"Found {len(npz_files)} npz files to check")
    
    if max_files:
        npz_files = npz_files[:max_files]
        print(f"Checking first {max_files} files only")
    
    corrupt_files = []
    
    for npz_path in tqdm(npz_files, desc="Checking npz files"):
        issues = check_npz_file(npz_path)
        if issues:
            corrupt_files.append((str(npz_path), issues))
    
    return corrupt_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan SAM cache for corrupt npz files")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to SAM cache directory")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to check (for quick scan)")
    parser.add_argument("--delete", action="store_true", help="Delete corrupt files")
    args = parser.parse_args()
    
    corrupt_files = scan_sam_cache(args.cache_dir, args.max_files)
    
    if corrupt_files:
        print(f"\n{'='*80}")
        print(f"Found {len(corrupt_files)} corrupt npz files:")
        print(f"{'='*80}\n")
        
        for path, issues in corrupt_files:
            print(f"❌ {path}")
            for issue in issues:
                print(f"   └─ {issue}")
        
        if args.delete:
            print(f"\n{'='*80}")
            print(f"DELETING {len(corrupt_files)} corrupt files...")
            print(f"{'='*80}\n")
            
            for path, _ in corrupt_files:
                try:
                    os.remove(path)
                    print(f"✓ Deleted: {path}")
                except Exception as e:
                    print(f"✗ Failed to delete {path}: {e}")
        else:
            print(f"\n💡 To delete these files, run with --delete flag")
    else:
        print("\n✅ All npz files are valid!")
