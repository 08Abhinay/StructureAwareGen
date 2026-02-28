# ============================================================
# Extract ijepa_embeddings.h5 → folder of NPZ files
# Source: h5_embeddings/ijepa_embeddings.h5
#   H5 layout: scratch/gilbreth/.../ijepa_embeddings/{class_id}/{sample_id}/emb
#   Each leaf: emb (1280,) float32
# Output: {DST_ROOT}/{class_id}/{sample_id}.npz  with key "emb"
# ============================================================
import os, h5py, numpy as np, time
from tqdm import tqdm

SRC_H5   = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/ijepa_embeddings.h5"
DST_ROOT = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/ijepa_embeddings"
# Internal H5 prefix that mirrors the original filesystem path
H5_PREFIX = "scratch/gilbreth/abelde/Thesis/StructureAwareGen/ijepa_embeddings"

print(f"Source H5 : {SRC_H5}")
print(f"Output dir: {DST_ROOT}")

t0 = time.time()
os.makedirs(DST_ROOT, exist_ok=True)

with h5py.File(SRC_H5, "r") as f:
    base = f[H5_PREFIX]
    classes = sorted(base.keys(), key=lambda x: int(x))
    print(f"Classes: {len(classes)}")

    total = 0
    errors = []
    for ci, cls_id in enumerate(tqdm(classes, desc="Extracting classes")):
        cls_grp = base[cls_id]
        cls_dir = os.path.join(DST_ROOT, cls_id)
        os.makedirs(cls_dir, exist_ok=True)

        for sample_name in cls_grp:
            try:
                item = cls_grp[sample_name]
                # Could be a Group with 'emb' key, or a Dataset directly
                if isinstance(item, h5py.Group):
                    emb = item["emb"][:]
                elif isinstance(item, h5py.Dataset):
                    emb = item[:]
                else:
                    continue

                np.savez_compressed(
                    os.path.join(cls_dir, f"{sample_name}.npz"),
                    emb=emb,
                )
                total += 1
            except Exception as e:
                errors.append((cls_id, sample_name, str(e)))

elapsed = time.time() - t0
print(f"\nDone — {total:,} files written to {DST_ROOT}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
if errors:
    print(f"Errors: {len(errors)}")
    for path_cls, path_name, msg in errors[:10]:
        print(f"  {path_cls}/{path_name}: {msg}")
