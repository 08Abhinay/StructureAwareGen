"""
Quick verification that SEG-RDM can load and forward-pass with REN DINOv2 test H5.
Run on a GPU node:
    python verify_seg_rdm_ren.py
"""
import os, sys
sys.path.insert(0, "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM")

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Point H5 to the small test file
TEST_H5 = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_ren_dinov2_test.h5"
CONFIG  = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/configs/unified_seg_rdm_ren_dinov2.yaml"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    cfg = OmegaConf.load(CONFIG)

    # Override H5 path to test file
    cfg.data.params.h5_path = TEST_H5

    # ── 1. Dataset verification ──
    print("\n=== Dataset Verification ===")
    from rdm.data.seg_dataset import SegmentationMaskDataset
    ds = SegmentationMaskDataset(**cfg.data.params)
    print(f"  Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"  seg_embs shape:   {sample['seg_embs'].shape}")
    print(f"  num_segments:     {sample['num_segments']}")
    print(f"  scores shape:     {sample['scores'].shape}")
    print(f"  cls_emb shape:    {sample['cls_emb'].shape}")
    print(f"  image shape:      {sample['image'].shape}")
    print(f"  emb_source:       {sample['emb_source']}")

    expected_max_seg = cfg.model.params.max_segments
    expected_dim = cfg.model.params.channels
    assert sample['seg_embs'].shape == (expected_max_seg, expected_dim), \
        f"seg_embs shape mismatch: {sample['seg_embs'].shape} != ({expected_max_seg}, {expected_dim})"
    assert sample['cls_emb'].shape == (expected_dim,), \
        f"cls_emb shape mismatch: {sample['cls_emb'].shape} != ({expected_dim},)"
    print("  ✓ Shape assertions passed")

    # ── 2. DataLoader verification ──
    print("\n=== DataLoader Verification ===")
    from rdm.data.seg_dataset import collate_seg_batch
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                    collate_fn=collate_seg_batch)
    batch = next(iter(dl))
    print(f"  Batch seg_embs:    {batch['seg_embs'].shape}")
    print(f"  Batch cls_emb:     {batch['cls_emb'].shape}")
    print(f"  Batch num_segments: {batch['num_segments'].tolist()}")

    # ── 3. Model instantiation ──
    print("\n=== Model Verification ===")
    from rdm.util import instantiate_from_config
    model = instantiate_from_config(cfg.model).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable params: {n_train:,}")

    # ── 4. Forward pass ──
    print("\n=== Forward Pass ===")
    # Move batch to device
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    with torch.no_grad():
        result = model.get_input(batch_gpu, "image")
        x, c, padding_mask = result[0], result[1], result[2]
        print(f"  x shape (diffusion): {x.shape}")
        print(f"  padding_mask shape:  {padding_mask.shape}")
        print(f"  padding_mask sum:    {padding_mask.sum(dim=1).tolist()}")
        print(f"  conditioning:        {c}")

    # Test actual forward (loss computation)
    model.train()
    loss, loss_dict = model(None, None, batch=batch_gpu)
    print(f"\n  Loss: {loss.item():.4f}")
    for k, v in sorted(loss_dict.items()):
        print(f"  {k}: {v.item():.4f}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
