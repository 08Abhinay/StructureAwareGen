# Quick Start Guide: SAM On-the-Fly Extraction

## Prerequisites

1. **Install segment_anything**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

2. **Download SAM checkpoint**:
```bash
# Base model (fastest, recommended)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or Large model (better quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Or Huge model (best quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

3. **Prepare your dataset**:
```
dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
└── ...
```

## Step 1: Test SAM Integration

```bash
cd scripts/StyleGAN2/seg-aware-stylegan2/

# Copy a test image to current directory
cp /path/to/test/image.jpg ./test_image.jpg

# Run unit tests
python training/test_sam_integration.py
```

Expected output:
```
==============================================================
SAM EXTRACTOR INTEGRATION TESTS
==============================================================

==============================================================
TEST 1: Single Image Extraction
==============================================================
✓ SAMExtractor created
✓ Extraction completed in 12.34s
✓ All validations passed

==============================================================
TEST 2: Cache Save/Load
==============================================================
✓ First extraction: 12.34s
✓ Second extraction: 0.15s
✓ Cache speedup: 82.3x faster
✓ All cache tests passed

...

✓ PASSED: Single Extraction
✓ PASSED: Cache Save/Load
✓ PASSED: Batch Padding
✓ PASSED: Stochastic Conditioning

Total: 4/4 tests passed

🎉 All tests passed!
```

## Step 2: Run Training

### Basic Training (No Pre-Training)

```bash
python train.py \
    --outdir=./training-runs \
    --data=./datasets/your_dataset \
    --gpus=8 \
    --cfg=auto \
    --kimg=25000 \
    --sam-enabled=true \
    --sam-prob=0.25 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=./sam_cache
```

### With I-JEPA Guidance

```bash
python train.py \
    --outdir=./training-runs \
    --data=./datasets/your_dataset \
    --gpus=8 \
    --cfg=auto \
    --kimg=25000 \
    --ijepa_checkpoint=./checkpoints/ijepa_vit.pth \
    --ijepa_lambda=1.0 \
    --ijepa_warmup_kimg=500 \
    --sam-enabled=true \
    --sam-prob=0.25 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=./sam_cache
```

### Resume from Checkpoint

```bash
python train.py \
    --outdir=./training-runs \
    --data=./datasets/your_dataset \
    --gpus=8 \
    --cfg=auto \
    --kimg=25000 \
    --resume=./training-runs/00000-your-run/network-snapshot-001234.pkl \
    --sam-enabled=true \
    --sam-prob=0.25 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=./sam_cache
```

## Step 3: Monitor Training

### Check Logs

```bash
tail -f training-runs/00000-your-run/log.txt
```

Look for:
```
[SAMExtractor] SAM model loaded successfully
[Loss] SAM extractor enabled with prob=0.25
Cache stats: hits=1250, misses=250, hit_rate=0.833
tick 10    kimg 10.0      Loss/G/loss 1.234
```

### Monitor Cache Growth

```bash
# Check cache size
du -sh ./sam_cache

# Count cached images
find ./sam_cache -name "*.npz" | wc -l

# Monitor in real-time
watch -n 5 'find ./sam_cache -name "*.npz" | wc -l'
```

### TensorBoard (if available)

```bash
tensorboard --logdir=./training-runs
```

## Step 4: Experiment with sam_prob

Try different values to find the sweet spot:

```bash
# More regularization (fewer SAM batches)
--sam-prob=0.10  # 10% of batches

# Recommended default
--sam-prob=0.25  # 25% of batches

# More SAM guidance
--sam-prob=0.50  # 50% of batches

# Always use SAM (no stochastic conditioning)
--sam-prob=1.0   # 100% of batches
```

## Troubleshooting

### Problem: "segment_anything not found"
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Problem: "SAM checkpoint not found"
Make sure you downloaded it:
```bash
ls -lh sam_vit_b_01ec64.pth
# Should show ~375MB file
```

### Problem: "Out of memory"
Reduce memory usage:
```bash
--sam-max-masks=100        # Reduce from 250
--batch=16                 # Reduce batch size
--sam-model-type=vit_b     # Use base model (not vit_l or vit_h)
```

### Problem: "Training very slow first epoch"
This is **expected**! Cache is cold. Solutions:
- Be patient - it gets faster after ~10 epochs
- Or pre-compute embeddings (see advanced usage)
- Monitor cache hit rate in logs

### Problem: "Cache taking too much disk space"
```bash
# Check cache size
du -sh ./sam_cache

# Clean cache if needed
rm -rf ./sam_cache
```

Cache size estimate:
- ~50KB per image (250 masks × 256 dims × 2 bytes)
- 1M images = ~50GB
- Compression: ~10x smaller than original images

## Advanced Usage

### Share Cache Between Experiments

All runs using same dataset can share cache:
```bash
# Run 1
python train.py ... --sam-cache-dir=/shared/sam_cache

# Run 2 (reuses cache from Run 1!)
python train.py ... --sam-cache-dir=/shared/sam_cache
```

### Multi-Dataset Training

Different datasets need separate caches:
```bash
# ImageNet
python train.py ... --sam-cache-dir=./cache_imagenet

# COCO
python train.py ... --sam-cache-dir=./cache_coco
```

### Hybrid: Pre-Computed + On-the-Fly

Not supported yet, but you can:
1. Pre-compute 75% of dataset
2. Use on-the-fly for remaining 25%

## Performance Tips

1. **Start with small sam_prob**: Try 0.1-0.25 first
2. **Use base model**: `vit_b` is 5x faster than `vit_h`
3. **Share cache**: Across multiple runs
4. **SSD storage**: Put cache on SSD (not HDD)
5. **Monitor hit rate**: Should reach >90% by epoch 10

## Next Steps

1. **Read full documentation**: [SAM_INTEGRATION_README.md](SAM_INTEGRATION_README.md)
2. **Adjust sam_prob**: Experiment with different values
3. **Try different models**: Compare vit_b vs vit_l
4. **Monitor quality**: Check FID/IS metrics
5. **Share results**: Report findings!

## Example Commands

### Toy Dataset (Quick Test)
```bash
python train.py \
    --outdir=./test-runs \
    --data=./test_data \
    --gpus=1 \
    --cfg=auto \
    --kimg=100 \
    --sam-enabled=true \
    --sam-prob=1.0 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=./test_cache
```

### CIFAR-10
```bash
python train.py \
    --outdir=./cifar10-runs \
    --data=./datasets/cifar10 \
    --gpus=2 \
    --cfg=cifar \
    --sam-enabled=true \
    --sam-prob=0.25 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=./cache_cifar10
```

### ImageNet (Full Scale)
```bash
python train.py \
    --outdir=./imagenet-runs \
    --data=/datasets/imagenet \
    --gpus=8 \
    --cfg=auto \
    --kimg=25000 \
    --batch=32 \
    --sam-enabled=true \
    --sam-prob=0.25 \
    --sam-checkpoint=./sam_vit_b_01ec64.pth \
    --sam-cache-dir=/shared/cache_imagenet \
    --ijepa_checkpoint=./checkpoints/ijepa_vit.pth \
    --ijepa_lambda=1.0
```

## Expected Timeline

### First Run (Cold Cache)
- **Epoch 1**: Slow (~25% slowdown if sam_prob=0.25)
- **Epoch 5**: Faster (~15% slowdown)
- **Epoch 10**: Nearly full speed (~5% slowdown)
- **Epoch 15+**: Full speed (cache ~99% hit rate)

### Subsequent Runs (Warm Cache)
- **All epochs**: Full speed (cache hit rate ~100%)

## Support

- GitHub Issues: [link]
- Documentation: [SAM_INTEGRATION_README.md](SAM_INTEGRATION_README.md)
- Email: [your-email]
