#!/bin/bash
# Quick start script for Unified Segmentation RDM
# Run this to verify your setup before training

set -e  # Exit on error

echo "==================================="
echo "Unified Segmentation RDM - Quick Start"
echo "==================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "\n${YELLOW}[1/6] Checking Python environment...${NC}"
python3 --version || { echo -e "${RED}Python 3 not found!${NC}"; exit 1; }
echo -e "${GREEN}✓ Python OK${NC}"

# Check PyTorch
echo -e "\n${YELLOW}[2/6] Checking PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" || \
    { echo -e "${RED}PyTorch not found! Install with: pip install torch${NC}"; exit 1; }
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo -e "${GREEN}✓ PyTorch OK${NC}"

# Check dependencies
echo -e "\n${YELLOW}[3/6] Checking dependencies...${NC}"
python3 -c "import numpy, PIL, tqdm, yaml, omegaconf, einops" || \
    { echo -e "${RED}Missing dependencies! Install with: pip install numpy pillow tqdm pyyaml omegaconf einops${NC}"; exit 1; }
echo -e "${GREEN}✓ Dependencies OK${NC}"

# Check data paths
echo -e "\n${YELLOW}[4/6] Checking data paths...${NC}"

# Read paths from config
IMAGE_DIR=$(python3 -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('rdm/configs/unified_seg_rdm.yaml'); print(cfg.data.params.image_dir)")
MASK_DIR=$(python3 -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('rdm/configs/unified_seg_rdm.yaml'); print(cfg.data.params.mask_npz_dir)")

if [ -d "$IMAGE_DIR" ]; then
    NUM_IMAGES=$(find "$IMAGE_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo -e "${GREEN}✓ Image directory found: $IMAGE_DIR ($NUM_IMAGES images)${NC}"
else
    echo -e "${RED}✗ Image directory not found: $IMAGE_DIR${NC}"
    echo -e "${YELLOW}  Update image_dir in rdm/configs/unified_seg_rdm.yaml${NC}"
fi

if [ -d "$MASK_DIR" ]; then
    NUM_MASKS=$(find "$MASK_DIR" -name "*.npz" | wc -l)
    echo -e "${GREEN}✓ Mask directory found: $MASK_DIR ($NUM_MASKS npz files)${NC}"
else
    echo -e "${RED}✗ Mask directory not found: $MASK_DIR${NC}"
    echo -e "${YELLOW}  Update mask_npz_dir in rdm/configs/unified_seg_rdm.yaml${NC}"
    echo -e "${YELLOW}  Run segmentation pipeline first: scripts/segProto/segmentation-play.ipynb${NC}"
fi

# Test dataset loader
echo -e "\n${YELLOW}[5/6] Testing dataset loader...${NC}"
python3 rdm/data/seg_dataset.py || \
    { echo -e "${RED}Dataset test failed! Check paths in config.${NC}"; exit 1; }
echo -e "${GREEN}✓ Dataset loader OK${NC}"

# Test transformer
echo -e "\n${YELLOW}[6/6] Testing UnifiedSegTransformer...${NC}"
python3 rdm/modules/diffusionmodules/unified_transformer.py || \
    { echo -e "${RED}Transformer test failed!${NC}"; exit 1; }
echo -e "${GREEN}✓ Transformer OK${NC}"

# Summary
echo -e "\n==================================="
echo -e "${GREEN}✓ All checks passed!${NC}"
echo -e "==================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Update paths in rdm/configs/unified_seg_rdm.yaml:"
echo "   - image_dir: Path to your images"
echo "   - mask_npz_dir: Path to SAM .npz files"
echo "   - pretrained_enc_path: Path to I-JEPA checkpoint"
echo ""
echo "2. Start training:"
echo "   python3 train_unified_seg_rdm.py --config rdm/configs/unified_seg_rdm.yaml"
echo ""
echo "3. Monitor with Weights & Biases or TensorBoard"
echo ""
echo "4. Generate samples after training:"
echo "   python3 sample_unified_seg_rdm.py \\"
echo "       --config rdm/configs/unified_seg_rdm.yaml \\"
echo "       --checkpoint checkpoints/unified_seg_rdm/checkpoint_ema_step_XXXXXX.pt \\"
echo "       --num_samples 100"
echo ""
echo -e "${YELLOW}See README_UNIFIED_SEG_RDM.md for detailed instructions${NC}"
echo ""
