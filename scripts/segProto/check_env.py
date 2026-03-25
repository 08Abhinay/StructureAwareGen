#!/usr/bin/env python
"""Quick environment check."""
import sys
print(f"Python: {sys.version}")

try:
    import timm
    print(f"timm: {timm.__version__}")
except ImportError:
    print("timm: NOT INSTALLED")

try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch: NOT INSTALLED")

try:
    import kornia
    print(f"kornia: {kornia.__version__}")
except ImportError:
    print("kornia: NOT INSTALLED")

try:
    import fast_slic
    print(f"fast_slic: OK")
except ImportError:
    print("fast_slic: NOT INSTALLED")

try:
    import h5py
    print(f"h5py: {h5py.__version__}")
except ImportError:
    print("h5py: NOT INSTALLED")

try:
    import scipy
    print(f"scipy: {scipy.__version__}")
except ImportError:
    print("scipy: NOT INSTALLED")
