import numpy as np
import torch
from torch.utils.data import Dataset


class RandomRepProvider(Dataset):
    def __init__(self, num_samples, rep_dim):
        self.num_samples = int(num_samples)
        self.rep_dim = int(rep_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        _ = idx
        return torch.randn(self.rep_dim, dtype=torch.float32)


class NPZRepDataset(Dataset):
    def __init__(self, npz_path, key=None, mmap_mode="r"):
        data = np.load(npz_path, mmap_mode=mmap_mode)
        if key is None:
            if "reps" in data.files:
                key = "reps"
            elif "embeddings" in data.files:
                key = "embeddings"
            elif len(data.files) == 1:
                key = data.files[0]
            else:
                raise ValueError("NPZ contains multiple arrays; specify --npz_key.")
        if key not in data.files:
            raise KeyError(f"Key '{key}' not found in NPZ file.")
        reps = data[key]
        if reps.ndim == 4 and reps.shape[-2:] == (1, 1):
            reps = reps[:, :, 0, 0]
        if reps.ndim != 2:
            raise ValueError(f"Expected reps of shape [N, D], got {reps.shape}.")
        self.reps = np.ascontiguousarray(reps, dtype=np.float32)
        self.rep_dim = self.reps.shape[1]

    def __len__(self):
        return self.reps.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.reps[idx])
