import importlib
import math
import os
import random
from inspect import isfunction

import numpy as np
import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config):
    if config is None:
        return None
    if not hasattr(config, "get"):
        raise TypeError("Config must be a mapping with a 'target' entry.")
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")
    target = config["target"]
    params = config.get("params", {})
    return get_obj_from_str(target)(**params)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, args):
    if not getattr(args, "cosine_lr", False):
        return optimizer.param_groups[0]["lr"]
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / max(1, args.warmup_epochs)
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(path, model, optimizer=None, scaler=None, epoch=None, model_ema=None, args=None):
    state = {"model": model.state_dict()}
    if model_ema is not None:
        state["model_ema"] = model_ema.state_dict()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if args is not None:
        state["args"] = vars(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, model_ema=None):
    if not path:
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if model_ema is not None and "model_ema" in checkpoint:
        model_ema.load_state_dict(checkpoint["model_ema"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("epoch", 0)
