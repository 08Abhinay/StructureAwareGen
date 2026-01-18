import os

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def all_reduce_mean(value):
    world_size = get_world_size()
    if world_size == 1:
        return value
    tensor = torch.tensor(value, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    dist.all_reduce(tensor)
    tensor /= world_size
    return tensor.item()


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
