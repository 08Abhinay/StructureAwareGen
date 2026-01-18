import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from seg_rdm.data.reps_dataset import NPZRepDataset, RandomRepProvider
from seg_rdm.engine_rdm import train_one_epoch
from seg_rdm.utils import distributed as dist_utils
from seg_rdm.utils.logging import create_summary_writer
from seg_rdm.utils.misc import instantiate_from_config, load_checkpoint, save_checkpoint, seed_all


def get_args_parser():
    parser = argparse.ArgumentParser("SEG-RDM training", add_help=False)
    parser.add_argument("--config", type=str, default="seg_rdm/configs/default_rdm.yaml")

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_epochs", default=0, type=int)

    parser.add_argument("--rep_source", type=str, choices=["random", "npz"], default="random")
    parser.add_argument("--npz_path", type=str, default="")
    parser.add_argument("--npz_key", type=str, default="")
    parser.add_argument("--rep_dim", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=100000)

    parser.add_argument("--output_dir", default="./seg_rdm_out")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")

    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")

    return parser


def _override_rep_dim(cfg, rep_dim):
    cfg.rep_dim = int(rep_dim)
    cfg.model.params.channels = int(rep_dim)
    cfg.model.params.image_size = 1
    cfg.model.params.unet_config.params.in_channels = int(rep_dim)
    cfg.model.params.unet_config.params.out_channels = int(rep_dim)
    if "time_embed_dim" in cfg.model.params.unet_config.params:
        cfg.model.params.unet_config.params.time_embed_dim = int(rep_dim)


def _build_dataset(args, rep_dim):
    if args.rep_source == "random":
        dataset = RandomRepProvider(num_samples=args.num_samples, rep_dim=rep_dim)
    elif args.rep_source == "npz":
        if not args.npz_path:
            raise ValueError("--npz_path is required when rep_source=npz")
        key = args.npz_key if args.npz_key else None
        dataset = NPZRepDataset(args.npz_path, key=key)
    else:
        raise ValueError(f"Unknown rep_source: {args.rep_source}")
    return dataset


def smoke_test(cfg, device):
    model = instantiate_from_config(cfg.model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        reps = torch.randn(2, cfg.rep_dim, device=device)
        loss, loss_dict = model(reps)
        print(f"Smoke test loss: {loss.item():.6f}")
        print(f"Smoke test loss_dict: {loss_dict}")
        samples = model.sample(batch_size=2)
        print(f"Smoke test samples shape: {samples.shape}")


def main(args):
    dist_utils.init_distributed_mode(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        if dist_utils.is_main_process():
            print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    if args.distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")

    seed_all(args.seed + dist_utils.get_rank())

    if args.log_dir is None:
        args.log_dir = args.output_dir

    log_writer = None
    if dist_utils.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = create_summary_writer(args.log_dir)

    cfg = OmegaConf.load(args.config)

    if args.rep_dim is not None:
        _override_rep_dim(cfg, args.rep_dim)

    dataset = _build_dataset(args, cfg.rep_dim)

    if args.rep_source == "npz":
        if args.rep_dim is None and dataset.rep_dim != cfg.rep_dim:
            _override_rep_dim(cfg, dataset.rep_dim)

    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = instantiate_from_config(cfg.model)
    model.to(device)

    if args.smoke_test:
        smoke_test(cfg, device)
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size

    if dist_utils.is_main_process():
        print(f"Effective batch size: {eff_batch_size}")
        print(f"Base LR: {args.lr / eff_batch_size:.2e}")
        print(f"Actual LR: {args.lr:.2e}")

    if args.amp and device.type != "cuda":
        if dist_utils.is_main_process():
            print("AMP requested but CUDA is not available; disabling AMP.")
        args.amp = False
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    start_epoch = 0
    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint-last.pt")
        if os.path.exists(resume_path):
            start_epoch = load_checkpoint(resume_path, model_without_ddp, optimizer, scaler, getattr(model_without_ddp, "model_ema", None))
            if dist_utils.is_main_process():
                print(f"Resumed from {resume_path} at epoch {start_epoch}")
        elif dist_utils.is_main_process():
            print(f"Resume path not found: {resume_path}")

    if dist_utils.is_main_process():
        trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    if dist_utils.is_main_process() and args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if dist_utils.is_main_process():
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader,
            optimizer,
            device,
            epoch,
            scaler,
            log_writer=log_writer,
            args=args,
        )

        if dist_utils.is_main_process() and args.output_dir:
            last_path = os.path.join(args.output_dir, "checkpoint-last.pt")
            save_checkpoint(
                last_path,
                model_without_ddp,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                model_ema=getattr(model_without_ddp, "model_ema", None),
                args=args,
            )
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                epoch_path = os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pt")
                save_checkpoint(
                    epoch_path,
                    model_without_ddp,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    model_ema=getattr(model_without_ddp, "model_ema", None),
                    args=args,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if dist_utils.is_main_process():
        print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SEG-RDM training", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
