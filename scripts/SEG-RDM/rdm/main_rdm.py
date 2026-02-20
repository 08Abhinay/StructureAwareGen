import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from omegaconf import OmegaConf

from rdm import util
from rdm.engine_rdm import train_one_epoch
from rdm.models.diffusion.ddim import DDIMSampler
# from rdm.env_debug import print_env
# print_env(__name__, globals())



def get_args_parser():
    parser = argparse.ArgumentParser('RDM training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # config
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--config', type=str, help='config file')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--cosine_lr', action='store_true',
                        help='Use cosine lr scheduling.')
    parser.add_argument('--warmup_epochs', default=0, type=int)
    
    # Segmentation dataset parameters
    parser.add_argument('--use_seg_dataset', action='store_true',
                        help='Use SegmentationMaskDataset with pre-computed SAM embeddings')
    parser.add_argument('--mask_npz_dir', type=str, default=None,
                        help='Directory containing SAM .npz embeddings (required if use_seg_dataset=True)')
    parser.add_argument('--max_segments', type=int, default=250,
                        help='Maximum number of segments for padding')
    parser.add_argument('--ijepa_cache_dir', type=str, default=None,
                        help='Optional directory with pre-cached IJEPA embeddings (speeds up training)')
    parser.add_argument('--emb_source', type=str, default='sam',
                        choices=['sam', 'ijepa', 'dinov2', 'dino'],
                        help='Embedding source: sam (SAM encoder), ijepa/dinov2/dino (ViT patch tokens). '
                             'Non-sam sources expect emb_image_mean in npz for mean-subtracted embeddings.')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Debug / smoke mode
    parser.add_argument('--debug', action='store_true',
                        help='Debugger-friendly mode: num_workers=0, pin_mem=False, no DDP sampler, small subset.')
    parser.add_argument('--debug_n', default=100, type=int,
                        help='Number of training images to use in --debug mode.')
    parser.add_argument('--debug_steps', default=5, type=int,
                        help='Max train iterations per epoch in --debug mode.')


    return parser


def smoke_test(config_path, device='cuda', batch_size=2, num_steps=5):
    config = OmegaConf.load(config_path)
    model = util.instantiate_from_config(config.model)
    model.to(device)
    model.eval()

    images = torch.rand(batch_size, 3, 256, 256, device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    batch = {
        "image": images.permute(0, 2, 3, 1),
        "class_label": labels,
    }

    with torch.no_grad():
        loss, loss_dict = model(x=None, c=None, batch=batch)
        _, cond = model.get_input(batch, model.first_stage_key)
        if isinstance(cond, (dict, list)):
            cond = model.get_learned_conditioning(cond)
        sampler = DDIMSampler(model)
        samples, _ = sampler.sample(
            S=num_steps,
            batch_size=batch_size,
            shape=(model.channels, model.image_size, model.image_size),
            conditioning=cond,
            verbose=False,
        )

    return loss, loss_dict, samples


def main(args):
    util.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = util.get_world_size()
    global_rank = util.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # -------------------------
    # Conditional Dataset Loading
    # -------------------------
    if args.use_seg_dataset:
        # Import SegmentationMaskDataset
        from rdm.data.seg_dataset import SegmentationMaskDataset
        
        # Validate required arguments
        if args.mask_npz_dir is None:
            raise ValueError("--mask_npz_dir must be specified when --use_seg_dataset is True")
        
        # Create segmentation dataset (will filter to only images with SAM embeddings)
        dataset_train = SegmentationMaskDataset(
            image_dir=os.path.join(args.data_path, 'train'),
            mask_npz_dir=args.mask_npz_dir,
            max_segments=args.max_segments,
            image_size=args.input_size,
            file_ext="*.JPEG",  # ImageNet uses .JPEG
            normalize=True,  # DDPM expects [-1, 1] range
            ijepa_cache_dir=args.ijepa_cache_dir,  # Optional pre-cached IJEPA embeddings
            emb_source=args.emb_source,  # sam, ijepa, dinov2, dino
        )
        print(f"Using SegmentationMaskDataset: {len(dataset_train)} samples")
        print(f"  Embeddings: {args.mask_npz_dir}")
        print(f"  emb_source: {args.emb_source}")
        print(f"  max_segments: {args.max_segments}")
        if args.ijepa_cache_dir:
            print(f"  IJEPA cache: {args.ijepa_cache_dir} (pre-cached embeddings)")
        else:
            print(f"  IJEPA cache: None (runtime extraction)")
        
    else:
        # Original ImageFolder pipeline (unchanged)
        transform_train = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(f"Using ImageFolder: {len(dataset_train)} samples")
    
    # -------------------------
    # Debug subset (100 images)
    # -------------------------
    if args.debug:
        from torch.utils.data import Subset
        n = min(args.debug_n, len(dataset_train))
        dataset_train = Subset(dataset_train, list(range(n)))
        print(f"[DEBUG] Using subset of dataset_train: {n} samples")


    # -------------------------
    # Conditional collate function
    # -------------------------
    if args.use_seg_dataset:
        from rdm.data.seg_dataset import collate_seg_batch
        collate_fn = collate_seg_batch
        print("Using collate_seg_batch for dict batches")
    else:
        collate_fn = None
    
    # -------------------------
    # Sampler / DataLoader
    # -------------------------
    if args.debug:
        sampler_train = None
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=False,            # deterministic
            batch_size=args.batch_size,
            num_workers=0,            # KEY
            pin_memory=False,         # KEY
            drop_last=False,
            collate_fn=collate_fn,
        )
        print("[DEBUG] DataLoader: num_workers=0, pin_memory=False, shuffle=False, no DistributedSampler")
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_fn,
        )

    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_train = %s" % str(sampler_train))

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    config = OmegaConf.load(args.config)
    model = util.instantiate_from_config(config.model)

    args.class_cond = config.model.params.get("class_cond", False)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * util.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size

    print("base lr: %.2e" % (args.lr / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.gpu], find_unused_parameters=True
    #     )
    #     model_without_ddp = model.module
    if args.distributed and (not args.debug):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module


    params = list(model_without_ddp.model.parameters())
    if model_without_ddp.cond_stage_model is not None:
        params += list(model_without_ddp.cond_stage_model.parameters())
    # Include alignment projection only when trainable.
    if hasattr(model_without_ddp, 'seg_align_proj'):
        align_proj_params = [p for p in model_without_ddp.seg_align_proj.parameters() if p.requires_grad]
        if align_proj_params:
            params += align_proj_params
            print(
                f"  Added seg_align_proj to optimizer "
                f"({sum(p.numel() for p in align_proj_params)} trainable params)"
            )
        else:
            print("  seg_align_proj is frozen (not added to optimizer)")
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    if global_rank == 0:
        log_writer.add_scalar('num_params', n_params / 1e6, 0)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = util.NativeScalerWithGradNormCount()

    util.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        if args.distributed and (not args.debug):
            data_loader_train.sampler.set_epoch(epoch)

        # Conditional training function based on dataset type
        if args.use_seg_dataset:
            from rdm.engine_rdm import train_one_epoch_seg
            train_stats = train_one_epoch_seg(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        # ---- Extract EMA parameters (if model uses LitEma) ---- #
        ema_params = None
        if hasattr(model_without_ddp, 'model_ema') and model_without_ddp.model_ema is not None:
            ema_module = model_without_ddp.model_ema
            # LitEma stores buffers with dots stripped from param names.
            # Reverse-map: s_name (dot-free) -> original dotted name
            s2m = {v: k for k, v in ema_module.m_name2s_name.items()}
            buffers = dict(ema_module.named_buffers())
            ema_params = []
            for name, _p in model_without_ddp.named_parameters():
                s_name = ema_module.m_name2s_name.get(name)
                if s_name is not None and s_name in buffers:
                    ema_params.append(buffers[s_name])
                else:
                    ema_params.append(_p.data)  # fallback: use raw weight

        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            util.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, config=config)

        util.save_model_last(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, config=config)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and util.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
