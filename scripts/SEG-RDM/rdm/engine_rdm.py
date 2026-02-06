import math
import sys
from typing import Iterable

import torch

from rdm import util
# from rdm.env_debug import print_env
# print_env(__name__, globals())



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, class_label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # iterable = metric_logger.log_every(data_loader, print_freq, header)

        ##############
        ##############
    # iterable = data_loader  # <-- debugger friendly
    # # for data_iter_step, batch in enumerate(iterable):
    # it = iter(iterable)      # or iter(data_loader) to bypass logger
    # data_iter_step = 0

    # while True:
    #     try:
    #         batch = next(it)   # <-- put breakpoint here
    #     except StopIteration:
    #         break
    #     samples, class_label = batch
        ###############
        ##############
        
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            util.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        class_label = class_label.to(device, non_blocking=True)
        if args.class_cond:
            batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": class_label}
        else:
            batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": torch.zeros_like(class_label)}
        loss, loss_dict = model(x=None, c=None, batch=batch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = util.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        # data_iter_step += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_seg(model: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        loss_scaler,
                        log_writer=None,
                        args=None):
    """
    Training loop for SegmentationMaskDataset (dict batches).
    Identical structure to train_one_epoch() but handles dict batches.
    """
    model.train(True)
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            util.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move batch dict to device
        # batch keys: image, seg_embs, num_segments, scores, filename
        batch_device = {
            'image': batch['image'].to(device, non_blocking=True),
            'seg_embs': batch['seg_embs'].to(device, non_blocking=True),
            'num_segments': batch['num_segments'].to(device, non_blocking=True),
        }

        # Add IJEPA embeddings if available (for cached mode)
        if 'ijepa_emb' in batch:
            batch_device['ijepa_emb'] = batch['ijepa_emb'].to(device, non_blocking=True)
        
        # Forward pass - UnifiedSegRDM.forward() accepts batch=dict
        loss, loss_dict = model(x=None, c=None, batch=batch_device)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        # Log additional losses from loss_dict if available
        if loss_dict:
            for k, v in loss_dict.items():
                if 'loss' in k.lower():
                    metric_logger.update(**{k: v.item() if hasattr(v, 'item') else v})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = util.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            # Log auxiliary losses (diversity, alignment)
            if loss_dict:
                for k, v in loss_dict.items():
                    if 'loss' in k.lower():
                        log_writer.add_scalar(k, v.item() if hasattr(v, 'item') else v, epoch_1000x)
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}