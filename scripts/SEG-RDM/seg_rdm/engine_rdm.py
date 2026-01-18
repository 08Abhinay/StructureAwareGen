import math

import torch

from seg_rdm.utils import distributed as dist_utils
from seg_rdm.utils.logging import MetricLogger
from seg_rdm.utils.misc import adjust_learning_rate


def train_one_epoch(model, data_loader, optimizer, device, epoch, scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, reps in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        reps = reps.to(device, non_blocking=True)

        if args.amp:
            amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss, loss_dict = model(reps)
        else:
            loss, loss_dict = model(reps)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_value)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = dist_utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            step = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, step)
            log_writer.add_scalar("lr", lr, step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
