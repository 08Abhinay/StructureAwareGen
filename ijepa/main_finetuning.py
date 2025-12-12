# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from engine_finetune import main as app_main

import os

# if "LOCAL_RANK" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    
parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


# if __name__ == '__main__':
#     args = parser.parse_args()

#     num_gpus = len(args.devices)
#     mp.set_start_method('spawn')

#     for rank in range(num_gpus):
#         mp.Process(
#             target=process_main,
#             args=(rank, args.fname, num_gpus, args.devices)
#         ).start()

# ---------------------------------------------------------
#   main_finetuning.py  (add this just above the last line)
# ---------------------------------------------------------
if __name__ == '__main__':
    import os, torch, sys
    args = parser.parse_args()

    # --- Case ➊: we were started *by torchrun* -------------
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        # let torchrun decide which GPU this rank sees
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        process_main(local_rank, args.fname, world_size, devices)
        sys.exit(0)

    # --- Case ➋: vanilla python -> fall back to old spawn --
    num_gpus = len(args.devices)
    mp.set_start_method('spawn', force=True)
    for rank in range(num_gpus):
        mp.Process(target=process_main,
                   args=(rank, args.fname, num_gpus, args.devices)).start()
