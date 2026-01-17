# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, capture_labels=False, max_items=None):
        self.capture_all      = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.capture_labels   = capture_labels
        self.max_items        = max_items
        self.num_items        = 0
        self.num_features     = None
        self.all_features     = None
        self.raw_mean         = None
        self.raw_cov          = None
        if self.capture_labels:
            self.all_labels    = []

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean     = np.zeros([num_features], dtype=np.float64)
            self.raw_cov      = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        # truncate if exceeding max_items
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[: self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov  += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        # collect features from all GPUs
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)  # interleave
        self.append(x.cpu().numpy())

    def append_labels(self, labels):
        """Call this in tandem with append_torch to collect the batch’s class‐ids."""
        if not self.capture_labels:
            return
        # convert to numpy
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.asarray(labels)
        # truncate to max_items if needed
        if self.max_items is not None:
            collected = sum(len(arr) for arr in self.all_labels)
            if collected >= self.max_items:
                return
            take = min(len(labels), self.max_items - collected)
            labels = labels[:take]
        self.all_labels.append(labels)

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov  = self.raw_cov / self.num_items
        cov  = cov - np.outer(mean, mean)
        return mean, cov

    @property
    def labels(self):
        """Returns all captured labels as a 1D torch tensor."""
        assert self.capture_labels
        return torch.from_numpy(np.concatenate(self.all_labels, axis=0))

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        # re-construct with the same flags
        obj = FeatureStats(
            capture_all      = s.capture_all,
            capture_mean_cov = s.capture_mean_cov,
            capture_labels   = getattr(s, 'capture_labels', False),
            max_items        = s.max_items
        )
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    data_loader_kwargs=None,
    max_items=None,
    capture_labels=False,       # <-- new flag
    **stats_kwargs
):
    """
    Collects features *and* optionally labels from a dataset, using a pretrained
    feature detector. If capture_labels=True, the returned FeatureStats will have
    a `.labels` attribute (a torch.Tensor of shape (N,) on opts.device).
    """
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to load from cache.
    cache_file = None
    if opts.cache:
        args = dict(
            dataset_kwargs=opts.dataset_kwargs,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            stats_kwargs=stats_kwargs
        )
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(flag, src=0)
            flag = (float(flag.cpu()) != 0)

        if flag:
            stats = FeatureStats.load(cache_file)
            if (capture_labels and stats.capture_labels) or (not capture_labels):
                return stats
            # otherwise fall through to re-compute so we can capture labels

    # Initialize stats  (labels may be requested)
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(
        capture_all      = stats_kwargs.get('capture_all', False),
        capture_mean_cov = stats_kwargs.get('capture_mean_cov', False),
        capture_labels   = capture_labels,      # <<–– this is new
        max_items        = num_items,
    )


    progress = opts.progress.sub(tag='dataset features', num_items=num_items,
                                 rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose
    )

    # iterate
    item_subset = [
        (i * opts.num_gpus + opts.rank) % num_items
        for i in range((num_items - 1)//opts.num_gpus + 1)
    ]
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=item_subset,
        batch_size=batch_size,
        **data_loader_kwargs
    )
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        feats = detector(images.to(opts.device), **detector_kwargs)
        
        stats.append_torch(feats, num_gpus=opts.num_gpus, rank=opts.rank)
        
        if capture_labels:
            # 1) pull labels out as a numpy array
            lbl = labels.cpu().numpy()
            # 2) if it’s one-hot / multi-dim, collapse to int
            if lbl.ndim > 1:
                lbl = lbl.argmax(axis=-1)
            # 3) now append the 1-D array of ints
            stats.append_labels(lbl)
            
        progress.update(stats.num_items)

    # Save to cache
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)


    return stats
#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(
        capture_all      = stats_kwargs.get('capture_all', False),
        capture_mean_cov = stats_kwargs.get('capture_mean_cov', False),
        capture_labels   = True,
        max_items        = stats_kwargs['max_items'],
    )

    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        batch_ids = []

        # build one big batch of images *and* record the discrete class IDs
        for _ in range(batch_size // batch_gen):
            # 1) sample integer class indices
            idxs = np.random.randint(len(dataset), size=batch_gen)
            batch_ids.append(idxs)

            # 2) fetch the corresponding conditioning vectors
            c_list = [dataset.get_label(i) for i in idxs]
            c_tensor = torch.from_numpy(np.stack(c_list)).pin_memory().to(opts.device)

            # 3) generate images
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            images.append(run_generator(z, c_tensor))

        # concatenate everything
        images = torch.cat(images, dim=0)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        # extract features
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

        # flatten and append the 1-D class indices
        all_ids = np.concatenate(batch_ids, axis=0)
        stats.append_labels(all_ids)

        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
