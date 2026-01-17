# metric_pr_class_cond.py
# ------------------------------------------------------------
#   Class–conditional Precision / Recall  (Kynkaanniemi et al.)
#   Original author: NVIDIA
#   Extension: per–class scores + macro average
# ------------------------------------------------------------
import torch
from . import metric_utils_pr

#----------------------------------------------------------------------------

def _compute_distances(row_features, col_features,
                       num_gpus, rank, col_batch_size):
    """Identical to the vanilla helper; kept verbatim."""
    assert 0 <= rank < num_gpus
    num_cols    = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(
        col_features, [0, 0, 0, -num_cols % num_batches]
    ).chunk(num_batches)

    dist_batches = []
    for col_batch in col_batches[rank::num_gpus]:
        
        rf = row_features.float()
        cf = col_batch .float()
        
        dist_batch = torch.cdist(
            rf.unsqueeze(0),
            cf.unsqueeze(0),
        )[0]  # (rows, cols_chunk)

        for src in range(num_gpus):
            d_b = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(d_b, src=src)
            dist_batches.append(d_b.cpu() if rank == 0 else None)

    return (
        torch.cat(dist_batches, dim=1)[:, :num_cols]
        if rank == 0 else None
    )

#----------------------------------------------------------------------------

def _pr_single_class(real_feat, gen_feat, opts,
                     nhood_size, row_bs, col_bs):
    """
    Compute precision & recall for a single class.
      real_feat : (N_real_cls, C)
      gen_feat  : (N_gen_cls , C)
    """
    results = {}
    for name, manifold, probes in (
        ('precision', real_feat, gen_feat),
        ('recall',    gen_feat, real_feat),
    ):
        # 1) compute kth‐nearest distances in the manifold
        kth = []
        for mb in manifold.split(row_bs):
            dist = _compute_distances(
                mb, manifold, opts.num_gpus, opts.rank, col_bs)
            # take (nhood_size+1)th smallest => neighborhood radius
            kth.append(
                dist.float()
                    .kthvalue(nhood_size + 1).values
                    .half()
            if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None

        # 2) for each probe, check if any manifold point lies within that radius
        pred = []
        for pb in probes.split(row_bs):
            dist = _compute_distances(
                pb, manifold, opts.num_gpus, opts.rank, col_bs)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)

        # 3) average over all probes
        results[name] = (
            torch.cat(pred).float().mean().item()
            if opts.rank == 0 else float('nan')
        )
    return results['precision'], results['recall']


def compute_pr_class_cond(opts,
                          max_real,
                          num_gen,
                          nhood_size     = 5,
                          row_batch_size = 10000,
                          col_batch_size = 10000):
    """
    Class‐conditional Precision & Recall.

    Returns:
        cls_prec   : dict[class_id → precision]
        cls_recall : dict[class_id → recall]
        macro_P    : float, mean precision over classes
        macro_R    : float, mean recall    over classes
    """
    detector_url = (
        'https://nvlabs-fi-cdn.nvidia.com/'
        'stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    )
    detector_kw = dict(return_features=True)

    # 1) extract *all* features + labels for real & generated
    real_stats = metric_utils_pr.compute_feature_stats_for_dataset(
        opts, detector_url, detector_kw,
        rel_lo=0, rel_hi=0,
        capture_all=True,
        capture_labels=True,     # <— capture labels
        max_items=max_real
    )
    gen_stats = metric_utils_pr.compute_feature_stats_for_generator(
        opts, detector_url, detector_kw,
        rel_lo=0, rel_hi=1,
        capture_all=True,
        capture_labels=True,     # <— capture labels
        max_items=num_gen
    )

    
    
    real_feat   = real_stats.get_all_torch().half().to(opts.device)  # (N_real, C)
    gen_feat    = gen_stats .get_all_torch().half().to(opts.device)  # (N_gen , C)
    real_labels = real_stats.labels.to(opts.device)                  # (N_real,)
    gen_labels  = gen_stats .labels.to(opts.device)                  # (N_gen ,)

    print('real_feat', real_feat.shape,
      'real_labels', real_labels.shape, real_labels.dtype)
    print('gen_feat ', gen_feat .shape,
      'gen_labels ', gen_labels.shape,  gen_labels.dtype)

     # ——————————————————————————————
    # Ensure labels are 1-D class indices, not one-hot vectors
    if real_labels.ndim > 1:
        real_labels = real_labels.argmax(dim=-1)
    if gen_labels.ndim > 1:
        gen_labels = gen_labels .argmax(dim=-1)
    # ——————————————————————————————
    
    # 2) iterate over each class present
    classes = torch.unique(torch.cat([real_labels, gen_labels])).tolist()
    cls_prec   = {}
    cls_recall = {}
    for cls in classes:
        r_mask = (real_labels == cls)
        g_mask = (gen_labels  == cls)
        # skip if either set has zero samples
        if not r_mask.any() or not g_mask.any():
            continue
        P, R = _pr_single_class(
            real_feat[r_mask], gen_feat[g_mask],
            opts, nhood_size, row_batch_size, col_batch_size
        )
        cls_prec[int(cls)]   = P
        cls_recall[int(cls)] = R

    # 3) macro‐averages
    if cls_prec:
        macro_P = sum(cls_prec.values())   / len(cls_prec)
        macro_R = sum(cls_recall.values()) / len(cls_recall)
    else:
        macro_P = 0.0
        macro_R = 0.0

    return cls_prec, cls_recall, macro_P, macro_R
