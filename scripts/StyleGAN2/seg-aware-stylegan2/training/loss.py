# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.ijepa_encoder import build_ijepa_encoder
from training.sam_extractor import SAMExtractor, pad_embeddings_batch
import torch.nn.functional as F
import torch.nn as nn
import random


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D,
                 augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 ijepa_ckpt=None, lambda_ijepa=0.0, ijepa_img=256, ijepa_in_ch=3,
                 ijepa_warmup_kimg=500,
                 sam_enabled=False, sam_prob=0.25, sam_checkpoint=None,
                 sam_cache_dir=None, sam_model_type='vit_b', sam_max_masks=250,
                 sam_emb_logging=False,
                 # AMG parameters (matching precompute_sam_embeddings.py)
                 sam_points_per_side=32, sam_pred_iou_thresh=0.82,
                 sam_stability_score_thresh=0.85, sam_box_nms_thresh=0.70,
                 sam_crop_n_layers=0, sam_dedup_iou_thresh=0.95,
                 sam_min_mask_region_area=100,
                 lambda_seg_align=0.1, lambda_seg_diversity=0.05,
                 seg_align_tau=0.07,
                 origin_map=None,
                 rank=0, num_gpus=1):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        
        # Alignment and diversity loss weights
        self.lambda_seg_align = lambda_seg_align
        self.lambda_seg_diversity = lambda_seg_diversity
        self.seg_align_tau = seg_align_tau
        
        # ------------ SAM extractor (optional) -----------------------
        self.sam_enabled = sam_enabled
        self.sam_prob = sam_prob
        self.sam_emb_logging = sam_emb_logging
        self.sam_extractor = None
        
        # SAM embedding logging counters
        self._sam_log_pre_extracted = 0   # pre-computed from AlignedSegDataset reused
        self._sam_log_cache_hits = 0      # loaded by SAMExtractor from unified cache
        self._sam_log_on_the_fly = 0      # just extracted on-the-fly
        self._sam_log_fallback = 0        # missing pre-computed → extracted on-the-fly per-sample
        self._sam_log_dropout = 0         # dropped (stochastic conditioning)
        self._sam_log_total = 0           # total batches with SAM decision
        
        if sam_enabled:
            if sam_checkpoint is None or sam_cache_dir is None:
                raise ValueError("sam_checkpoint and sam_cache_dir required when sam_enabled=True")
            
            self.sam_extractor = SAMExtractor(
                sam_checkpoint=sam_checkpoint,
                cache_dir=sam_cache_dir,
                device=device,
                model_type=sam_model_type,
                max_masks=sam_max_masks,
                rank=rank,
                world_size=num_gpus,
                origin_map=origin_map,
                # AMG parameters
                points_per_side=sam_points_per_side,
                pred_iou_thresh=sam_pred_iou_thresh,
                stability_score_thresh=sam_stability_score_thresh,
                box_nms_thresh=sam_box_nms_thresh,
                crop_n_layers=sam_crop_n_layers,
                dedup_iou_thresh=sam_dedup_iou_thresh,
                min_mask_region_area=sam_min_mask_region_area,
            )
            print(f"[Loss] SAM extractor enabled with prob={sam_prob}, logging={sam_emb_logging}")
        
        # Separate RNG for SAM decisions (independent of training seed)
        self.sam_rng = random.Random(42)

        # ------------ frozen I‑JEPA encoder ---------------------------
        # Build encoder FIRST so we know the output dimension for downstream layers
        self.enc, self.enc_meta = build_ijepa_encoder(
            ijepa_ckpt,
            device=device,
            in_channels_override=ijepa_in_ch)
        self.enc.eval().requires_grad_(False)
        self.ijepa_out_dim = self.enc_meta['out_dim']  # e.g. 1280 for ViT-H
        print(f"[Loss] I-JEPA encoder output dim = {self.ijepa_out_dim}")

        # SAM -> I-JEPA projection MLP for alignment loss
        if sam_enabled and lambda_seg_align > 0:
            self.sam_proj_mlp = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, self.ijepa_out_dim)
            ).to(device)
            print(f"[Loss] SAM projection MLP created: 256→512→{self.ijepa_out_dim} (weight={lambda_seg_align})")
        else:
            self.sam_proj_mlp = None

        # self.lambda_ijepa = float(lambda_ijepa)
        # self.expect_c = ijepa_in_ch
        # self.resize_to = ijepa_img

        self.lambda_base = float(lambda_ijepa)
        self.warmup_kimg = ijepa_warmup_kimg
        self.cur_kimg = torch.zeros([], device=device)

        self.expect_c = ijepa_in_ch
        self.resize_to = ijepa_img

        # ── unwrap if it’s a DDP container ────────────────────────────────────
        core = self.G_mapping.module if isinstance(
            self.G_mapping, nn.parallel.DistributedDataParallel) else self.G_mapping

        # Verify that mapping is the fusion version.
        if not hasattr(core, "proj_ijepa"):
            raise ValueError("G.mapping must be an IJEPAFusionMapping instance.")
    
    # ------------------------------------------------------------------
    # Diversity & Alignment Loss Methods
    # ------------------------------------------------------------------
    def compute_seg_diversity_loss(self, seg_tokens, seg_pad_mask=None, eps=1e-6):
        """
        Diversity loss: -log det(C + εI) on SAM segment embeddings.
        Matches the log-determinant form in theory.tex (Lemma 1).
        Penalises collapsed / low-rank covariance spectra.

        Tokens are L2-normalised first so eigenvalues live in [0, 1] and
        the log-determinant is divided by the embedding dimension C to make
        the loss scale-invariant (otherwise it grows ∝ C ≈ 256).
        
        Args:
            seg_tokens: [B, N, 256] SAM segment embeddings
            seg_pad_mask: [B, N] boolean mask (True = padded)
            eps: Regularization constant for numerical stability
            
        Returns:
            Scalar diversity loss (higher = more collapsed)
        """
        B, N, C = seg_tokens.shape
        
        # Remove padding
        if seg_pad_mask is not None:
            mask = ~seg_pad_mask  # [B, N]
        else:
            mask = torch.ones(B, N, dtype=torch.bool, device=seg_tokens.device)
        
        losses = []
        for i in range(B):
            valid_tokens = seg_tokens[i, mask[i]]  # [N_valid, C]
            if valid_tokens.shape[0] < 2:
                continue
            
            # L2-normalise so covariance eigenvalues are bounded in [0, 1]
            valid_tokens = F.normalize(valid_tokens, dim=1)
            
            # Compute sample covariance matrix
            mean = valid_tokens.mean(dim=0, keepdim=True)  # [1, C]
            centered = valid_tokens - mean  # [N_valid, C]
            cov = (centered.T @ centered) / valid_tokens.shape[0]  # [C, C]
            
            # Regularise and compute log determinant
            cov_reg = cov + eps * torch.eye(C, device=cov.device)
            
            # slogdet for numerical stability
            sign, logdet = torch.slogdet(cov_reg)
            if sign > 0:  # Only use if positive definite
                # Normalise by dimension so loss doesn't scale with C
                losses.append(-logdet / C)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=seg_tokens.device)
    
    def compute_seg_alignment_loss(self, ijepa_features, seg_tokens, seg_pad_mask=None):
        """
        Contrastive alignment loss (InfoNCE) between I-JEPA global features
        and SAM segment embeddings.  Uses in-batch negatives with temperature
        self.seg_align_tau, matching theory.tex §1.4.
        
        Satisfies assumption (iii) of Lemmas 2–3: constant embeddings across
        all images cannot minimise this loss because the cross-image negatives
        make the softmax uniform.
        
        Args:
            ijepa_features: [B, ijepa_out_dim] I-JEPA global features
            seg_tokens: [B, N, 256] SAM segment embeddings
            seg_pad_mask: [B, N] boolean mask (True = padded)
            
        Returns:
            Scalar alignment loss
        """
        if self.sam_proj_mlp is None:
            return torch.tensor(0.0, device=ijepa_features.device)
        
        B, N, C = seg_tokens.shape
        
        # Pool SAM segments (masked average over valid tokens)
        if seg_pad_mask is not None:
            valid_mask = ~seg_pad_mask  # [B, N]
            mask_float = valid_mask.float().unsqueeze(2)  # [B, N, 1]
            seg_pooled = (seg_tokens * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)  # [B, 256]
        else:
            seg_pooled = seg_tokens.mean(dim=1)  # [B, 256]
        
        # Project SAM pooled features to I-JEPA space
        seg_projected = self.sam_proj_mlp(seg_pooled)  # [B, ijepa_out_dim]
        
        # Fallback for B=1 (e.g. final partial batch): simple cosine
        if B < 2:
            return 1.0 - F.cosine_similarity(ijepa_features, seg_projected, dim=1).mean()
        
        # L2 normalise both sides
        g_norm = F.normalize(ijepa_features, dim=1)   # [B, D]
        s_norm = F.normalize(seg_projected, dim=1)     # [B, D]
        
        # Similarity matrix: sim[i,j] = cos(g_i, s_j) / tau
        sim_matrix = torch.mm(g_norm, s_norm.t()) / self.seg_align_tau  # [B, B]
        
        # InfoNCE: positive pairs are on the diagonal
        labels = torch.arange(B, device=sim_matrix.device)
        alignment_loss = F.cross_entropy(sim_matrix, labels)
        
        return alignment_loss

    # ------------------------------------------------------------------
    # helper: global (B, D=384) feature
    # ------------------------------------------------------------------
    def _feat(self, img):
        c = img.shape[1]
        if c < self.expect_c:
            img = img.repeat(1, self.expect_c // c, 1, 1)
        elif c > self.expect_c:
            img = img.mean(1, keepdim=True)
        return self.enc(img)  # pool patch tokens
    
    # ------------------------------------------------------------------
    # helper: SAM embeddings extraction or loading from cache
    # ------------------------------------------------------------------
    def _get_sam_embeddings(self, real_img, image_paths):
        """
        Extract or load SAM embeddings for a batch of images.
        
        Args:
            real_img: (B, C, H, W) tensor in [-1, 1] range
            image_paths: List of B image file paths
            
        Returns:
            Tuple of (seg_tokens, seg_pad_mask) or (None, None) if no paths
        """
        if image_paths is None or len(image_paths) == 0:
            return None, None
        
        if self.sam_extractor is None:
            return None, None
        
        # Extract or load from cache
        embeddings_list = self.sam_extractor.extract_or_load(image_paths, real_img)
        
        # Pad to batch format
        seg_tokens, seg_pad_mask = pad_embeddings_batch(embeddings_list, device=self.device)
        
        return seg_tokens, seg_pad_mask

    def run_G(self, z, c, e_ijepa, sem_ramp, sync, seg_tokens=None, seg_pad_mask=None, seg_ramp=1.0):
        # Mapping (always returns ws, film)
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c, e_ijepa=e_ijepa, sem_ramp=sem_ramp)

            # style‑mixing (optional)
            if self.style_mixing_prob > 0:
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]))
                mix_ws = self.G_mapping(
                    torch.randn_like(z), c, e_ijepa=e_ijepa, sem_ramp=sem_ramp, skip_w_avg_update=True)
                ws[:, cutoff:] = mix_ws[:, cutoff:]

        # Synthesis
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, seg_tokens=seg_tokens, seg_pad_mask=seg_pad_mask, seg_ramp=seg_ramp)
        return img, ws

    def run_D(self, img, c, e_ijepa, sem_ramp, sync, seg_tokens=None, seg_pad_mask=None, seg_ramp=0.0):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
            
            # if e_ijepa is not None:
            #     e_ijepa = self._feat(img).detach()
                        
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, e_ijepa=e_ijepa, seg_tokens=seg_tokens, seg_pad_mask=seg_pad_mask,
                            sem_ramp=sem_ramp, seg_ramp=seg_ramp)
        return logits
    
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain,
                             real_seg_tokens=None, real_seg_pad_mask=None, image_paths=None,
                             real_global_vec=None):

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        
        # ───────────────── Stochastic SAM Conditioning ─────────────────────────
        # sam_prob controls whether SAM tokens are USED this batch (dropout-like).
        # If use_sam=True  → use tokens (pre-computed if available, else extract)
        # If use_sam=False → dropout: set tokens to None (unconditional generation)
        # I-JEPA global guidance is ALWAYS active regardless of this decision.
        use_sam = False
        if self.sam_enabled and image_paths is not None:
            use_sam = self.sam_rng.random() < self.sam_prob
            self._sam_log_total += 1

            if use_sam:
                has_precomputed = (real_seg_tokens is not None and real_seg_pad_mask is not None)

                if has_precomputed:
                    # ── Per-sample hybrid: reuse valid pre-computed, extract missing ──
                    B = real_seg_pad_mask.shape[0]
                    missing_idx = [i for i in range(B) if real_seg_pad_mask[i].all()]
                    valid_count = B - len(missing_idx)

                    if valid_count > 0:
                        self._sam_log_pre_extracted += valid_count

                    # On-the-fly extraction ONLY for samples without pre-computed SAM
                    if missing_idx and self.sam_extractor is not None and image_paths is not None:
                        missing_paths = [image_paths[i] for i in missing_idx]
                        missing_imgs  = real_img[missing_idx] if real_img is not None else None
                        embeddings_list = self.sam_extractor.extract_or_load(missing_paths, missing_imgs)

                        max_seg = real_seg_tokens.shape[1]  # AlignedSegDataset pad dim (e.g. 250)
                        for j, bi in enumerate(missing_idx):
                            emb = embeddings_list[j]['emb']  # (N, 256) numpy
                            n = min(emb.shape[0], max_seg)
                            if n > 0:
                                real_seg_tokens[bi, :n] = torch.from_numpy(
                                    emb[:n].astype(np.float32)).to(real_seg_tokens.device)
                                real_seg_pad_mask[bi, :n] = False  # mark as valid
                        self._sam_log_fallback += len(missing_idx)

                else:
                    # No pre-computed tokens at all → full batch extraction
                    sam_seg_tokens, sam_seg_pad_mask = self._get_sam_embeddings(real_img, image_paths)
                    if sam_seg_tokens is not None:
                        real_seg_tokens  = sam_seg_tokens
                        real_seg_pad_mask = sam_seg_pad_mask
                        self._sam_log_fallback += real_seg_tokens.shape[0]
            else:
                # Stochastic dropout: don't use SAM this batch
                real_seg_tokens = None
                real_seg_pad_mask = None
                self._sam_log_dropout += 1

            # Report logging stats (when enabled)
            if self.sam_emb_logging and self._sam_log_total > 0:
                training_stats.report('SAM/pre_extracted_reused', self._sam_log_pre_extracted)
                training_stats.report('SAM/cache_hits', self._sam_log_cache_hits)
                training_stats.report('SAM/on_the_fly_extractions', self._sam_log_on_the_fly)
                training_stats.report('SAM/fallback_extractions', self._sam_log_fallback)
                training_stats.report('SAM/dropout_batches', self._sam_log_dropout)
                total_used = self._sam_log_pre_extracted + self._sam_log_cache_hits + self._sam_log_on_the_fly + self._sam_log_fallback
                hit_pct = (self._sam_log_pre_extracted + self._sam_log_cache_hits) / max(1, total_used) * 100
                training_stats.report('SAM/cache_hit_rate_pct', hit_pct)
                if self.sam_extractor is not None:
                    training_stats.report('SAM/lock_wait_time_s', self.sam_extractor.lock_wait_time)

        # ───────────────── ramp & weight ─────────────────────────────────────────
        ramp      = ((self.cur_kimg - 2.0) / (self.warmup_kimg - 2.0)).clamp(0.0, 1.0)
        sem_ramp  = float(ramp.item())
        seg_ramp  = sem_ramp if real_seg_tokens is not None else 0.0
        lam       = self.lambda_base * sem_ramp
        training_stats.report("Loss/IJEPA_weight", lam)

        # ───────────────── semantic targets ─────────────────────────────────────
        # Use pre-computed I-JEPA embedding if available, else run live encoder
        if real_global_vec is not None:
            target_f = real_global_vec.detach()                   # (B, ijepa_out_dim)
        else:
            target_f = self._feat(real_img).detach()              # (B, ijepa_out_dim)
        batch_size_pl   = gen_z.shape[0] // self.pl_batch_shrink
        target_f_small  = target_f[:batch_size_pl] if do_Gpl else None
        seg_tokens_pl   = real_seg_tokens[:batch_size_pl] if (do_Gpl and real_seg_tokens is not None) else None
        seg_pad_mask_pl = real_seg_pad_mask[:batch_size_pl] if (do_Gpl and real_seg_pad_mask is not None) else None
        

        # ────────────────────────── Gmain ───────────────────────────────────────
        
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # target_f already computed above (pre-computed or live encoder)

                # Generate a fake image conditioned on that embedding:
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c,
                    e_ijepa=target_f,  # semantic conditioning
                    sem_ramp=sem_ramp,
                    seg_tokens=real_seg_tokens,
                    seg_pad_mask=real_seg_pad_mask,
                    seg_ramp=seg_ramp,
                    sync=(sync and not do_Gpl))

                # Ask the discriminator for its logit on that fake:
                gen_logits = self.run_D(
                    gen_img, gen_c,
                    e_ijepa=target_f,  # same embedding
                    sem_ramp=sem_ramp,
                    seg_tokens=real_seg_tokens,
                    seg_pad_mask=real_seg_pad_mask,
                    seg_ramp=seg_ramp,
                    sync=False
                )

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gadv = torch.nn.functional.softplus(-gen_logits)
                fake_f = self._feat(gen_img)
                loss_Gfm = 1.0 - F.cosine_similarity(fake_f, target_f, dim=1)
                loss_Gmain = loss_Gadv + lam * loss_Gfm
                
                # Add alignment and diversity losses (if SAM is enabled)
                if real_seg_tokens is not None:
                    # Diversity loss (monitor real embeddings for collapse)
                    if self.lambda_seg_diversity > 0:
                        loss_diversity = self.compute_seg_diversity_loss(
                            real_seg_tokens, real_seg_pad_mask
                        )
                        training_stats.report('Loss/G/diversity', loss_diversity)
                        loss_Gmain = loss_Gmain + self.lambda_seg_diversity * loss_diversity * sem_ramp
                    
                    # Alignment loss (semantic coherence between I-JEPA and SAM)
                    if self.lambda_seg_align > 0:
                        loss_alignment = self.compute_seg_alignment_loss(
                            target_f, real_seg_tokens, real_seg_pad_mask
                        )
                        training_stats.report('Loss/G/alignment', loss_alignment)
                        loss_Gmain = loss_Gmain + self.lambda_seg_align * loss_alignment * sem_ramp

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
        
        # ────────────────────────── Gpl ─────────────────────────────────────────
        
        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                # Disable segmentation conditioning during Gpl to avoid double-gradient error
                # with efficient attention kernels (create_graph=True incompatible)
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size_pl], gen_c[:batch_size_pl],
                    e_ijepa=target_f_small,
                    sem_ramp=sem_ramp,
                    seg_tokens=None,
                    seg_pad_mask=None,
                    seg_ramp=0.0,
                    sync=sync,
                )
                
                
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()  # [batch] we get path lengths mean for each sample

                # self.pl_mean is already calculated for the previous batch
                # so when we do pl_lengths.mean() -> this collapses into a single scalar. So a single scalar for the batch
                # lerp is a formulae
                # pl_mean_new = (1 - pl_decay) * pl_mean_old + pl_decay * pl_lengths.mean() [pl_lengths.mean() gives the batch mean]
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
                
        # ────────────────────────── Dmain (fake) ────────────────────────────────
        
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _ = self.run_G(
                    gen_z, gen_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
                    seg_tokens=real_seg_tokens,
                    seg_pad_mask=real_seg_pad_mask,
                    seg_ramp=seg_ramp,
                    sync=False,
                )
                
                gen_logits = self.run_D(
                    gen_img, gen_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
                    seg_tokens=real_seg_tokens,
                    seg_pad_mask=real_seg_pad_mask,
                    seg_ramp=seg_ramp,
                    sync=False,
                )

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        
        # ────────────────── Dreal & optional R1 regularization ──────────────────
        
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits  = self.run_D(
                    real_img_tmp, real_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
                    seg_tokens=real_seg_tokens,
                    seg_pad_mask=real_seg_pad_mask,
                    seg_ramp=seg_ramp,
                    sync=sync,
                )
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # ----------------------------------------------------------------------------
        if phase in ("Dmain", "Dboth"):  # exactly once per iter
            self.cur_kimg += gen_z.shape[0] / 1000.0
