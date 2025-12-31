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
import torch.nn.functional as F
import torch.nn as nn


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
                 ijepa_warmup_kimg=500):
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

        # ------------ frozen I‑JEPA encoder ---------------------------
        self.enc, _ = build_ijepa_encoder(
            ijepa_ckpt,
            device=device,
            in_channels_override=ijepa_in_ch)
        self.enc.eval().requires_grad_(False)

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
    # helper: global (B, D=384) feature
    # ------------------------------------------------------------------
    def _feat(self, img):
        c = img.shape[1]
        if c < self.expect_c:
            img = img.repeat(1, self.expect_c // c, 1, 1)
        elif c > self.expect_c:
            img = img.mean(1, keepdim=True)
        return self.enc(img)  # pool patch tokens

    def run_G(self, z, c, e_ijepa, sem_ramp, sync):
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
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, e_ijepa, sem_ramp, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
            
            # if e_ijepa is not None:
            #     e_ijepa = self._feat(img).detach()
                        
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, e_ijepa=e_ijepa)
        return logits
    
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # neutral embedding  (all‑zeros)  used for phases that don't care
        zero_embed = torch.zeros(gen_z.size(0), 2048, device=self.device)
        # # ramp = (self.cur_kimg / self.warmup_kimg).clamp_(0.0, 1.0)
        # # ramp_tensor = (self.cur_kimg / self.warmup_kimg).clamp(0.0, 1.0)
        # # sem_ramp = float(ramp_tensor.item())
        # # lam = self.lambda_base * sem_ramp
       
        # # compute a 0→1 ramp that starts at 2 kimg and ends at self.warmup_kimg
        # ramp = ((self.cur_kimg - 2.0) / (self.warmup_kimg - 2.0)).clamp(0.0, 1.0)
        # sem_ramp = float(ramp.item())
       
        # # sem_ramp = float(ramp_tensor.item())
        # lam = self.lambda_base * sem_ramp  
        # # print(f"[DEBUG] cur_kimg={self.cur_kimg:.6f}kimg, λ={lam:.6f}")
        # training_stats.report("Loss/IJEPA_weight", lam)
        
        
        # ───────────────── ramp & weight ─────────────────────────────────────────
        ramp      = ((self.cur_kimg - 2.0) / (self.warmup_kimg - 2.0)).clamp(0.0, 1.0)
        sem_ramp  = float(ramp.item())
        lam       = self.lambda_base * sem_ramp
        training_stats.report("Loss/IJEPA_weight", lam)

        # ───────────────── semantic targets ─────────────────────────────────────
        target_f        = self._feat(real_img).detach()          # (B, 2048)
        batch_size_pl   = gen_z.shape[0] // self.pl_batch_shrink
        target_f_small  = target_f[:batch_size_pl] if do_Gpl else None
        

        # ────────────────────────── Gmain ───────────────────────────────────────
        
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                target_f = self._feat(real_img).detach()
                # scaled_target_f = target_f * sem_ramp

                # Generate a fake image conditioned on that embedding:
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c,
                    e_ijepa=target_f,  # semantic conditioning
                    sem_ramp=sem_ramp,
                    sync=(sync and not do_Gpl))

                # Ask the discriminator for its logit on that fake:
                gen_logits = self.run_D(
                    gen_img, gen_c,
                    e_ijepa=target_f,  # same embedding
                    sem_ramp=sem_ramp,
                    sync=False
                )

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gadv = torch.nn.functional.softplus(-gen_logits)
                fake_f = self._feat(gen_img)
                loss_Gfm = 1.0 - F.cosine_similarity(fake_f, target_f, dim=1)
                loss_Gmain = loss_Gadv + lam * loss_Gfm

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
        
        # ────────────────────────── Gpl ─────────────────────────────────────────
        
        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                # batch_size = gen_z.shape[0] // self.pl_batch_shrink
                # gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], e_ijepa=zero_embed[:batch_size],
                #                              sem_ramp=0.0, sync=sync)
                
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size_pl], gen_c[:batch_size_pl],
                    e_ijepa=target_f_small,
                    sem_ramp=sem_ramp,
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
                # gen_img, _gen_ws = self.run_G(gen_z, gen_c, e_ijepa=zero_embed, sem_ramp=0.0, sync=False)
                
                gen_img, _ = self.run_G(
                    gen_z, gen_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
                    sync=False,
                )
                
                gen_logits = self.run_D(
                    gen_img, gen_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
                    sync=False,
                )
                

                # gen_logits = self.run_D(gen_img, gen_c, e_ijepa=zero_embed, sem_ramp=0.0, sync=False)  # Gets synced by loss_Dreal.
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
                # real_logits = self.run_D(real_img_tmp, real_c, e_ijepa=zero_embed, sem_ramp=0.0, sync=sync)
                real_logits  = self.run_D(
                    real_img_tmp, real_c,
                    e_ijepa=target_f,
                    sem_ramp=sem_ramp,
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
