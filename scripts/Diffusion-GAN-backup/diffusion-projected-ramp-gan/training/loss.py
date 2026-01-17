# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from training.ijepa_encoder import build_ijepa_encoder 
import torch.nn as nn                                      

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0,
                 ijepa_ckpt=None, lambda_ijepa=0.0,
                 ijepa_img=256, ijepa_in_ch=3,    # NEW args – keep same names as before
                 ijepa_warmup_kimg=500, **kwargs):
        
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        
        # ── I-JEPA encoder (frozen) ──────────────────────────────────────
        self.enc, _ = build_ijepa_encoder(
            ijepa_ckpt,
            device=device,
            in_channels_override=ijepa_in_ch)
        self.enc.eval().requires_grad_(False)
        
        self.lambda_base = float(lambda_ijepa)
        self.warmup_kimg = ijepa_warmup_kimg
        self.cur_kimg    = torch.zeros([], device=device)

        self.expect_c = ijepa_in_ch
        self.resize_to = ijepa_img
        
    # --- helper: I-JEPA global feature (B, 2048) ------------------------
    def _feat(self, img):
        c = img.shape[1]
        if c < self.expect_c:
            img = img.repeat(1, self.expect_c // c, 1, 1)
        elif c > self.expect_c:
            img = img.mean(1, keepdim=True)
        return self.enc(img)         # returns pooled 2048-d vector
        
        
    def run_G(self, z, c,  e_ijepa=None, sem_ramp=None, update_emas=False):
        ws = self.G.mapping(z, c, e_ijepa=e_ijepa, sem_ramp=sem_ramp, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0,  e_ijepa=None, sem_ramp=None, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c, e_ijepa=e_ijepa, sem_ramp=sem_ramp)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        zero_embed = torch.zeros(gen_z.size(0), 2048, device=self.device)
        
        # ───────────────── ramp & weight ─────────────────────────────────────────
        ramp      = ((self.cur_kimg - 2.0) / (self.warmup_kimg - 2.0)).clamp(0.0, 1.0)
        sem_ramp  = float(ramp.item())
        lam       = self.lambda_base * sem_ramp
        training_stats.report("Loss/IJEPA_weight", lam)

        # ───────────────── semantic targets ─────────────────────────────────────
        target_f        = self._feat(real_img).detach()          # (B, 2048)
        
        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c, e_ijepa=target_f,  # semantic conditioning
                                    sem_ramp=sem_ramp)
                gen_logits = self.run_D(gen_img, gen_c, e_ijepa=target_f,  # same embedding
                                        sem_ramp=sem_ramp, 
                                        blur_sigma=blur_sigma)
                # Adversarial term (scalar)
                loss_Gadv = F.softplus(-gen_logits).mean()

                # Semantic feature‐matching term (scalar)
                fake_f    = self._feat(gen_img)
                loss_Gfm  = (1.0 - F.cosine_similarity(fake_f, target_f, dim=1)).mean()

                # Combined loss
                loss_Gmain = loss_Gadv + lam * loss_Gfm

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake',  gen_logits.sign())
                training_stats.report('Loss/G/adv',       loss_Gadv)
                training_stats.report('Loss/G/fm',        loss_Gfm)
                training_stats.report('Loss/G/loss',      loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                
                gen_img = self.run_G(gen_z, gen_c, e_ijepa=target_f,  # semantic conditioning
                                    sem_ramp=sem_ramp, update_emas=True)
                
                gen_logits = self.run_D(gen_img, gen_c, e_ijepa=target_f,  # same embedding
                                        sem_ramp=sem_ramp, blur_sigma=blur_sigma)
                
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, e_ijepa=target_f,  # same embedding
                                        sem_ramp=sem_ramp, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()

        if phase in ("Dmain", "Dboth"):  # exactly once per iter
            self.cur_kimg += gen_z.shape[0] / 1000.0