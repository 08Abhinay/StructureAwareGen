# # training/loss_ijepa.py
# import torch, torch.nn.functional as F
# from torchvision.transforms.functional import resize
# from training.loss import StyleGAN2Loss           # original base loss
# from torch_utils import training_stats
# from training.ijepa_encoder import build_ijepa_encoder   # helper

# class StyleGAN2IJEPALoss(StyleGAN2Loss):
#     def __init__(self, device,
#                  ijepa_ckpt,
#                  lambda_ijepa = 1.0,
#                  ijepa_img    = 224,
#                  ijepa_in_ch  = 1,
#                  **sg_kwargs):

#         super().__init__(device=device, **sg_kwargs)

#         self.enc, self.meta = build_ijepa_encoder(
#             ijepa_ckpt, device=device, in_channels_override=ijepa_in_ch, img_size=ijepa_img)
#         self.enc.eval().requires_grad_(False)
#         self.lambda_ijepa = lambda_ijepa
#         self.resize_to   = ijepa_img
#         self.expect_c    = ijepa_in_ch

#     def _feat(self, img):
#         c = img.shape[1]
#         if c < self.expect_c:  # up‑repeat (1→3)
#             img = img.repeat(1, self.expect_c // c, 1, 1)
#         elif c > self.expect_c:  # down‑mix (3→1)
#             img = img.mean(1, keepdim=True)  # or img[:, :1] …
#         # now img.shape[1] == self.expect_c
#         # img = resize(img, [self.resize_to, self.resize_to], antialias=True)
#         return self.enc(img).mean(1)

#     # pooled token → (B, D)

#     # ---- override only G-main gradient ---------------------------------
#     def accumulate_gradients(self, phase, real_img, real_c,
#                               gen_z, gen_c, sync, gain):

#         # run original SG2-ADA losses first
#         super().accumulate_gradients(phase, real_img, real_c,
#                                      gen_z, gen_c, sync, gain)

#         if phase != "Gmain" or self.lambda_ijepa == 0:
#             return

#         with torch.autograd.profiler.record_function('Gijepa_forward'):
#             real_f = self._feat(real_img).detach()
#             gen_img, _ = self.run_G(gen_z, gen_c, sync=sync)
#             fake_f = self._feat(gen_img)
#             f_loss = F.mse_loss(fake_f, real_f)
#             training_stats.report('Loss/IJEPA_feat', f_loss)

#         with torch.autograd.profiler.record_function('Gijepa_backward'):
#             (f_loss * self.lambda_ijepa * gain).backward()
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from training.loss import StyleGAN2Loss        # base class
from torch_utils import misc, training_stats
from training.ijepa_encoder import build_ijepa_encoder


class StyleGAN2IJEPALoss(StyleGAN2Loss):
    """
    Extends StyleGAN2Loss by
    • passing an external semantic vector (`extra`) into G.mapping
    • (optionally) adding an MSE feature‑matching term in I‑JEPA space
    """

    def __init__(
        self,
        device,
        ijepa_ckpt: str,            # path / URL to ViT I‑JEPA checkpoint
        lambda_ijepa: float = 0.0,  # 0 = disable feature loss; 0.05–0.1 is mild
        ijepa_img: int = 256,       # resize side length fed to encoder
        ijepa_in_ch: int = 3,       # input channels expected by encoder
        **sg_kwargs,                # all the usual StyleGAN2 kwargs
    ):
        super().__init__(device=device, **sg_kwargs)

        # ----------------------------------------------------------------
        # Freeze the pre‑trained I‑JEPA encoder
        # ----------------------------------------------------------------
        self.enc, self.meta = build_ijepa_encoder(
            ijepa_ckpt,
            device=device,
            in_channels_override=ijepa_in_ch,
        )
        self.enc.eval().requires_grad_(False)

        self.lambda_ijepa = lambda_ijepa
        self.expect_c = ijepa_in_ch
        self.resize_to = ijepa_img

    # --------------------------------------------------------------------
    # Helper: get a single (B,D) global feature per image
    # --------------------------------------------------------------------
    def _feat(self, img):
        c = img.shape[1]
        if c < self.expect_c:             # 1→3 (repeat)
            img = img.repeat(1, self.expect_c // c, 1, 1)
        elif c > self.expect_c:           # 3→1 (mean)
            img = img.mean(1, keepdim=True)

        # img = resize(img, [self.resize_to, self.resize_to], antialias=True)
        return self.enc(img).mean(1)      # pool over patch tokens   (B,D)

    # --------------------------------------------------------------------
    # Override run_G so we can hand an `extra` vector to mapping()
    # --------------------------------------------------------------------
    def run_G(self, z, c, extra=None, sync=True):
        """
        Same as parent, but forwards `extra` to G.mapping.
        If `extra` is None the behaviour is unchanged.
        """
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c, extra=extra)

            # style‑mixing (only when extra=None, keeps logic simple)
            if (self.style_mixing_prob > 0):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.G_mapping(
                    torch.randn_like(z), c, skip_w_avg_update=True
                )[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    # --------------------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------------------
    def accumulate_gradients(
        self,
        phase,
        real_img,
        real_c,
        gen_z,
        gen_c,
        sync,
        gain,
    ):
        # 1.  run ordinary SG2‑ADA losses first (no semantic vector)
        super().accumulate_gradients(
            phase, real_img, real_c, gen_z, gen_c, sync, gain
        )

        # 2.  optional feature‑matching loss (Gmain only)
        if (phase != "Gmain") or (self.lambda_ijepa == 0):
            return

        with torch.autograd.profiler.record_function("Gijepa_forward"):
            ijepa_f = self._feat(real_img).detach()                 # (B,D)
            gen_img, _ = self.run_G(gen_z, gen_c, extra=ijepa_f, sync=True)
            fake_f = self._feat(gen_img)

            # cosine similarity is often smoother than raw MSE
            f_loss = 1.0 - F.cosine_similarity(fake_f, ijepa_f, dim=1).mean()
            training_stats.report("Loss/IJEPA_feat", f_loss)

        with torch.autograd.profiler.record_function("Gijepa_backward"):
            (f_loss * self.lambda_ijepa * gain).backward()