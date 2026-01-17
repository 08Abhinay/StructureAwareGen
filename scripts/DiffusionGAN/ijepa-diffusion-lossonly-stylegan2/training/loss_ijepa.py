# training/loss_ijepa.py
import torch, torch.nn.functional as F
from torchvision.transforms.functional import resize
from training.loss import StyleGAN2Loss           # original base loss
from torch_utils import training_stats
from training.ijepa_encoder import build_ijepa_encoder   # helper

class StyleGAN2IJEPALoss(StyleGAN2Loss):
    def __init__(self, device,
                 ijepa_ckpt,
                 lambda_ijepa = 1.0,
                 ijepa_img    = 224,
                 ijepa_in_ch  = 1,
                 **sg_kwargs):

        super().__init__(device=device, **sg_kwargs)

        self.enc, self.meta = build_ijepa_encoder(
            ijepa_ckpt, device=device, in_channels_override=ijepa_in_ch, img_size=ijepa_img)
        self.enc.eval().requires_grad_(False)
        self.lambda_ijepa = lambda_ijepa
        self.resize_to   = ijepa_img
        self.expect_c    = ijepa_in_ch

    def _feat(self, img):
        c = img.shape[1]
        if c < self.expect_c:  # up‑repeat (1→3)
            img = img.repeat(1, self.expect_c // c, 1, 1)
        elif c > self.expect_c:  # down‑mix (3→1)
            img = img.mean(1, keepdim=True)  # or img[:, :1] …
        # now img.shape[1] == self.expect_c
        # img = resize(img, [self.resize_to, self.resize_to], antialias=True)
        return self.enc(img).mean(1)

    # pooled token → (B, D)

    # ---- override only G-main gradient ---------------------------------
    def accumulate_gradients(self, phase, real_img, real_c,
                              gen_z, gen_c, sync, gain):

        # run original SG2-ADA losses first
        super().accumulate_gradients(phase, real_img, real_c,
                                     gen_z, gen_c, sync, gain)

        if phase != "Gmain" or self.lambda_ijepa == 0:
            return

        with torch.autograd.profiler.record_function('Gijepa_forward'):
            real_f = self._feat(real_img).detach()
            gen_img, _ = self.run_G(gen_z, gen_c, sync=sync)
            fake_f = self._feat(gen_img)
            f_loss = F.mse_loss(fake_f, real_f)
            training_stats.report('Loss/IJEPA_feat', f_loss)

        with torch.autograd.profiler.record_function('Gijepa_backward'):
            (f_loss * self.lambda_ijepa * gain).backward()
