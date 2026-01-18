import numpy as np
import torch
import torch.nn as nn

from seg_rdm.diffusion.schedules import make_beta_schedule
from seg_rdm.diffusion.utils import extract_into_tensor, noise_like
from seg_rdm.models.ema import LitEma
from seg_rdm.utils.misc import count_params, default, exists, instantiate_from_config


class DDPM(nn.Module):
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=None,
        load_only_unet=False,
        use_ema=True,
        first_stage_key="rep",
        image_size=1,
        channels=256,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        learn_logvar=False,
        logvar_init=0.0,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        if ignore_keys is None:
            ignore_keys = []
        assert parameterization in ["eps", "x0"], "parameterization must be 'eps' or 'x0'"
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas must be defined for each timestep"

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            "posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            alphas_cumprod_t = torch.tensor(alphas_cumprod, dtype=torch.float32)
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod_t) / (2.0 - alphas_cumprod_t)
        else:
            raise NotImplementedError("parameterization not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)

    def ema_scope(self):
        if not self.use_ema:
            return _NullContextManager()
        return _EmaContext(self)

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        if only_model:
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if missing:
            print(f"Missing Keys: {missing}")
        if unexpected:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def apply_model(self, x_noisy, t, cond=None):
        if self.model.conditioning_key is None or cond is None:
            return self.model(x_noisy, t)
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        return self.model(x_noisy, t, **cond)

    def p_mean_variance(self, x, t, clip_denoised=True, cond=None):
        model_out = self.apply_model(x, t, cond=cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError("parameterization not supported")
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, cond=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False, cond=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
                cond=cond,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False, cond=None):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            return_intermediates=return_intermediates,
            cond=cond,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")
        return loss

    def p_losses(self, x_start, t, noise=None, cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, cond=cond)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not supported")

        loss = self.get_loss(model_out, target, mean=False)
        loss = loss.mean(dim=list(range(1, loss.ndim)))

        log_prefix = "train" if self.training else "val"
        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)


class RDM(DDPM):
    def __init__(
        self,
        unet_config,
        input_scale=1.0,
        num_timesteps_cond=None,
        cond_stage_key=None,
        cond_stage_trainable=False,
        cond_stage_config=None,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        class_cond=False,
        **kwargs,
    ):
        _ = (num_timesteps_cond, cond_stage_key, cond_stage_trainable, cond_stage_config, concat_mode, cond_stage_forward, class_cond)
        if conditioning_key is None:
            conditioning_key = None
        super().__init__(unet_config=unet_config, conditioning_key=conditioning_key, **kwargs)
        self.input_scale = float(input_scale)

    def prepare_input(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected input shape [B, C] or [B, C, 1, 1], got {x.shape}.")
        return x * self.input_scale

    def forward(self, x, cond=None, *args, **kwargs):
        x = self.prepare_input(x)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        loss, loss_dict = self.p_losses(x, t, cond=cond, *args, **kwargs)
        if self.use_ema:
            self.model_ema(self.model)
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False, cond=None, return_flat=True):
        out = self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            return_intermediates=return_intermediates,
            cond=cond,
        )
        if return_intermediates:
            samples, intermediates = out
            intermediates = [s / self.input_scale for s in intermediates]
        else:
            samples = out
            intermediates = None
        samples = samples / self.input_scale
        if return_flat and samples.ndim == 4 and samples.shape[2:] == (1, 1):
            samples = samples[:, :, 0, 0]
            if return_intermediates:
                intermediates = [s[:, :, 0, 0] for s in intermediates]
        return (samples, intermediates) if return_intermediates else samples


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm"]

    def forward(self, x, t, c_concat=None, c_crossattn=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()
        return out


class _NullContextManager(object):
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _EmaContext(object):
    def __init__(self, ddpm):
        self.ddpm = ddpm

    def __enter__(self):
        self.ddpm.model_ema.store(self.ddpm.model.parameters())
        self.ddpm.model_ema.copy_to(self.ddpm.model)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ddpm.model_ema.restore(self.ddpm.model.parameters())
        return False
