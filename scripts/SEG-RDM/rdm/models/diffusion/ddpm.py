import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from rdm.util import exists, default, count_params, instantiate_from_config, rank_zero_only
from rdm.modules.ema import LitEma
from rdm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from rdm.pretrained_enc import models_pretrained_enc
# from rdm.env_debug import print_env
# print_env(__name__, globals())



__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
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

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(logvar, requires_grad=True)
        else:
            self.register_buffer('logvar', logvar)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        # Ensure a plain numpy array (avoids OmegaConf/ListConfig or Tensor quirks).
        betas = np.asarray(betas, dtype=np.float64)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * self.alphas_cumprod * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * torch.sqrt(self.alphas_cumprod) / (2. * 1 - self.alphas_cumprod)
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if k == "image":
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x


class RDM(DDPM):
    """main class"""
    def __init__(self,
                 cond_stage_config,
                 class_cond=False,
                 input_scale=1.0,
                 pretrained_enc_config=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 first_stage_config=None,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.input_scale = input_scale
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.class_cond = class_cond
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if pretrained_enc_config is not None:
            self.instantiate_pretrained_enc(pretrained_enc_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_pretrained_enc(self, config):
        self.pretrained_encoder = models_pretrained_enc.__dict__[config.params.pretrained_enc_arch](
            proj_dim=config.params.get("proj_dim", 256))
        # load pre-trained encoder parameters
        if 'moco' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_moco(self.pretrained_encoder,
                                                                                 config.params.pretrained_enc_path)
        elif 'dino' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_dino(self.pretrained_encoder,
                                                                                 config.params.pretrained_enc_path)
        elif 'ibot' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_ibot(self.pretrained_encoder,
                                                                                 config.params.pretrained_enc_path)
        elif 'mae' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_mae(self.pretrained_encoder,
                                                                                config.params.pretrained_enc_path)
        elif 'deit' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_deit(self.pretrained_encoder,
                                                                                 config.params.pretrained_enc_path)
        elif 'ijepa' in config.params.pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_ijepa(self.pretrained_encoder,
                                                                                  config.params.pretrained_enc_path)
        else:
            raise NotImplementedError

        self.pretrained_encoder.eval()
        self.pretrained_encoder.train = disabled_train
        self.pretrained_enc_proj = None
        try:
            self.pretrained_enc_withproj = config.params.pretrained_enc_withproj
        except:
            self.pretrained_enc_withproj = False

        if self.pretrained_enc_withproj:
            enc_out_dim = config.params.get("proj_dim", self.channels)
        else:
            enc_out_dim = getattr(self.pretrained_encoder, "embed_dim", None)
            if enc_out_dim is None:
                enc_out_dim = getattr(self.pretrained_encoder, "num_features", None)
            if enc_out_dim is None and hasattr(self.pretrained_encoder, "head"):
                head = self.pretrained_encoder.head
                if hasattr(head, "weight"):
                    enc_out_dim = head.weight.shape[1]
        if enc_out_dim is not None and enc_out_dim != self.channels:
            self.pretrained_enc_proj = nn.Linear(enc_out_dim, self.channels)

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        device = self.betas.device
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        # img to feature
        with torch.no_grad():
            self.pretrained_encoder.eval()
            device = x.device
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x_normalized = (x - mean) / std
            x_normalized = torch.nn.functional.interpolate(x_normalized, 224, mode='bicubic', align_corners=False)
            rep = self.pretrained_encoder.forward_features(x_normalized)
            if rep.dim() == 3:
                # timm ViT returns tokens; pool before optional projection head.
                if hasattr(self.pretrained_encoder, "forward_head"):
                    rep = self.pretrained_encoder.forward_head(
                        rep, pre_logits=not self.pretrained_enc_withproj
                    )
                else:
                    rep = rep[:, 0]
                    if self.pretrained_enc_withproj:
                        rep = self.pretrained_encoder.head(rep)
            elif self.pretrained_enc_withproj:
                rep = self.pretrained_encoder.head(rep)
        if self.pretrained_enc_proj is not None:
            rep = self.pretrained_enc_proj(rep)
        rep_std = torch.std(rep, dim=1, keepdim=True)
        rep_mean = torch.mean(rep, dim=1, keepdim=True)
        rep = (rep - rep_mean) / rep_std

        x = rep.unsqueeze(-1).unsqueeze(-1)
        x = x * self.input_scale
        z = x

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
        else:
            c = None
            xc = None
        out = [z, c]
        if return_original_cond:
            out.append(xc)
        return out

    def forward(self, x, c, batch=None, gen_img=False, cfg=0.0, class_label_gen=None, *args, **kwargs):
        if gen_img:
            return self.gen_imgs(cfg=cfg, class_label_gen=class_label_gen)
        if batch is not None:
            x, c = self.get_input(batch, self.first_stage_key)
            if isinstance(c, dict) and 'class_label' in c:
                c = {'class_label': c['class_label']}
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                cond_ids = self.cond_ids.to(t.device)
                tc = cond_ids[t]
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        if self.use_ema and batch is not None:
            self.model_ema(self.model)
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar = self.logvar.to(x_start.device)
        logvar_t = logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        device = self.betas.device
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, dtype=torch.long, device=device)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                cond_ids = self.cond_ids.to(device)
                tc = cond_ids[ts]
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                cond_ids = self.cond_ids.to(device)
                tc = cond_ids[ts]
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)


class UnifiedSegRDM(RDM):
    """Unified Representation Diffusion Model for Global + Segmentation Tokens.
    
    Extends RDM to handle variable-length sequences:
    - 1 global token (from I-JEPA encoder)
    - N segmentation tokens (from SAM, N=145-200)
    
    Uses pre-computed SAM embeddings instead of computing them during training.
    """
    
    def __init__(self, seg_npz_dir: str = None, max_segments: int = 250, 
                 lambda_diversity: float = 0.1, lambda_alignment: float = 0.05,
                 *args, **kwargs):
        """
        Args:
            seg_npz_dir: Path to directory containing SAM .npz files
            max_segments: Maximum number of segments (for padding)
            lambda_diversity: Weight for diversity loss (default: 0.1)
            lambda_alignment: Weight for alignment loss (default: 0.05)
        """
        super().__init__(*args, **kwargs)
        self.seg_npz_dir = seg_npz_dir
        self.max_segments = max_segments
        self.lambda_diversity = lambda_diversity
        self.lambda_alignment = lambda_alignment
        self.first_stage_key = "image"
        print(f"UnifiedSegRDM: max_segments={max_segments}, seg_npz_dir={seg_npz_dir}")
        print(f"  Diversity loss weight: {lambda_diversity}")
        print(f"  Alignment loss weight: {lambda_alignment}")
        
    def _to_diffusion_format(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert [B, N, C] to [B, C, 1, N] for DDPM compatibility."""
        return tokens.permute(0, 2, 1).unsqueeze(2)
    
    def _from_diffusion_format(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, 1, N] to [B, N, C]."""
        return x.squeeze(2).permute(0, 2, 1)
    
    def compute_diversity_loss(self, seg_tokens, padding_mask=None, eps=1e-6):
        """
        Diversity loss: -log det(C + εI) encourages diverse segment embeddings.
        Prevents collapse where all masks encode the same information.
        
        Args:
            seg_tokens: [B, N_seg, C] segment embeddings (excluding global token)
            padding_mask: [B, N_seg+1] boolean mask (True = padded), includes global
            eps: Small constant for numerical stability
            
        Returns:
            Scalar diversity loss (higher = more collapsed)
        """
        import torch.nn.functional as F
        
        B, N, C = seg_tokens.shape
        
        # Remove padding
        if padding_mask is not None:
            mask = ~padding_mask[:, 1:]  # Exclude global token position
        else:
            mask = torch.ones(B, N, dtype=torch.bool, device=seg_tokens.device)
        
        losses = []
        for i in range(B):
            valid_tokens = seg_tokens[i, mask[i]]  # [N_valid, C]
            if valid_tokens.shape[0] < 2:
                continue  # Need at least 2 tokens for covariance
            
            # Compute covariance matrix
            mean = valid_tokens.mean(dim=0, keepdim=True)  # [1, C]
            centered = valid_tokens - mean  # [N_valid, C]
            cov = (centered.T @ centered) / valid_tokens.shape[0]  # [C, C]
            
            # Regularize and compute log determinant
            cov_reg = cov + eps * torch.eye(C, device=cov.device)
            
            # Use slogdet for numerical stability
            sign, logdet = torch.slogdet(cov_reg)
            if sign > 0:  # Only use if positive definite
                losses.append(-logdet)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=seg_tokens.device)
    
    def compute_alignment_loss(self, global_vec, seg_tokens, padding_mask=None, tau=0.07):
        """
        Alignment loss: Encourages global and segment embeddings to be semantically aligned.
        Uses InfoNCE-style contrastive loss.
        
        Args:
            global_vec: [B, C] global I-JEPA embeddings
            seg_tokens: [B, N_seg, C] segment SAM embeddings
            padding_mask: [B, N_seg+1] boolean mask (True = padded)
            tau: Temperature for contrastive loss
            
        Returns:
            Scalar alignment loss
        """
        import torch.nn.functional as F
        
        B, N, C = seg_tokens.shape
        
        # L2 normalize
        global_norm = F.normalize(global_vec, dim=1)  # [B, C]
        seg_norm = F.normalize(seg_tokens, dim=2)  # [B, N, C]
        
        # Compute similarities: global[i] with all segments
        sim = torch.einsum('bc,bnc->bn', global_norm, seg_norm) / tau  # [B, N]
        
        # Apply padding mask
        if padding_mask is not None:
            mask = ~padding_mask[:, 1:]  # [B, N] exclude global token
            sim = sim.masked_fill(~mask, -1e9)
        
        # Positive: mean similarity with own segments
        if padding_mask is not None:
            mask_float = (~padding_mask[:, 1:]).float()
            pos_sim = (sim * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)  # [B]
        else:
            pos_sim = sim.mean(dim=1)  # [B]
        
        # Negative: cross-image similarities (in-batch negatives)
        # For each sample, compute mean similarity with other samples' segments
        neg_sims = []
        for i in range(B):
            neg_indices = [j for j in range(B) if j != i]
            if neg_indices:
                if padding_mask is not None:
                    mask_neg = ~padding_mask[neg_indices, 1:]  # [B-1, N]
                    sim_neg = torch.einsum('c,bnc->bn', global_norm[i], seg_norm[neg_indices])  # [B-1, N]
                    sim_neg = (sim_neg * mask_neg.float()).sum(dim=1) / mask_neg.float().sum(dim=1).clamp(min=1)
                    neg_sims.append(sim_neg.mean())
                else:
                    sim_neg = torch.einsum('c,bnc->bn', global_norm[i], seg_norm[neg_indices]).mean()
                    neg_sims.append(sim_neg)
        
        if not neg_sims:
            return torch.tensor(0.0, device=global_vec.device)
        
        neg_sim = torch.stack(neg_sims)  # [B]
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)
        )
        
        return loss.mean()
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        Extract unified token sequence from batch.
        
        Returns:
            x: [B, 256, 1, N+1] unified tokens (global + segments)
            padding_mask: [B, N+1] boolean mask (True = padded)
            c: conditioning (if any)
        """
        # Extract raw image from batch dict directly (skip DDPM.get_input's buggy rearrange)
        # Our dataset returns [B, C, H, W] format already (PyTorch standard from ToTensor)
        # DDPM.get_input assumes [B, H, W, C] and does rearrange, which breaks our data
        if isinstance(batch, dict) and k in batch:
            x_img = batch[k]
        else:
            # Fallback for non-dict batches
            x_img = batch
        
        if bs is not None:
            x_img = x_img[:bs]
        
        x_img = x_img.to(memory_format=torch.contiguous_format).float()
        device = x_img.device
        
        # Check if pre-cached IJEPA embeddings are available in batch
        if 'ijepa_emb' in batch and batch['ijepa_emb'] is not None:
            # Use pre-cached embeddings (much faster!)
            rep = batch['ijepa_emb'].to(device)  # [B, 1280] raw ViT-H/14 dimension
            
            # Project to 256-dim if needed
            if self.pretrained_enc_proj is not None:
                rep = self.pretrained_enc_proj(rep)
                
            # Normalize global vector
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std
            rep = rep * self.input_scale
            
            global_vec = rep  # [B, 256]
        else:
            # Fall back to runtime extraction using I-JEPA encoder
            with torch.no_grad():
                self.pretrained_encoder.eval()
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                x_normalized = (x_img - mean) / std
                x_normalized = torch.nn.functional.interpolate(x_normalized, 224, mode='bicubic', align_corners=False)
                rep = self.pretrained_encoder.forward_features(x_normalized)
                
                if rep.dim() == 3:
                    if hasattr(self.pretrained_encoder, "forward_head"):
                        rep = self.pretrained_encoder.forward_head(
                            rep, pre_logits=not self.pretrained_enc_withproj
                        )
                    else:
                        rep = rep[:, 0]
                        if self.pretrained_enc_withproj:
                            rep = self.pretrained_encoder.head(rep)
                elif self.pretrained_enc_withproj:
                    rep = self.pretrained_encoder.head(rep)
                    
            if self.pretrained_enc_proj is not None:
                rep = self.pretrained_enc_proj(rep)
                
            # Normalize global vector
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std
            rep = rep * self.input_scale
            
            global_vec = rep  # [B, 256]
        
        # Get segmentation tokens from batch (pre-computed SAM embeddings)
        if 'seg_embs' in batch:
            seg_tokens = batch['seg_embs'].to(device)  # [B, max_segments, 256]
            num_segments = batch['num_segments']  # [B] actual segment counts
        else:
            raise ValueError("Batch must contain 'seg_embs' and 'num_segments' from SAM")
            
        # Create padding mask
        B, max_seg, C = seg_tokens.shape
        padding_mask = torch.zeros(B, max_seg + 1, dtype=torch.bool, device=device)
        for i, n in enumerate(num_segments):
            if n < max_seg:
                padding_mask[i, n+1:] = True  # +1 to account for global token
        
        # Concatenate: [global, seg_1, ..., seg_N]
        unified_tokens = torch.cat([
            global_vec.unsqueeze(1),  # [B, 1, 256]
            seg_tokens                # [B, max_segments, 256]
        ], dim=1)  # [B, max_segments+1, 256]
        
        # Convert to DDPM format
        x = self._to_diffusion_format(unified_tokens)  # [B, 256, 1, max_segments+1]
        
        # Handle conditioning (same as RDM)
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    # Get conditioning data (not the main image)
                    xc = DDPM.get_input(self, batch, cond_key).to(device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
        else:
            c = None
            xc = None
            
        out = [x, c, padding_mask]
        if return_original_cond:
            out.append(xc)
        return out
    
    def apply_model(self, x_noisy, t, cond, padding_mask=None, return_ids=False):
        """Apply model with padding mask support."""
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        # Pass padding mask to model
        x_recon = self.model(x_noisy, t, padding_mask=padding_mask, **cond)
        
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    def p_losses(self, x_start, cond, t, padding_mask=None, noise=None):
        """Compute losses with padding mask support."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, padding_mask=padding_mask)
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        # Compute per-element loss
        loss_simple = self.get_loss(model_output, target, mean=False)  # [B, C, 1, N]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Expand mask to match loss shape: [B, N] -> [B, 1, 1, N]
            mask_expanded = (~padding_mask).float().unsqueeze(1).unsqueeze(2)
            loss_simple = loss_simple * mask_expanded
            # Average over non-padded tokens only
            loss_simple = loss_simple.sum(dim=[1, 2, 3]) / mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1)
        else:
            loss_simple = loss_simple.mean(dim=[1, 2, 3])
        
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        
        logvar = self.logvar.to(x_start.device)
        logvar_t = logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        
        loss = self.l_simple_weight * loss.mean()
        
        # VLB loss with masking
        loss_vlb_per_elem = self.get_loss(model_output, target, mean=False)
        if padding_mask is not None:
            loss_vlb_per_elem = loss_vlb_per_elem * mask_expanded
            loss_vlb = (loss_vlb_per_elem.sum(dim=[1, 2, 3]) / mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1))
        else:
            loss_vlb = loss_vlb_per_elem.mean(dim=(1, 2, 3))
            
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        # Add diversity and alignment losses (only during training)
        if self.training and (self.lambda_diversity > 0 or self.lambda_alignment > 0):
            # Reconstruct x_0 from model output
            if self.parameterization == "eps":
                # Predict x_0 from noise prediction
                x_recon = self.predict_start_from_noise(x_noisy, t, model_output)
            elif self.parameterization == "x0":
                x_recon = model_output
            else:
                x_recon = x_start  # Fallback
            
            # Convert to token space: [B, C, 1, N+1] -> [B, N+1, C]
            tokens_recon = self._from_diffusion_format(x_recon)  # [B, N+1, C]
            
            # Split global and segment tokens
            global_recon = tokens_recon[:, 0, :]  # [B, C]
            seg_recon = tokens_recon[:, 1:, :]  # [B, N, C]
            
            # Compute diversity loss
            if self.lambda_diversity > 0:
                loss_div = self.compute_diversity_loss(seg_recon, padding_mask)
                loss_dict[f'{prefix}/loss_diversity'] = loss_div
                loss = loss + self.lambda_diversity * loss_div
            
            # Compute alignment loss
            if self.lambda_alignment > 0:
                loss_align = self.compute_alignment_loss(global_recon, seg_recon, padding_mask)
                loss_dict[f'{prefix}/loss_alignment'] = loss_align
                loss = loss + self.lambda_alignment * loss_align
        
        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict
    
    def forward(self, x, c, batch=None, gen_img=False, cfg=0.0, class_label_gen=None, *args, **kwargs):
        """Forward pass with padding mask extraction."""
        if gen_img:
            return self.gen_imgs(cfg=cfg, class_label_gen=class_label_gen)
        if batch is not None:
            result = self.get_input(batch, self.first_stage_key)
            x, c, padding_mask = result[0], result[1], result[2]
            if isinstance(c, dict) and 'class_label' in c:
                c = {'class_label': c['class_label']}
        else:
            padding_mask = kwargs.get('padding_mask', None)
            
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                cond_ids = self.cond_ids.to(t.device)
                tc = cond_ids[t]
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        loss, loss_dict = self.p_losses(x, c, t, padding_mask=padding_mask, *args, **kwargs)
        if self.use_ema and batch is not None:
            self.model_ema(self.model)
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, num_segments=180, **kwargs):
        """Sample unified tokens (global + segments)."""
        if shape is None:
            # Shape: [B, 256, 1, num_segments+1]
            shape = (batch_size, self.channels, 1, num_segments + 1)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, padding_mask: torch.Tensor = None):
        """Forward with optional padding mask support for variable-length sequences."""
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, padding_mask=padding_mask)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, padding_mask=padding_mask)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, padding_mask=padding_mask)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, padding_mask=padding_mask)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc, padding_mask=padding_mask)
        else:
            raise NotImplementedError()

        return out