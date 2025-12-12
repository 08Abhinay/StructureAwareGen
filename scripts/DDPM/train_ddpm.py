# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────────────────────
import os, time, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import set_global_policy

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyper‑parameters (EDIT HERE ONLY)  ★
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    img_size        = 256,
    batch_size      = 32,
    num_epochs      = 40,
    timesteps       = 1_000,
    lr              = 2e-4,
    norm_groups     = 8,
    first_channels  = 64,
    channel_mult    = [1,2,4,8],
    num_res_blocks  = 2,
    data_root       = Path("/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/Brain_cancer/Training"),
    save_root       = Path("/scratch/gilbreth/abelde/Thesis/scripts/DDPM_RUN"),
    fid_num_images  = 512,          # fake & real batch size for FID/IS
    sample_grid     = (2,8),        # rows, cols  (set None to disable)
    use_mixed_precision = True,     # turn off if GPU < Ampere
)
CFG["save_root"].mkdir(parents=True, exist_ok=True)

if CFG["use_mixed_precision"]:
    set_global_policy("mixed_float16")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Discover image files
# ──────────────────────────────────────────────────────────────────────────────
EXTS = ["jpg","jpeg","png","bmp","tiff","tif"]
all_files = [str(p) for ext in EXTS for p in CFG["data_root"].rglob(f"*.{ext}")]
if not all_files:
    raise RuntimeError(f"No images found in {CFG['data_root']}")
print(f"✅ {len(all_files)} images found.")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Preprocess fn  (decode → center‑crop → resize → flip → [-1,1])
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)      # [0,1]
    h,w  = tf.shape(img)[0], tf.shape(img)[1]
    s    = tf.minimum(h,w)
    img  = tf.image.crop_to_bounding_box(img, (h-s)//2, (w-s)//2, s, s)
    img  = tf.image.resize(img, [CFG["img_size"], CFG["img_size"]], antialias=True)
    img  = tf.image.random_flip_left_right(img)
    return img*2.0 - 1.0                                     # [-1,1]

# 4. tf.data pipeline (cached & prefetched)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (tf.data.Dataset.from_tensor_slices(all_files)
            .map(preprocess, AUTOTUNE)
            .cache()
            .shuffle(len(all_files))
            .batch(CFG["batch_size"], drop_remainder=True)
            .prefetch(AUTOTUNE))



# ------------------------------- sanity check -----------------------------------------
# One quick batch to ensure shapes & ranges are correct
for batch in train_ds.take(1):
    print("Batch shape:", batch.shape, "range:", (tf.reduce_min(batch).numpy(), tf.reduce_max(batch).numpy()))


class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
    

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, :
        ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_unet():
    widths       = [CFG["first_channels"]*m for m in CFG["channel_mult"]]
    has_attention= [False, False, True, True]
    img_channels = 3
    img_size     = CFG["img_size"]

    # --- same code as your build_model() but using local vars above ---
    image_input = layers.Input((img_size,img_size,img_channels))
    time_input  = layers.Input((), dtype=tf.int64)

    x = layers.Conv2D(CFG["first_channels"],3,padding="same",kernel_initializer=kernel_init(1.0))(image_input)
    temb = TimeEmbedding(CFG["first_channels"]*4)(time_input)
    temb = TimeMLP(CFG["first_channels"]*4)(temb)

    skips=[x]
    for i,w in enumerate(widths):
        for _ in range(CFG["num_res_blocks"]):
            x = ResidualBlock(w, CFG["norm_groups"])([x,temb])
            if has_attention[i]:
                x = AttentionBlock(w, CFG["norm_groups"])(x)
            skips.append(x)
        if w!=widths[-1]:
            x = DownSample(w)(x); skips.append(x)

    x = ResidualBlock(widths[-1], CFG["norm_groups"])([x,temb])
    x = AttentionBlock(widths[-1], CFG["norm_groups"])(x)
    x = ResidualBlock(widths[-1], CFG["norm_groups"])([x,temb])

    for i,w in reversed(list(enumerate(widths))):
        for _ in range(CFG["num_res_blocks"]+1):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(w, CFG["norm_groups"])([x,temb])
            if has_attention[i]:
                x = AttentionBlock(w, CFG["norm_groups"])(x)
        if i!=0: x = UpSample(w)(x)

    x = layers.GroupNormalization(groups=CFG["norm_groups"])(x)
    x = keras.activations.swish(x)
    out = layers.Conv2D(3,3,padding="same",kernel_initializer=kernel_init(0.))(x)
    return keras.Model([image_input,time_input], out, name="unet")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Torch‑metrics helpers
# ──────────────────────────────────────────────────────────────────────────────
device_tm = "cuda" if torch.cuda.is_available() else "cpu"
fid_metric = FrechetInceptionDistance(normalize=True).to(device_tm)
is_metric  = InceptionScore(normalize=True).to(device_tm)

def eval_fid_is(fake, real):
    fake = torch.tensor(fake/255.,dtype=torch.float32).permute(0,3,1,2).to(device_tm)
    real = torch.tensor(real/255.,dtype=torch.float32).permute(0,3,1,2).to(device_tm)
    fid_metric.reset(); fid_metric.update(real,True); fid_metric.update(fake,False)
    is_metric.reset();  is_metric.update(fake)
    fid = fid_metric.compute().item()
    is_m,is_s = map(float, is_metric.compute())
    return fid,is_m,is_s

# ──────────────────────────────────────────────────────────────────────────────
# 8. DiffusionModel (train_step + generate_images)
# ──────────────────────────────────────────────────────────────────────────────
class DiffusionModel(keras.Model):
    def __init__(self, net, ema_net, gd, ema=0.999):
        super().__init__()
        self.net, self.ema_net, self.gd, self.ema = net, ema_net, gd, ema

    def train_step(self, images):
        bs = tf.shape(images)[0]
        t  = tf.random.uniform((bs,),0,self.gd.timesteps,dtype=tf.int64)
        noise = tf.random.normal(tf.shape(images), dtype=images.dtype)
        with tf.GradientTape() as tape:
            x_t  = self.gd.q_sample(images,t,noise)
            pred = self.net([x_t,t], training=True)
            loss = self.compiled_loss(noise, pred)
        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        for w,ew in zip(self.net.weights, self.ema_net.weights):
            ew.assign(self.ema*ew + (1-self.ema)*w)
        return {"loss":loss}

    def generate_images(self, n):
        x = tf.random.normal((n,CFG["img_size"],CFG["img_size"],3))
        for t in reversed(range(self.gd.timesteps)):
            tt = tf.cast(tf.fill((n,),t),tf.int64)
            eps = self.ema_net([x,tt], training=False)
            x   = self.gd.p_sample(eps,x,tt)
        return x

# ──────────────────────────────────────────────────────────────────────────────
# 9. Keras Callbacks  (Checkpoint, FID/IS Logger, Sample grid)
# ──────────────────────────────────────────────────────────────────────────────
class CheckpointResume(keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__(); self.dir=save_dir; self.dir.mkdir(exist_ok=True)
    def on_train_begin(self, logs=None):
        latest = tf.train.latest_checkpoint(self.dir)
        if latest: print(f"🔄 Resuming from {latest}"); self.model.load_weights(latest)
    def on_epoch_end(self, epoch, logs=None):  # always save
        self.model.save_weights(self.dir/f"weights_{epoch:04d}.ckpt")

class FIDISCallback(keras.callbacks.Callback):
    def __init__(self, real_ds):
        super().__init__()
        self.real = next(iter(real_ds.unbatch().batch(CFG["fid_num_images"])))
        self.kimg = 0; self.t0=time.time()
    def on_epoch_end(self, epoch, logs=None):
        fake = self.model.generate_images(CFG["fid_num_images"])
        fake = tf.clip_by_value((fake+1)*127.5,0,255).numpy().astype(np.uint8)
        real = tf.clip_by_value((self.real+1)*127.5,0,255).numpy().astype(np.uint8)
        fid,is_m,is_s = eval_fid_is(fake,real)
        self.kimg += len(train_ds)*CFG["batch_size"]/1000
        mins = (time.time()-self.t0)/60
        print(f"tick {epoch:<4d} kimg {self.kimg:6.1f}  FID {fid:6.2f}  "
              f"IS {is_m:5.2f}±{is_s:4.2f}  time {mins:5.1f}m")
        logs.update({"fid":fid,"inception_score":is_m})

class SampleGridCallback(keras.callbacks.Callback):
    def __init__(self, rows, cols, out_dir):
        super().__init__(); self.r,self.c = rows,cols; self.dir=out_dir
    def on_epoch_end(self, epoch, logs=None):
        n = self.r*self.c
        imgs = self.model.generate_images(n)
        imgs = tf.clip_by_value((imgs+1)*127.5,0,255).numpy().astype(np.uint8)
        fig,ax = plt.subplots(self.r,self.c,figsize=(2*self.c,2*self.r))
        for i,img in enumerate(imgs): ax.flat[i].imshow(img); ax.flat[i].axis("off")
        plt.tight_layout(); p=self.dir/f"grid_{epoch:03d}.png"; fig.savefig(p); plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 10. Build model, compile & train
# ──────────────────────────────────────────────────────────────────────────────
unet        = build_unet()
ema_unet    = build_unet(); ema_unet.set_weights(unet.get_weights())
gdf         = GaussianDiffusion(timesteps=CFG["timesteps"])
model       = DiffusionModel(unet, ema_unet, gdf)

model.compile(
    loss      = keras.losses.MeanSquaredError(),
    optimizer = keras.optimizers.Adam(CFG["lr"])
)

cbs=[CheckpointResume(CFG["save_root"]),
     FIDISCallback(train_ds)]
if CFG["sample_grid"]: cbs.append(SampleGridCallback(*CFG["sample_grid"], CFG["save_root"]))

model.fit(train_ds,
          epochs=CFG["num_epochs"],
          steps_per_epoch=len(train_ds),
          callbacks=cbs)