import torch
import torch.nn as nn

from seg_rdm.diffusion.utils import timestep_embedding, zero_module


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=512,
    ):
        super().__init__()
        self.use_context = use_context

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(mid_channels, channels, bias=True)),
        )

        if use_context:
            self.context_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, mid_channels, bias=True),
            )
        else:
            self.context_layers = None

    def forward(self, x, emb, context=None):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        if self.use_context:
            if context is None:
                raise ValueError("Context is required when use_context=True")
            context_out = self.context_layers(context)
            h = h + emb_out + context_out
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h


class SimpleMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        bottleneck_channels,
        out_channels,
        num_res_blocks,
        dropout=0.0,
        use_context=False,
        context_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.use_context = use_context

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                    bottleneck_channels,
                    time_embed_dim,
                    dropout,
                    use_context=use_context,
                    context_channels=context_channels,
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def _flatten_input(self, x):
        if x.ndim == 2:
            return x, False
        if x.ndim == 4 and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0], True
        raise ValueError(f"Expected input shape [B, C] or [B, C, 1, 1], got {x.shape}.")

    def _flatten_context(self, context, batch_size):
        if context is None:
            return None
        if context.ndim == 2:
            return context
        if context.ndim == 4 and context.shape[2] == 1 and context.shape[3] == 1:
            return context[:, :, 0, 0]
        if context.ndim == 3:
            # [B, T, C] -> pool over tokens
            return context.mean(dim=1)
        raise ValueError(f"Unsupported context shape {context.shape} for batch {batch_size}.")

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        _ = y
        x, was_4d = self._flatten_input(x)
        if self.use_context:
            context = self._flatten_context(context, x.shape[0])
        else:
            context = None

        x = self.input_proj(x)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        for block in self.res_blocks:
            x = block(x, emb, context)

        out = self.out(x)
        if was_4d:
            return out.unsqueeze(-1).unsqueeze(-1)
        return out
