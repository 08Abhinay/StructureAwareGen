"""
Cross-Attention Transformer for Segmentation-aware RDM.

Separates global token (MoCo CLS / I-JEPA CLS) from segment tokens
(I-JEPA region embeddings) and uses cross-attention for inter-space
communication, since they live in different embedding spaces.

Block structure per layer:
  1. Segment self-attention   (segments attend to each other)
  2. Global→Segments cross-attention  (global queries, segment keys/values)
  3. Segments→Global cross-attention  (segment queries, global key/value)
  4. FFN for segments
  5. FFN for global

Drop-in replacement for UnifiedSegTransformer — same forward() signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from rdm.modules.diffusionmodules.unified_transformer import (
    AdaptiveLayerNorm,
    SinusoidalPosEmb,
)


class CrossAttentionBlock(nn.Module):
    """
    A single layer with:
      - Segment self-attention
      - Bidirectional cross-attention between global and segments
      - Separate FFNs for global and segment streams
    All sub-layers use adaptive (DiT-style) layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        # --- segment self-attention ---
        self.norm_seg_sa = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.seg_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # --- global → segments cross-attention (global as Q) ---
        self.norm_global_ca = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.norm_seg_kv_for_global = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.global_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # --- segments → global cross-attention (segments as Q) ---
        self.norm_seg_ca = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.norm_global_kv_for_seg = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.seg_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # --- feedforward networks (separate for each stream) ---
        self.norm_seg_ff = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.seg_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm_global_ff = AdaptiveLayerNorm(d_model, time_emb_dim)
        self.global_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        global_tok: torch.Tensor,
        seg_tok: torch.Tensor,
        time_emb: torch.Tensor,
        seg_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            global_tok: [B, 1, D]
            seg_tok:    [B, N_seg, D]
            time_emb:   [B, time_emb_dim]
            seg_padding_mask: [B, N_seg] bool, True = padded
        Returns:
            global_tok: [B, 1, D]
            seg_tok:    [B, N_seg, D]
        """
        # 1) Segment self-attention
        seg_norm = self.norm_seg_sa(seg_tok, time_emb)
        seg_sa_out, _ = self.seg_self_attn(
            seg_norm, seg_norm, seg_norm,
            key_padding_mask=seg_padding_mask,
            need_weights=False,
        )
        seg_tok = seg_tok + seg_sa_out

        # 2) Global → Segments cross-attention  (global queries attend to segment keys)
        global_q = self.norm_global_ca(global_tok, time_emb)          # [B, 1, D]
        seg_kv = self.norm_seg_kv_for_global(seg_tok, time_emb)       # [B, N_seg, D]
        global_ca_out, _ = self.global_cross_attn(
            global_q, seg_kv, seg_kv,
            key_padding_mask=seg_padding_mask,
            need_weights=False,
        )
        global_tok = global_tok + global_ca_out

        # 3) Segments → Global cross-attention  (segment queries attend to global key)
        seg_q = self.norm_seg_ca(seg_tok, time_emb)                   # [B, N_seg, D]
        global_kv = self.norm_global_kv_for_seg(global_tok, time_emb) # [B, 1, D]
        seg_ca_out, _ = self.seg_cross_attn(
            seg_q, global_kv, global_kv,
            need_weights=False,
            # No key_padding_mask needed — global is always valid (single token)
        )
        seg_tok = seg_tok + seg_ca_out

        # 4) Segment FFN
        seg_ff_norm = self.norm_seg_ff(seg_tok, time_emb)
        seg_tok = seg_tok + self.seg_ffn(seg_ff_norm)

        # 5) Global FFN
        global_ff_norm = self.norm_global_ff(global_tok, time_emb)
        global_tok = global_tok + self.global_ffn(global_ff_norm)

        return global_tok, seg_tok


class CrossAttentionSegTransformer(nn.Module):
    """
    Cross-Attention Transformer for diffusion-based generation of
    global + segmentation tokens that live in different embedding spaces.

    Drop-in replacement for UnifiedSegTransformer.
    Same forward() signature and I/O shapes.
    """

    def __init__(
        self,
        token_dim: int = 256,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        if d_ff is None:
            d_ff = 4 * d_model

        # Separate input projections for each embedding space
        self.input_proj_global = nn.Linear(token_dim, d_model)
        self.input_proj_seg = nn.Linear(token_dim, d_model)

        # Positional embeddings for segments (global has only 1 position)
        self.global_pos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seg_pos_emb = nn.Parameter(
            torch.randn(1, max_seq_len - 1, d_model) * 0.02
        )

        # Timestep embedding network (shared)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Cross-attention blocks
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    time_emb_dim=time_emb_dim,
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer norms (separate)
        self.final_norm_global = nn.LayerNorm(d_model)
        self.final_norm_seg = nn.LayerNorm(d_model)

        # Separate output projections back to token_dim
        self.output_proj_global = nn.Linear(d_model, token_dim)
        self.output_proj_seg = nn.Linear(d_model, token_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init output projections for stable start
        nn.init.zeros_(self.output_proj_global.weight)
        nn.init.zeros_(self.output_proj_seg.weight)
        if self.output_proj_global.bias is not None:
            nn.init.zeros_(self.output_proj_global.bias)
        if self.output_proj_seg.bias is not None:
            nn.init.zeros_(self.output_proj_seg.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Same interface as UnifiedSegTransformer.forward().

        Args:
            x: [B, 256, 1, N] or [B, N, 256]  (position 0 = global)
            timesteps: [B]
            context: ignored (API compat)
            padding_mask: [B, N] bool, True = padded

        Returns:
            Same shape as input.
        """
        # --- reshape to [B, N, C] ---
        if len(x.shape) == 4:  # [B, C, 1, N]
            B, C, H, N = x.shape
            assert H == 1 and C == self.token_dim
            x = x.squeeze(2).transpose(1, 2)  # [B, N, C]
            reshape_output = True
        elif len(x.shape) == 3:
            B, N, C = x.shape
            assert C == self.token_dim
            reshape_output = False
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        assert N <= self.max_seq_len

        # --- split global / segments ---
        global_raw = x[:, :1, :]   # [B, 1, token_dim]
        seg_raw = x[:, 1:, :]      # [B, N-1, token_dim]
        n_seg = seg_raw.shape[1]

        # --- separate input projections ---
        global_tok = self.input_proj_global(global_raw)  # [B, 1, d_model]
        seg_tok = self.input_proj_seg(seg_raw)            # [B, N-1, d_model]

        # --- add positional embeddings ---
        global_tok = global_tok + self.global_pos_emb
        seg_tok = seg_tok + self.seg_pos_emb[:, :n_seg, :]

        # --- timestep embedding ---
        time_emb = self.time_embed(timesteps)  # [B, time_emb_dim]

        # --- build segment padding mask (exclude global position) ---
        seg_pad_mask = None
        if padding_mask is not None:
            seg_pad_mask = padding_mask[:, 1:]  # [B, N-1]

        # --- transformer blocks ---
        for block in self.blocks:
            global_tok, seg_tok = block(
                global_tok, seg_tok, time_emb, seg_pad_mask
            )

        # --- final norms ---
        global_tok = self.final_norm_global(global_tok)
        seg_tok = self.final_norm_seg(seg_tok)

        # --- output projections ---
        global_out = self.output_proj_global(global_tok)  # [B, 1, token_dim]
        seg_out = self.output_proj_seg(seg_tok)            # [B, N-1, token_dim]

        # --- recombine ---
        out = torch.cat([global_out, seg_out], dim=1)  # [B, N, token_dim]

        if reshape_output:
            out = out.transpose(1, 2).unsqueeze(2)  # [B, C, 1, N]

        return out
