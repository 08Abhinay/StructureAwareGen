"""
Unified Transformer for Segmentation-aware RDM
Handles variable-length sequences: [global_token, seg_1, seg_2, ..., seg_N]
where N ranges from 145-200 based on SAM segmentation output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with timestep conditioning (DiT-style).
    Modulates scale and shift based on diffusion timestep embedding.
    """
    def __init__(self, d_model: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model, bias=True)
        )
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            time_emb: [B, time_emb_dim] timestep embeddings
        Returns:
            [B, N, D] normalized and modulated tokens
        """
        # Normalize
        x = self.norm(x)
        
        # Get scale and shift from timestep
        scale_shift = self.modulation(time_emb)  # [B, 2*D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each [B, D]
        
        # Apply adaptive modulation
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block with adaptive normalization.
    Includes self-attention, feedforward, and residual connections.
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
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Pre-attention adaptive norm
        self.norm1 = AdaptiveLayerNorm(d_model, time_emb_dim)
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Pre-feedforward adaptive norm
        self.norm2 = AdaptiveLayerNorm(d_model, time_emb_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            time_emb: [B, time_emb_dim] timestep embeddings
            padding_mask: [B, N] boolean mask (True = padded/ignore)
        Returns:
            [B, N, D] transformed tokens
        """
        # Self-attention with residual
        x_norm = self.norm1(x, time_emb)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        x = x + attn_out
        
        # Feedforward with residual
        x_norm = self.norm2(x, time_emb)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps
        Returns:
            [B, dim] sinusoidal embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class UnifiedSegTransformer(nn.Module):
    """
    Unified Transformer for diffusion-based generation of both global and segmentation tokens.
    
    Architecture:
    - Handles sequences: [global_token, seg_1, ..., seg_N] where N=145-200
    - Token type embeddings distinguish global (type 0) vs segments (type 1)
    - Learnable positional encodings for sequence order
    - Adaptive layer normalization for timestep conditioning
    - Supports variable sequence lengths via padding masks
    
    Design optimized for:
    - 256-dim tokens (matching SAM and I-JEPA outputs)
    - 768 d_model transformer (standard BERT-base size)
    - 8-12 layers for sufficient capacity
    - Efficient O(n²) attention for n~200 tokens
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
        """
        Args:
            token_dim: Dimension of input/output tokens (256 for SAM/I-JEPA)
            d_model: Hidden dimension of transformer (768 standard)
            n_heads: Number of attention heads (12 standard)
            n_layers: Number of transformer blocks (8-12 recommended)
            d_ff: Feedforward dimension (default: 4 * d_model)
            dropout: Dropout probability
            max_seq_len: Maximum sequence length (250 segments + 1 global + buffer)
            time_emb_dim: Dimension of timestep embeddings
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        # Input projection: token_dim -> d_model
        self.input_proj = nn.Linear(token_dim, d_model)
        
        # Token type embeddings: 0=global, 1=segment
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        # Learnable positional encodings
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Timestep embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection: d_model -> token_dim
        self.output_proj = nn.Linear(d_model, token_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        # Xavier uniform for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
        # Zero-initialize output projection for stable initialization
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def create_token_type_ids(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create token type IDs: 0 for global token, 1 for all segment tokens.
        
        Args:
            seq_len: Sequence length (including global token)
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            [B, N] token type IDs
        """
        token_types = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        token_types[:, 0] = 0  # First token is global
        return token_types
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through unified transformer.
        
        Args:
            x: [B, 256, 1, N] or [B, N, 256] input tokens
               First token is global, remaining N-1 are segments
            timesteps: [B] diffusion timestep integers (0 to num_timesteps-1)
            context: Optional conditioning (not used, for API compatibility)
            padding_mask: [B, N] boolean mask (True = padded position to ignore)
            
        Returns:
            [B, 256, 1, N] or [B, N, 256] denoised tokens (same shape as input)
        """
        # Handle different input shapes
        input_shape = x.shape
        if len(x.shape) == 4:  # [B, C, 1, N] format from DDPM
            B, C, H, N = x.shape
            assert H == 1 and C == self.token_dim, f"Expected [B, {self.token_dim}, 1, N], got {x.shape}"
            x = x.squeeze(2).transpose(1, 2)  # [B, N, 256]
            reshape_output = True
        elif len(x.shape) == 3:  # [B, N, 256] format
            B, N, C = x.shape
            assert C == self.token_dim, f"Expected last dim {self.token_dim}, got {C}"
            reshape_output = False
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        # Project to d_model
        x = self.input_proj(x)  # [B, N, d_model]
        
        # Add token type embeddings
        token_type_ids = self.create_token_type_ids(seq_len, B, x.device)
        x = x + self.token_type_embeddings(token_type_ids)
        
        # Add positional embeddings
        x = x + self.pos_embeddings[:, :seq_len, :]
        
        # Get timestep embeddings
        time_emb = self.time_embed(timesteps)  # [B, time_emb_dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, time_emb, padding_mask)
            
        # Final normalization
        x = self.final_norm(x)
        
        # Project back to token_dim
        x = self.output_proj(x)  # [B, N, 256]
        
        # Reshape output if needed
        if reshape_output:
            x = x.transpose(1, 2).unsqueeze(2)  # [B, 256, 1, N]
            
        return x


# Utility function for testing
def test_unified_transformer():
    """Test the UnifiedSegTransformer with various configurations."""
    print("Testing UnifiedSegTransformer...")
    
    # Test parameters
    batch_size = 4
    num_segments = 180
    seq_len = num_segments + 1  # +1 for global token
    token_dim = 256
    
    # Create model
    model = UnifiedSegTransformer(
        token_dim=token_dim,
        d_model=768,
        n_heads=12,
        n_layers=8,
        dropout=0.1,
        max_seq_len=256,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test DDPM-style input [B, C, 1, N]
    x_ddpm = torch.randn(batch_size, token_dim, 1, seq_len)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"\nInput shape (DDPM): {x_ddpm.shape}")
    output_ddpm = model(x_ddpm, timesteps)
    print(f"Output shape (DDPM): {output_ddpm.shape}")
    assert output_ddpm.shape == x_ddpm.shape, "Shape mismatch!"
    
    # Test sequence-style input [B, N, C]
    x_seq = torch.randn(batch_size, seq_len, token_dim)
    output_seq = model(x_seq, timesteps)
    print(f"\nInput shape (Sequence): {x_seq.shape}")
    print(f"Output shape (Sequence): {output_seq.shape}")
    assert output_seq.shape == x_seq.shape, "Shape mismatch!"
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, 150:] = True  # Mask last tokens for first sample
    padding_mask[1, 170:] = True  # Different padding for second sample
    
    output_masked = model(x_ddpm, timesteps, padding_mask=padding_mask)
    print(f"\nWith padding mask: {output_masked.shape}")
    assert output_masked.shape == x_ddpm.shape, "Shape mismatch with mask!"
    
    # Test gradient flow
    loss = output_masked.sum()
    loss.backward()
    print("\nGradient flow: OK")
    
    print("\n✓ All tests passed!")
    

if __name__ == "__main__":
    test_unified_transformer()
