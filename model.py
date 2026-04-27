"""
DiT-style generator for Drifting Models.
Adapted for 32x32 CIFAR-10 with adaLN-Zero conditioning.

Key differences from standard DiT:
- No timestep input (one-step generator, not diffusion)
- Conditioning = class_embed + alpha_embed + style_embed
- Uses register tokens, RoPE, SwiGLU, RMSNorm, QK-Norm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with QK-Norm and optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # QK-Norm for training stability
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Block with adaLN-Zero conditioning.

    Regresses 6 modulation parameters from conditioning:
    - shift_msa, scale_msa, gate_msa (for attention)
    - shift_mlp, scale_mlp, gate_mlp (for MLP)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        # Pre-norm with RMSNorm (no learned affine - will be modulated)
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        # Use SwiGLU as per paper
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        # adaLN-Zero modulation: 6 parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        # Self-attention with adaLN + gating
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )

        # MLP with adaLN + gating
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and linear projection."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

        # adaLN modulation: 2 parameters (shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings using Conv2d."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class LabelEmbedder(nn.Module):
    """Embed class labels with null class for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # +1 for null/unconditional class
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomly drop labels to null class for CFG training."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """Embed CFG strength using Fourier features.

    Note: the variable ``alpha`` here is the same quantity denoted ``gamma`` in
    the paper (CFG strength). The internal name ``alpha`` is kept so that the
    state dict keys (``alpha_embed.*``) remain compatible with the released
    checkpoints.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        """Create sinusoidal embeddings for alpha."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=alpha.device) / half
        )
        args = alpha[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier_features(alpha, self.frequency_embedding_size)
        return self.mlp(fourier)


class StyleEmbedder(nn.Module):
    """
    Style embeddings from paper Sec A.2.
    32 random style tokens index into a codebook of 64 learnable embeddings.
    """

    def __init__(self, hidden_size: int, num_tokens: int = 32, codebook_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random style embeddings."""
        # Random indices for each sample in the batch
        indices = torch.randint(
            0, self.codebook_size, (batch_size, self.num_tokens), device=device
        )
        embeddings = self.codebook(indices)  # (B, num_tokens, D)
        # Sum over tokens
        style = embeddings.sum(dim=1)  # (B, D)
        return style


class DriftDiT(nn.Module):
    """
    DiT-style generator for Drifting Models.

    Input: Gaussian noise epsilon ~ N(0, I), shape (B, C, 32, 32)
    Additional inputs: class label c, CFG scale alpha
    Output: generated image x, same shape as input
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        label_dropout: float = 0.1,
        num_register_tokens: int = 8,
        use_style_embed: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # Register (in-context) tokens
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )

        # RoPE for positional encoding
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 64,
        )

        # Conditioning embeddings
        self.label_embed = LabelEmbedder(num_classes, hidden_size, label_dropout)
        self.alpha_embed = AlphaEmbedder(hidden_size)
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qk_norm=True,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with specific strategy for adaLN-Zero."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        # Zero-init adaLN modulation layers (critical for training stability)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-init final layer adaLN modulation only
        # NOTE: For drifting models (one-step generator), we keep the final linear
        # layer with small random weights so the model outputs non-zero images initially
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # Use small initialization for final linear (not zero!)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to image.
        x: (B, N, patch_size^2 * C)
        Returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input noise, shape (B, C, H, W)
            labels: Class labels, shape (B,)
            alpha: CFG scale, shape (B,)
            force_drop_ids: Force label dropout for specific samples

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add register tokens
        register = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([register, x], dim=1)  # (B, num_reg + N, D)

        # Get RoPE embeddings
        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Conditioning
        c = self.label_embed(labels, self.training, force_drop_ids)
        c = c + self.alpha_embed(alpha)
        if self.use_style_embed:
            c = c + self.style_embed(B, device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        # Remove register tokens
        x = x[:, self.num_register_tokens:, :]

        # Final layer and unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        Runs two forward passes: conditional and unconditional.

        Args:
            x: Input noise, shape (B, C, H, W)
            labels: Class labels, shape (B,)
            alpha: CFG strength (scalar). Same quantity as ``gamma`` in the paper;
                the kwarg name is kept as ``alpha`` for backward compatibility.

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Create alpha tensor
        alpha_tensor = torch.full((B,), alpha, device=device, dtype=x.dtype)

        # Duplicate inputs for conditional and unconditional
        x_combined = torch.cat([x, x], dim=0)
        labels_combined = torch.cat([labels, labels], dim=0)
        alpha_combined = torch.cat([alpha_tensor, alpha_tensor], dim=0)

        # Force unconditional for second half
        force_drop = torch.cat([
            torch.zeros(B, device=device),
            torch.ones(B, device=device),
        ]).bool()

        # Forward pass
        out = self.forward(x_combined, labels_combined, alpha_combined, force_drop)

        # Split and apply CFG
        cond, uncond = out.chunk(2, dim=0)
        return uncond + alpha * (cond - uncond)


def DriftDiT_Small(img_size=32, in_channels=3, num_classes=10, **kwargs):
    """DriftDiT-Small: depth=8, hidden_dim=384, heads=6, ~26.6M params."""
    return DriftDiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


# Model registry
DriftDiT_models = {
    "DriftDiT-Small": DriftDiT_Small,
}
