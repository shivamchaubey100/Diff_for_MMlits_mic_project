# models.py
#
# Enhanced U-Net for conditional diffusion.
#
# Fixes vs previous version
# --------------------------
#   FIX (bug #4): num_heads in _attn() is now computed dynamically via
#                 _safe_num_heads() instead of being hardcoded to 4.
#                 AttentionBlock asserts channels % num_heads == 0, so
#                 hardcoding 4 silently fails for some base_ch values.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a safe group count for any channel width."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def _safe_num_heads(channels: int, preferred: int = 4) -> int:
    """
    Largest divisor of channels that is <= preferred.

    FIX (bug #4): prevents AttentionBlock assertion failure when
    channels is not divisible by the hardcoded preferred value.
    """
    for h in range(preferred, 0, -1):
        if channels % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(t):
            t = t.float()
        device = t.device
        half   = max(self.dim // 2, 1)
        freq_exp = math.log(10000) / (half - 1) if half > 1 else 0.0
        freqs    = torch.exp(-freq_exp * torch.arange(half, device=device))
        emb      = t[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1   = make_norm(in_ch)
        self.norm2   = make_norm(out_ch)
        self.act     = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.residual = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.residual(x)


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.norm      = make_norm(channels)
        self.qkv       = nn.Conv2d(channels, channels * 3, 1)
        self.proj      = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)

        q, k, v = rearrange(
            qkv, "b (t h d) x y -> t b h (x y) d",
            t=3, h=self.num_heads,
        )
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.softmax(dim=-1)
        out   = attn @ v
        out   = rearrange(out, "b h (x y) d -> b (h d) x y", x=H, y=W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Unified block wrapper
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    """Uniform (x, t_emb) call signature for ResBlock and AttentionBlock."""
    def __init__(self, block: nn.Module, needs_t: bool):
        super().__init__()
        self.block   = block
        self.needs_t = needs_t

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.block(x, t_emb) if self.needs_t else self.block(x)


def _res(in_ch, out_ch, time_dim, dropout):
    return _Block(ResBlock(in_ch, out_ch, time_dim, dropout), needs_t=True)

def _attn(channels, preferred_heads: int = 4):
    # FIX (bug #4): compute safe num_heads dynamically
    num_heads = _safe_num_heads(channels, preferred_heads)
    return _Block(AttentionBlock(channels, num_heads), needs_t=False)

def _identity():
    return _Block(nn.Identity(), needs_t=False)


# ---------------------------------------------------------------------------
# Enhanced UNet
# ---------------------------------------------------------------------------

class EnhancedUNet(nn.Module):
    """
    Conditional denoising U-Net for diffusion models.

    Args:
        in_ch          : channels in the noisy input (1 for grayscale CT)
        cond_ch        : channels in the conditioning tensor
                         (2 = healthy CT + liver mask)
        base_ch        : base channel width (multiples of 8 recommended)
        ch_mult        : channel multipliers per encoder level
        num_res_blocks : ResBlocks per level
        time_dim       : sinusoidal embedding + MLP output dimension
        dropout        : dropout probability in ResBlocks
        use_attention  : whether to place self-attention in the bottleneck
    """

    def __init__(
        self,
        in_ch:          int   = 1,
        cond_ch:        int   = 2,
        base_ch:        int   = 48,
        ch_mult:        tuple = (1, 2, 4),
        num_res_blocks: int   = 2,
        time_dim:       int   = 256,
        dropout:        float = 0.1,
        use_attention:  bool  = True,
    ):
        super().__init__()

        self.in_ch          = in_ch
        self.cond_ch        = cond_ch
        self.base_ch        = base_ch
        self.ch_mult        = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks

        # ── time embedding ──────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── initial convolution ─────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)

        # ── encoder ────────────────────────────────────────────────────────
        in_channels         = base_ch
        self.encoder_levels = nn.ModuleList()
        self.downsamples    = nn.ModuleList()

        for mult in self.ch_mult:
            out_channels = base_ch * mult
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                ch_in = in_channels if i == 0 else out_channels
                blocks.append(_res(ch_in, out_channels, time_dim, dropout))
            self.encoder_levels.append(blocks)
            in_channels = out_channels

            if mult != self.ch_mult[-1]:
                self.downsamples.append(
                    nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
                )

        # ── bottleneck ──────────────────────────────────────────────────────
        self.mid_blocks = nn.ModuleList([
            _res(in_channels, in_channels, time_dim, dropout),
            _attn(in_channels) if use_attention else _identity(),
            _res(in_channels, in_channels, time_dim, dropout),
        ])

        # ── decoder ─────────────────────────────────────────────────────────
        rev_mult            = list(reversed(self.ch_mult))
        self.upsamples      = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        cur_ch = in_channels

        for i, mult in enumerate(rev_mult):
            skip_ch = base_ch * mult
            out_ch  = base_ch * mult

            if i != 0:
                self.upsamples.append(
                    nn.ConvTranspose2d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1)
                )

            blocks = nn.ModuleList()
            blocks.append(_res(cur_ch + skip_ch, out_ch, time_dim, dropout))
            for _ in range(num_res_blocks):
                blocks.append(_res(out_ch, out_ch, time_dim, dropout))

            self.decoder_levels.append(blocks)
            cur_ch = out_ch

        # ── output head ─────────────────────────────────────────────────────
        self.out_norm = make_norm(base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x:         torch.Tensor,
        cond:      torch.Tensor,
        t:         torch.Tensor,
        drop_cond: bool = False,
    ) -> torch.Tensor:

        if drop_cond:
            cond = torch.zeros_like(cond)

        if cond.shape[1] < self.cond_ch:
            pad = torch.zeros(
                cond.size(0), self.cond_ch - cond.shape[1],
                cond.size(2), cond.size(3),
                device=cond.device, dtype=cond.dtype,
            )
            cond = torch.cat([cond, pad], dim=1)

        if not torch.is_floating_point(t):
            t = t.float()
        t_emb = self.time_mlp(t)

        h = self.input_conv(torch.cat([x, cond], dim=1))

        # Encoder
        skips = []
        for idx, blocks in enumerate(self.encoder_levels):
            for blk in blocks:
                h = blk(h, t_emb)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        # Bottleneck
        for blk in self.mid_blocks:
            h = blk(h, t_emb)

        # Decoder
        skip_idx = len(skips) - 1
        up_idx   = 0

        for lvl_idx, blocks in enumerate(self.decoder_levels):
            if lvl_idx != 0:
                h       = self.upsamples[up_idx](h)
                up_idx += 1

            skip = skips[skip_idx]
            skip_idx -= 1

            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")

            h = torch.cat([h, skip], dim=1)

            for blk in blocks:
                h = blk(h, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)