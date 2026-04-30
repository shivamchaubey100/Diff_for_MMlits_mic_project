# models.py
# Enhanced U-Net — skip connections aligned, cond padding correct.
# Fixes vs previous version:
#   1. GroupNorm group count is now computed dynamically so any base_ch works.
#   2. Mid-block dispatch uses a typed wrapper instead of isinstance checks,
#      making it safe to add new block types in future.
#   3. Minor: SinusoidalPosEmb half-dim edge-case guarded.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a safe group count.

    GroupNorm requires channels % num_groups == 0.  Using a fixed value of 8
    breaks whenever base_ch is not a multiple of 8.  We pick the largest
    power-of-two divisor up to 32, which works for any channel count >= 1.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)   # fallback (should never reach here)


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
        # guard against half == 1 which would make arange(1) and log degenerate
        if half > 1:
            freq_exp = math.log(10000) / (half - 1)
        else:
            freq_exp = 0.0
        freqs = torch.exp(-freq_exp * torch.arange(half, device=device))
        emb   = t[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)   # (B, dim)


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
# Unified block wrapper — avoids isinstance dispatch in the forward pass
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    """Thin wrapper that gives every block a uniform (x, t_emb) call signature.

    ResBlock needs t_emb; AttentionBlock does not.  Rather than scattering
    isinstance checks through the UNet forward, we wrap each block here.
    """
    def __init__(self, block: nn.Module, needs_t: bool):
        super().__init__()
        self.block   = block
        self.needs_t = needs_t

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.block(x, t_emb) if self.needs_t else self.block(x)


def _res(in_ch, out_ch, time_dim, dropout):
    return _Block(ResBlock(in_ch, out_ch, time_dim, dropout), needs_t=True)

def _attn(channels, num_heads=4):
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
        in_ch        : channels in the noisy input (e.g. 1 for grayscale CT)
        cond_ch      : channels in the conditioning tensor
                       (e.g. 2 = healthy CT + liver mask)
        base_ch      : base channel width; must be divisible by the largest
                       group count chosen by make_norm (multiples of 8 are safe)
        ch_mult      : channel multipliers per encoder level
        num_res_blocks: ResBlocks per level (excluding the extra skip-consuming
                        block at the top of each decoder level)
        time_dim     : sinusoidal embedding + MLP output dimension
        dropout      : dropout probability in ResBlocks
        use_attention: whether to place a self-attention block in the bottleneck
    """

    def __init__(
        self,
        in_ch:         int   = 1,
        cond_ch:       int   = 2,
        base_ch:       int   = 48,
        ch_mult:       tuple = (1, 2, 4),
        num_res_blocks: int  = 2,
        time_dim:      int   = 256,
        dropout:       float = 0.1,
        use_attention: bool  = True,
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
        in_channels          = base_ch
        self.encoder_levels  = nn.ModuleList()
        self.downsamples     = nn.ModuleList()

        for mult in self.ch_mult:
            out_channels = base_ch * mult
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                ch_in = in_channels if i == 0 else out_channels
                blocks.append(_res(ch_in, out_channels, time_dim, dropout))
            self.encoder_levels.append(blocks)
            in_channels = out_channels

            if mult != self.ch_mult[-1]:           # no downsample after last level
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
        # For each decoder level we:
        #   1. Upsample (except the very first level which is at bottleneck res)
        #   2. Concatenate the matching encoder skip
        #   3. Run (1 + num_res_blocks) ResBlocks
        #
        # Skip channel sizes (from encoder, in reverse order):
        #   level 0 (top of decoder) receives skip from last encoder level  → base_ch * ch_mult[-1]
        #   level 1                                                          → base_ch * ch_mult[-2]
        #   …
        rev_mult             = list(reversed(self.ch_mult))
        self.upsamples       = nn.ModuleList()
        self.decoder_levels  = nn.ModuleList()

        cur_ch = in_channels          # channels currently in h (= bottleneck channels)

        for i, mult in enumerate(rev_mult):
            skip_ch = base_ch * mult  # channels of the corresponding encoder skip
            out_ch  = base_ch * mult

            if i != 0:
                # ConvTranspose2d: in=cur_ch → out=cur_ch (channel count unchanged)
                self.upsamples.append(
                    nn.ConvTranspose2d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1)
                )

            blocks = nn.ModuleList()
            # first block consumes concatenated (cur_ch + skip_ch)
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
        """
        Args:
            x        : noisy image,   (B, in_ch, H, W)
            cond     : conditioning,  (B, cond_ch, H, W)
            t        : timesteps,     (B,)   integer or float
            drop_cond: if True, zero out the conditioning (CFG training)

        Returns:
            predicted v (or eps) with the same shape as x
        """

        # ── classifier-free guidance conditioning dropout ───────────────────
        if drop_cond:
            cond = torch.zeros_like(cond)

        # ── pad cond channels if fewer than expected (e.g. inference w/o mask) ─
        if cond.shape[1] < self.cond_ch:
            pad  = torch.zeros(
                cond.size(0),
                self.cond_ch - cond.shape[1],
                cond.size(2),
                cond.size(3),
                device=cond.device,
                dtype=cond.dtype,
            )
            cond = torch.cat([cond, pad], dim=1)

        # ── time embedding ──────────────────────────────────────────────────
        if not torch.is_floating_point(t):
            t = t.float()
        t_emb = self.time_mlp(t)

        # ── input projection ────────────────────────────────────────────────
        h = self.input_conv(torch.cat([x, cond], dim=1))

        # ── encoder ────────────────────────────────────────────────────────
        skips = []
        for idx, blocks in enumerate(self.encoder_levels):
            for blk in blocks:
                h = blk(h, t_emb)
            skips.append(h)                        # save skip before downsampling
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        # ── bottleneck ──────────────────────────────────────────────────────
        for blk in self.mid_blocks:
            h = blk(h, t_emb)

        # ── decoder ─────────────────────────────────────────────────────────
        skip_idx = len(skips) - 1
        up_idx   = 0

        for lvl_idx, blocks in enumerate(self.decoder_levels):

            if lvl_idx != 0:
                h       = self.upsamples[up_idx](h)
                up_idx += 1

            skip = skips[skip_idx]
            skip_idx -= 1

            # fix any spatial mismatch caused by odd input dimensions
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")

            h = torch.cat([h, skip], dim=1)

            for blk in blocks:
                h = blk(h, t_emb)

        # ── output ──────────────────────────────────────────────────────────
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)