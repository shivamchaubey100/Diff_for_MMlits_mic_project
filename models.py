# models.py
# Final stable Enhanced U-Net — skip connections aligned, cond padding correct,
# ready for training + inference.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# Sinusoidal positional emb
# -------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if not torch.is_floating_point(t):
            t = t.float()
        device = t.device
        half = self.dim // 2
        freq_exp = math.log(10000) / (half - 1)
        freqs = torch.exp(-freq_exp * torch.arange(half, device=device))
        emb = t[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# -------------------------
# ResBlock
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, max(1, in_ch))
        self.norm2 = nn.GroupNorm(8, max(1, out_ch))
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        t_out = self.time_mlp(t_emb)
        h = h + t_out[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual(x)


# -------------------------
# Attention block
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)

        q, k, v = rearrange(qkv,
                            'b (t h d) x y -> t b h (x y) d',
                            t=3, h=self.num_heads)
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out,
                        'b h (x y) d -> b (h d) x y',
                        x=H, y=W)
        return x + self.proj(out)


# -------------------------
# Enhanced UNet (FINAL)
# -------------------------
class EnhancedUNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        cond_ch=2,
        base_ch=48,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        dropout=0.1,
        use_attention=True
    ):
        super().__init__()

        self.in_ch = in_ch
        self.cond_ch = cond_ch
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks

        # time
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # first conv
        self.input_conv = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)

        # encoder
        in_channels = base_ch
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for mult in self.ch_mult:
            out_channels = base_ch * mult
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResBlock(in_channels if i == 0 else out_channels,
                                       out_channels,
                                       time_dim,
                                       dropout))
            self.encoder_levels.append(blocks)
            in_channels = out_channels

            if mult != self.ch_mult[-1]:
                self.downsamples.append(
                    nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
                )

        # middle
        self.mid_blocks = nn.ModuleList([
            ResBlock(in_channels, in_channels, time_dim, dropout),
            AttentionBlock(in_channels) if use_attention else nn.Identity(),
            ResBlock(in_channels, in_channels, time_dim, dropout)
        ])

        # decoder
        rev_ch = list(reversed(self.ch_mult))
        self.upsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        cur_ch = in_channels
        for i, mult in enumerate(rev_ch):
            out_ch = base_ch * mult

            if i != 0:
                self.upsamples.append(nn.ConvTranspose2d(cur_ch, cur_ch, 4, 2, 1))

            blocks = nn.ModuleList()
            blocks.append(ResBlock(cur_ch + out_ch, out_ch, time_dim, dropout))
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(out_ch, out_ch, time_dim, dropout))

            self.decoder_levels.append(blocks)
            cur_ch = out_ch

        # output
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x, cond, t, drop_cond=False):

        # drop-cond
        if drop_cond:
            cond = torch.zeros_like(cond)

        # pad cond channels if needed (inference without mask)
        if cond.shape[1] < self.cond_ch:
            pad = torch.zeros(cond.size(0),
                              self.cond_ch - cond.shape[1],
                              cond.size(2),
                              cond.size(3),
                              device=cond.device)
            cond = torch.cat([cond, pad], dim=1)

        # time embedding
        if not torch.is_floating_point(t):
            t = t.float()
        t_emb = self.time_mlp(t)

        # input
        h = torch.cat([x, cond], dim=1)
        h = self.input_conv(h)

        # ENCODER
        skips = []
        for idx, blocks in enumerate(self.encoder_levels):
            for b in blocks:
                h = b(h, t_emb)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        # MIDDLE
        for blk in self.mid_blocks:
            h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)

        # DECODER
        skip_idx = len(skips) - 1
        up_idx = 0

        for lvl_idx, blocks in enumerate(self.decoder_levels):

            if lvl_idx != 0:
                h = self.upsamples[up_idx](h)
                up_idx += 1

            skip = skips[skip_idx]
            skip_idx -= 1

            # fix spatial mismatch
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")

            h = torch.cat([h, skip], dim=1)

            for b in blocks:
                h = b(h, t_emb)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
