"""Mage-Flow sampling: FlowMatch Euler with static shift 6.0 (scheduler_config.json)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

SHIFT = 6.0
LATENT_CHANNELS = 128
DOWNSAMPLE = 16


def time_shift(s, shift: float = SHIFT):
    """Static z-image / FlowMatchEulerDiscreteScheduler time shift: shift*s / (1 + (shift - 1)*s)."""
    return shift * s / (1.0 + (shift - 1.0) * s)


def sigma_schedule(steps: int, sigmas=None, shift: float = SHIFT) -> torch.Tensor:
    """Shifted sigmas for `steps` steps with terminal 0 appended (length steps + 1)."""
    if sigmas is None:
        s = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
        s = time_shift(s, float(shift))
    else:
        s = np.asarray(sigmas, dtype=np.float64)
    return torch.tensor(np.append(s, 0.0), dtype=torch.float32)


def latent_hw(height: int, width: int) -> tuple[int, int]:
    return math.ceil(height / DOWNSAMPLE), math.ceil(width / DOWNSAMPLE)


def pack(latents: torch.Tensor) -> torch.Tensor:
    """[B, 128, h, w] -> [B, h*w, 128]."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2).contiguous()


def unpack(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B, h*w, 128] -> [B, 128, h, w]."""
    b, _, c = tokens.shape
    return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


def initial_noise(seed: int, height: int, width: int, generator_device: str = "cpu") -> torch.Tensor:
    """Standard Gaussian starting noise [1, 128, h, w] float32 (no watermark)."""
    h, w = latent_hw(height, width)
    gen = torch.Generator(generator_device).manual_seed(int(seed))
    return torch.randn(1, LATENT_CHANNELS, h, w, generator=gen, dtype=torch.float32)


@torch.no_grad()
def sample(
    dit,
    text_emb: torch.Tensor,
    height: int,
    width: int,
    *,
    steps: int = 30,
    seed: int = 0,
    sigmas=None,
    shift: float = SHIFT,
    cfg: float = 5.0,
    neg_emb: Optional[torch.Tensor] = None,
    device="cuda",
    dtype: torch.dtype = torch.bfloat16,
    generator_device: str = "cpu",
    noise: Optional[torch.Tensor] = None,
    on_step=None,
    text_mask: Optional[torch.Tensor] = None,
    neg_mask: Optional[torch.Tensor] = None,
    ref_latents=None,
    renormalization: bool = False,
) -> torch.Tensor:
    """Denoise one image; returns [1, 128, h, w] float32 latents."""
    h, w = latent_hw(height, width)
    n = h * w
    if noise is None:
        noise = initial_noise(seed, height, width, generator_device)
    x = pack(noise.float().to(device))
    sched = sigma_schedule(steps, sigmas=sigmas, shift=shift).to(device)

    enc = text_emb[None].to(device)
    enc_mask = None if text_mask is None else text_mask.reshape(1, -1).to(device)
    use_cfg = cfg > 1.0 and neg_emb is not None
    if use_cfg:
        nenc = neg_emb[None].to(device)
        nenc_mask = None if neg_mask is None else neg_mask.reshape(1, -1).to(device)

    refs = [r.float().to(device) for r in (ref_latents or [])]
    ref_tokens = torch.cat([pack(r) for r in refs], dim=1) if refs else None
    shapes = [[(1, h, w)] + [(1, r.shape[-2], r.shape[-1]) for r in refs]]

    for i in range(len(sched) - 1):
        if on_step is not None:
            on_step(i, len(sched) - 1)
        s, s_next = sched[i], sched[i + 1]
        t = s.view(1).to(dtype)
        xin = x if ref_tokens is None else torch.cat([x, ref_tokens], dim=1)
        v = dit(xin.to(dtype), enc.to(dtype), t, shapes, enc_mask)[:, :n].float()
        if use_cfg:
            vn = dit(xin.to(dtype), nenc.to(dtype), t, shapes, nenc_mask)[:, :n].float()
            comb = vn + cfg * (v - vn)
            if renormalization:
                comb = comb * (torch.norm(v, dim=-1, keepdim=True) / (torch.norm(comb, dim=-1, keepdim=True) + 1e-6))
            v = comb
        x = x + (s_next - s) * v
    return unpack(x, h, w)


@torch.no_grad()
def encode_image(vae, rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] float in [-1, 1] -> [B, 128, H/16, W/16]."""
    p = next(vae.parameters())
    return vae.encode(rgb.to(p.device, p.dtype))


@torch.no_grad()
def decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """[B, 128, h, w] -> [B, 3, H, W] float in [-1, 1]."""
    p = next(vae.parameters())
    return vae.decode(latents.to(p.device, p.dtype))
