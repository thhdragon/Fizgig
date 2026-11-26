# Ported from Microsoft Mage-Flow (https://github.com/microsoft/Mage, MIT License) via
# src/fizgig/mage_flow/source_reference.temp/models/modules/mage_vae.py.
# Changes for Fizgig: defaults sample_posterior=False (matching vae/config.json and ComfyUI's MageVAE),
# adds load_mage_vae() supporting single .safetensors files and diffusers vae/ directories.
"""MageVAE: symmetric one-step DConvEncoder + DConvDenoiser (CoD Decoder), 128 channels at 16x downsample."""
from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        b, c = x.shape[:2]
        return x * (1 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class _EncoderLayerNorm2d(LayerNorm2d):
    pass


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(in_dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, pca_dim: int = 128, embed_dim: int = 384, bias: bool = True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.proj2(torch.cat([self.proj1(x), cond], dim=1))


class DiCoBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)
        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, inp: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(inp), shift_msa, scale_msa)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + gate_msa[..., None, None] * x
        return x + gate_mlp[..., None, None] * self.conv5(
            F.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        )


class _EncoderDiCoBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)
        self.norm1 = _EncoderLayerNorm2d(hidden_size)
        self.norm2 = _EncoderLayerNorm2d(hidden_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + x
        return x + self.conv5(F.gelu(self.conv4(self.norm2(x))))


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True))

    @lru_cache
    def fetch_pos(self, patch_size: int, device, dtype):
        pos = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        fx = freqs[None, :, None]
        fy = freqs[None, None, :]
        coeffs = (1 + fx * fy) ** -1
        dct_x = torch.cos(pos_x * fx * torch.pi)
        dct_y = torch.cos(pos_y * fy * torch.pi)
        return (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p2, _ = x.shape
        ps = int(p2 ** 0.5)
        dct = self.fetch_pos(ps, x.device, x.dtype).expand(b, -1, -1)
        return self.embedder(torch.cat([x, dct], dim=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class _MLPResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels: int, model_channels: int, out_channels: int, z_channels: int,
                 num_res_blocks: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size ** 2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(_MLPResBlock(model_channels) for _ in range(num_res_blocks))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_embed(c).reshape(c.shape[0], self.patch_size ** 2, -1)
        for block in self.res_blocks:
            x = block(x, c)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        d = self.patch_size
        b, c, h, w = q.shape
        pad_h = (d - h % d) % d
        pad_w = (d - w % d) % d
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode="replicate")
            k = F.pad(k, (0, pad_w, 0, pad_h), mode="replicate")
            v = F.pad(v, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, h_pad, w_pad = q.shape
        nph, npw = h_pad // d, w_pad // d
        np_ = nph * npw

        def to_patches(t):
            return t.reshape(b, c, nph, d, npw, d).permute(0, 2, 4, 1, 3, 5).reshape(b * np_, c, d * d)

        q, k, v = to_patches(q), to_patches(k), to_patches(v)
        w_ = F.softmax(torch.bmm(q.permute(0, 2, 1), k) * (c ** -0.5), dim=2).permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, nph, npw, c, d, d).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h_pad, w_pad)
        if pad_h or pad_w:
            h_ = h_[:, :, :h, :w]
        return x + self.proj_out(h_)


class _ConstAdaLN(nn.Module):
    def __init__(self, modulation: torch.Tensor):
        super().__init__()
        self.register_buffer("modulation", modulation.detach().clone())

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        b = c.shape[0]
        if self.modulation.shape[0] != b:
            return self.modulation.expand(b, *self.modulation.shape[1:])
        return self.modulation


def _replace_adaln_with_const(module: nn.Module, c: torch.Tensor) -> int:
    n = 0
    for child in module.modules():
        if not isinstance(child, DiCoBlock):
            continue
        if isinstance(child.adaLN_modulation, _ConstAdaLN):
            continue
        with torch.no_grad():
            mod = child.adaLN_modulation(c)
        child.adaLN_modulation = _ConstAdaLN(mod)
        n += 1
    return n


class _Decoder(nn.Module):
    def __init__(self, out_ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ada = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.block(self.conv_in(z))
        return self.ada(self.conv_out(nonlinearity(self.norm_out(h))))


class _DConvEncoder(nn.Module):
    def __init__(self, z_ch: int = 128, hidden_size: int = 384, num_blocks: int = 21, patch_size: int = 16,
                 mlp_ratio: float = 4.0, head_size: int = 768, num_head_blocks: int = 2, out_ch_mult: int = 2):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.head_blocks = nn.ModuleList([_EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)])
        self.proj_down = nn.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)
        self.z_proj = nn.Conv2d(z_ch, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, z_ch * out_ch_mult, kernel_size=1, bias=True)

    def forward_pred(self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.patch_cond_embed(y)
        for block in self.head_blocks:
            cond = block(cond)
        cond = self.proj_down(cond)
        s = self.fuse_proj(torch.cat([cond, self.z_proj(z_t)], dim=1))
        c = self.t_embedder(t.view(-1))
        for block in self.blocks:
            s = block(s, c)
        return self.proj_out(self.norm_out(s))


class _YEmbedder(nn.Module):
    def __init__(self, ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.decoder = _Decoder(out_ch=ch, z_ch=z_ch)


class _DConvDenoiser(nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3, hidden_size: int = 384,
                 hidden_size_x: int = 32, mlp_ratio: float = 4.0, num_blocks: int = 24,
                 num_cond_blocks: int = 21, bottleneck_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x * patch_size ** 2, 1, 1, 0)
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList([DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)])
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = _YEmbedder(ch=hidden_size, z_ch=bottleneck_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        c = self.t_embedder(t.view(-1))
        s = self.s_embedder(x, cond)
        for block in self.blocks:
            s = block(s, c)
        length = s.shape[-2] * s.shape[-1]
        s = s.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = torch.cat([x, self.y_embedder_x(cond).flatten(2)], dim=1)
        x = x.reshape(b, -1, self.patch_size ** 2, length).permute(0, 3, 2, 1).flatten(0, 1)
        x = self.x_embedder(x)
        x = self.dec_net(x, s)
        x = self.final_layer(x)
        x = x.transpose(1, 2).reshape(b, length, -1)
        return F.fold(x.transpose(1, 2).contiguous(), (h, w), kernel_size=self.patch_size, stride=self.patch_size)


def _resolve_vae_path(path: str) -> str:
    if os.path.isdir(path):
        for cand in (
            os.path.join(path, "diffusion_pytorch_model.safetensors"),
            os.path.join(path, "vae", "diffusion_pytorch_model.safetensors"),
        ):
            if os.path.isfile(cand):
                return cand
    return path


def _load_state_dict(ckpt_path: str) -> dict:
    ckpt_path = _resolve_vae_path(ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(ckpt_path, device="cpu")
    if os.path.exists(os.path.join(ckpt_path, "checkpoint-state_dict.pt")):
        ckpt_path = os.path.join(ckpt_path, "checkpoint-state_dict.pt")
    elif os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "checkpoint", "mp_rank_00_model_states.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    if "module" in state:
        return state["module"]
    if "state_dict" in state:
        return state["state_dict"]
    return state


class MageVAE(nn.Module):
    latent_channels = 128
    downsample_factor = 16

    def __init__(self, ckpt_path: str, sample_posterior: bool = False):
        super().__init__()
        self.sample_posterior = sample_posterior
        self.dconv_encoder = _DConvEncoder()
        self.decoder_model = _DConvDenoiser()

        sd = _load_state_dict(ckpt_path)
        self._load_encoder(sd, ckpt_path)
        self._load_decoder(sd, ckpt_path)
        self._freeze_adaln_cache()

    def _load_encoder(self, sd: dict, ckpt_path: str):
        prefix = "student.dconv_encoder."
        enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not enc_sd:
            raise RuntimeError(f"MageVAE: no '{prefix}*' keys in {ckpt_path}")
        proj = enc_sd.get("proj_out.weight")
        if proj is None or proj.shape[0] != 2 * self.latent_channels:
            raise RuntimeError(f"MageVAE: expected proj_out out_channels={2 * self.latent_channels}, "
                               f"got {None if proj is None else tuple(proj.shape)}")
        missing, unexpected = self.dconv_encoder.load_state_dict(enc_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MageVAE encoder mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")

    def _load_decoder(self, sd: dict, ckpt_path: str):
        prefix = "pipeline."
        if not any(k.startswith(prefix) for k in sd):
            raise RuntimeError(f"MageVAE: no '{prefix}*' keys in {ckpt_path}")
        model_dict = self.decoder_model.state_dict()
        matched = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix):]
            if new_k.startswith("y_embedder.encoder.") or new_k.startswith("y_embedder.bottleneck."):
                continue
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                matched[new_k] = v
        if not matched:
            raise RuntimeError(f"MageVAE decoder: 0 params matched from {ckpt_path}")
        self.decoder_model.load_state_dict(matched, strict=False)

    def _freeze_adaln_cache(self):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        t = torch.zeros(1, device=device, dtype=dtype)
        _replace_adaln_with_const(self.dconv_encoder, self.dconv_encoder.t_embedder(t))
        _replace_adaln_with_const(self.decoder_model, self.decoder_model.t_embedder(t))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ps = self.dconv_encoder.patch_size
        b, _, h, w = x.shape
        if h % ps or w % ps:
            raise ValueError(f"H, W must be multiples of {ps}, got ({h}, {w})")
        z_t = torch.zeros(b, self.dconv_encoder.z_ch, h // ps, w // ps, device=x.device, dtype=x.dtype)
        t = torch.zeros(b, device=x.device, dtype=x.dtype)
        out = self.dconv_encoder.forward_pred(z_t, t, x)
        mean = out[:, :self.latent_channels]
        if self.sample_posterior:
            logvar = out[:, self.latent_channels:].clamp(min=-20.0, max=10.0)
            return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return mean

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        cond = self.decoder_model.y_embedder.decoder(z)
        b = z.shape[0]
        h = z.shape[2] * self.downsample_factor
        w = z.shape[3] * self.downsample_factor
        noise = torch.zeros(b, 3, h, w, device=z.device, dtype=z.dtype)
        t = torch.zeros(b, device=z.device, dtype=z.dtype)
        return self.decoder_model.forward(noise, t, cond)


def load_mage_vae(path: str, device="cuda", dtype=torch.bfloat16, sample_posterior: bool = False) -> MageVAE:
    return MageVAE(path, sample_posterior=sample_posterior).to(device, dtype).eval().requires_grad_(False)
