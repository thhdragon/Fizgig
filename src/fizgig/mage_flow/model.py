# Ported from Microsoft Mage-Flow (https://github.com/microsoft/Mage, MIT License) via
# src/fizgig/mage_flow/source_reference.temp/models/{mage_flow.py,modules/mage_layers.py}.
# Changes for Fizgig: pure PyTorch modules (no diffusers mixins), batched SDPA attention with optional padding /
# prompt-travel masks and comfy-kitchen INT8 attention hook, attn_probe trigger-word hooks, non-reentrant gradient
# checkpointing, ModelOffloader block swap, and single-file / ComfyUI int8 ConvRot weight loading.
"""Mage-Flow 4B double-stream MMDiT (12 blocks, patch_size=1, 128 latent channels)."""
from __future__ import annotations

import glob
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from fizgig.training import attn_probe

MAGE_FLOW_CONFIG = dict(
    in_channels=128,
    out_channels=128,
    context_in_dim=2560,
    hidden_size=3072,
    num_heads=24,
    depth=12,
    axes_dim=(16, 56, 56),
    patch_size=1,
    mlp_ratio=4.0,
)

_MASKED_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def apply_rotary_emb_mageflow(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary embeddings to `x` ([B, S, H, D]) using `freqs_cis` ([S, D/2] or [B, S, D/2])."""
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(-2 if freqs_cis.dim() == 3 else 1)
    x_out = torch.view_as_real(x_rotated * freqs).flatten(-2)
    return x_out.type_as(x)


class RMSNorm(nn.Module):
    """Elementwise-affine RMSNorm matching diffusers.models.normalization.RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x.to(input_dtype)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings with the reference's bf16 frequency-table rounding."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timestep).to(dtype=hidden_states.dtype))


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "tanh", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate=self.approximate)


class FeedForward(nn.Module):
    """Matches diffusers FeedForward(dim, dim_out, activation_fn='gelu-approximate') module paths (net.0.proj, net.2)."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            GELU(dim, inner_dim, approximate="tanh", bias=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.net:
            x = mod(x)
        return x


class MageFlowEmbedRope(nn.Module):
    """2D multi-scale RoPE for image tokens (text tokens are not rotated)."""

    def __init__(self, theta: int = 10000, axes_dim=(16, 56, 56), scale_rope: bool = True):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [self.rope_params(pos_index, d, theta) for d in self.axes_dim], dim=1
        )
        self.neg_freqs = torch.cat(
            [self.rope_params(neg_index, d, theta) for d in self.axes_dim], dim=1
        )
        self.scale_rope = scale_rope
        self.video_freq_cache = {}

    @staticmethod
    def rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, video_fhw, device: torch.device) -> torch.Tensor:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
        if isinstance(video_fhw, list) and video_fhw and isinstance(video_fhw[0], list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            key = (frame, height, width, idx)
            if key not in self.video_freq_cache:
                self.video_freq_cache[key] = self._compute_video_freqs(frame, height, width, idx)
            vid_freqs.append(self.video_freq_cache[key].to(device))
        return torch.cat(vid_freqs, dim=0)

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_frame = freqs_pos[0][idx:idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2):], freqs_pos[1][:height // 2]], dim=0
            ).view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2):], freqs_pos[2][:width // 2]], dim=0
            ).view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1).contiguous()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q, k, v: [B, H, S, D]. Routes through comfy-kitchen INT8 attention on opted-in unmasked workbench renders."""
    from fizgig.modules import int8_attention as _i8a
    out = _i8a.attend(q, k, v, mask)
    if out is not None:
        return out
    if mask is not None:
        with sdpa_kernel(_MASKED_BACKENDS, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k, v)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)

        self.add_k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.add_v_proj = nn.Linear(dim, inner_dim, bias=True)
        self.add_q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(inner_dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]

        img_q = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        img_k = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        img_v = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        txt_q = self.norm_added_q(self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
        txt_k = self.norm_added_k(self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
        txt_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

        img_q = apply_rotary_emb_mageflow(img_q, image_rotary_emb)
        img_k = apply_rotary_emb_mageflow(img_k, image_rotary_emb)

        # Joint attention order: [text, image]
        joint_q = torch.cat([txt_q, img_q], dim=1).transpose(1, 2)
        joint_k = torch.cat([txt_k, img_k], dim=1).transpose(1, 2)
        joint_v = torch.cat([txt_v, img_v], dim=1).transpose(1, 2)

        if attn_probe._ACTIVE is not None:
            attn_probe.see(joint_q, joint_k, attn_mask)

        out = _sdpa(joint_q, joint_k, joint_v, attn_mask).transpose(1, 2).flatten(2, 3).type_as(joint_q)
        txt_out = self.to_add_out(out[:, :seq_txt])
        img_out = self.to_out[1](self.to_out[0](out[:, seq_txt:seq_txt + seq_img]))
        return img_out, txt_out


class MageFlowTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0,
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio)

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio)

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn_out, txt_attn_out = self.attn(
            img_modulated, txt_modulated, image_rotary_emb=image_rotary_emb, attn_mask=attn_mask
        )
        hidden_states = hidden_states + img_gate1 * img_attn_out
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_out

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int,
                 elementwise_affine: bool = False, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MageFlowDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        context_in_dim: int = 2560,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 12,
        axes_dim=(16, 56, 56),
        patch_size: int = 1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = hidden_size
        self.axes_dim = list(axes_dim)
        self.num_attention_heads = num_heads
        self.attention_head_dim = self.inner_dim // self.num_attention_heads
        self.patch_size = patch_size
        assert sum(self.axes_dim) == self.attention_head_dim

        self.pos_embed = MageFlowEmbedRope(theta=10000, axes_dim=self.axes_dim, scale_rope=True)
        self.img_in = nn.Linear(self.in_channels, self.inner_dim, bias=True)
        self.txt_norm = RMSNorm(context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(context_in_dim, self.inner_dim, bias=True)
        self.time_text_embed = MageFlowTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            MageFlowTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = bool(on)

    # ---- block swap (Fizgig's shared offloader) ---------------------------------------------------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()
        n = len(self.transformer_blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("mage_flow", list(self.transformer_blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        blocks = self.transformer_blocks
        self.transformer_blocks = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.transformer_blocks = blocks
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.transformer_blocks))

    def disable_block_swap(self):
        if self.offloader is not None:
            self.offloader.remove_hooks()
        self.offloader = None
        self.blocks_to_swap = 0

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [B, N_img, C] (target first, optional reference latents appended).
        encoder_hidden_states: [B, L_txt, context_in_dim].
        timestep: [B] in [0, 1] (= sigma).
        img_shapes: [[(1, h, w), ...]] matching N_img.
        encoder_hidden_states_mask: optional [B, L_txt] bool padding mask or float [0..1] prompt-travel weights.
        """
        batch, n_img, _ = hidden_states.shape
        l_txt = encoder_hidden_states.shape[1]

        ms_pe = self.pos_embed(img_shapes, device=hidden_states.device)
        img = self.img_in(hidden_states)
        txt = self.txt_in(self.txt_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep.to(img.dtype), img)

        attn_mask = None
        valid_total = l_txt + n_img
        if encoder_hidden_states_mask is not None:
            m = encoder_hidden_states_mask.to(hidden_states.device)
            if m.dtype == torch.bool:
                if not bool(m.all()):
                    attn_mask = torch.cat([m, m.new_ones(batch, n_img)], dim=1)[:, None, None, :]
                    valid_total = int(m[0].sum().item()) + n_img if batch == 1 else l_txt + n_img
            else:
                w = torch.cat([m.float(), m.new_ones(batch, n_img, dtype=torch.float32)], dim=1)
                if not bool((w == 1.0).all()):
                    attn_mask = torch.log(w)[:, None, None, :].to(img.dtype)

        if attn_probe._ACTIVE is not None:
            # Joint attention sequence is [txt (0..l_txt), img (l_txt..l_txt+n_img)].
            target_len = math.prod(img_shapes[0][0]) if img_shapes else n_img
            attn_probe.layout((l_txt, l_txt + target_len), 0, l_txt + n_img, valid=valid_total)

        blocks = list(self.transformer_blocks) if self.blocks_to_swap else None
        for index, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if getattr(block, "_handles_checkpointing", False):
                txt, img = block(img, txt, temb, ms_pe, attn_mask)
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                txt, img = checkpoint(block, img, txt, temb, ms_pe, attn_mask, use_reentrant=False)
            else:
                txt, img = block(img, txt, temb, ms_pe, attn_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)

        img = self.norm_out(img, temb)
        return self.proj_out(img)


# ---- weight loading ---------------------------------------------------------------------------
def _resolve_dit_files(path: str) -> list[str]:
    if os.path.isdir(path):
        sub = os.path.join(path, "transformer")
        d = sub if os.path.isdir(sub) else path
        files = sorted(glob.glob(os.path.join(glob.escape(d), "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {d}")
        return files
    if path.endswith(".json"):
        return sorted(glob.glob(os.path.join(glob.escape(os.path.dirname(path)), "*.safetensors")))
    return [path]


def is_convrot(path: str) -> bool:
    """True for a ComfyUI int8 ConvRot checkpoint (`<module>.comfy_quant` markers)."""
    if not (isinstance(path, str) and os.path.isfile(path) and path.endswith(".safetensors")):
        return False
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return any(k.endswith(".comfy_quant") for k in f.keys())


def _normalize_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(".comfy_attention.config"):
            continue
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        out[k] = v
    return out


def _take_convrot(sd: dict) -> dict:
    from fizgig.minimax.convrot import parse_comfy_quant
    out = {}
    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        base = k[:-len(".comfy_quant")]
        conf = parse_comfy_quant(sd.pop(k))
        if conf.get("format") != "int8_tensorwise":
            raise ValueError(f"{base}: unsupported comfy quant format {conf.get('format')!r}")
        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
        q, sc = sd.pop(base + ".weight"), sd.pop(base + ".weight_scale").float().reshape(-1, 1)
        out[base] = (q, sc, rot)
    return out


def load_mage_flow_dit(path: str, device="cuda", dtype=torch.bfloat16, config=None, keep_convrot: bool = True) -> MageFlowDiT:
    """Load Mage-Flow DiT from a single .safetensors file (bf16 or ComfyUI int8 ConvRot) or diffusers dir."""
    from safetensors.torch import load_file
    cfg = dict(MAGE_FLOW_CONFIG)
    if config:
        cfg.update(config)
    with torch.device("meta"):
        model = MageFlowDiT(**cfg)

    sd = {}
    for f in _resolve_dit_files(path):
        sd.update(load_file(f, device="cpu"))
    sd = _normalize_keys(sd)
    rot = _take_convrot(sd)
    sd = {k: v.to(dtype) for k, v in sd.items()}

    if rot and not keep_convrot:
        from fizgig.minimax.convrot import dequantize_int8_convrot
        for name, (q, sc, r) in rot.items():
            sd[name + ".weight"] = dequantize_int8_convrot(
                q, sc, {"format": "int8_tensorwise", "convrot": r > 1, "convrot_groupsize": r}, out_dtype=dtype
            )
        rot = {}

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if rot:
        from fizgig.minimax.convrot import ConvRotInt8Linear
        for name, (q, sc, r) in rot.items():
            parent, attr = name.rsplit(".", 1)
            old = model.get_submodule(name)
            with torch.device("meta"):
                lin = ConvRotInt8Linear(old.in_features, old.out_features, bias=old.bias is not None,
                                        rot=r, compute_dtype=dtype)
            lin.qdata, lin.wscale = q.contiguous(), sc.contiguous()
            if old.bias is not None:
                lin.bias = old.bias
            setattr(model.get_submodule(parent), attr, lin)
        missing = [k for k in missing if k[:-len(".weight")] not in rot]

    if missing or unexpected:
        raise ValueError(f"Mage-Flow DiT keys mismatch: missing {missing[:8]} ({len(missing)}), "
                         f"unexpected {unexpected[:8]} ({len(unexpected)})")

    model.pos_embed = MageFlowEmbedRope(theta=10000, axes_dim=cfg["axes_dim"], scale_rope=True)
    model.time_text_embed.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
    return model.to(device)
