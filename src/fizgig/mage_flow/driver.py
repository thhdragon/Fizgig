"""Mage-Flow driver for Fizgig's standard layer (families/driver.py).

* model: mage_flow/model.py (12-block double-stream MMDiT, 4B params, patch_size=1, 128 latent channels),
  loading bf16 single files, ComfyUI int8 ConvRot files, or diffusers transformer/ dirs
* conditioning: Qwen3-VL-4B last_hidden_state (WITH final RMSNorm), system/user prefix stripped ->
  {"hidden_states": (L, 2560)}
* latents: Mage-VAE (128 channels, 16x downsample, posterior mean)
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0, logit-normal t with
  static shift 6.0 (schedule_mode="z-image"); optional edit references appended after target tokens
* sampling: FlowMatch Euler with static shift 6.0 (sampling.py); no Gaussian watermark
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import FamilyDriver, picture_losses, prior_base, weighted_loss, with_prior
from fizgig.mage_flow import sampling as S

logger = logging.getLogger(__name__)


class MageFlowDriver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.mage_flow.model import load_mage_flow_dit
        return load_mage_flow_dit(path, device=device).eval().requires_grad_(False)

    def load_planned(self, path, device, precision, blocks_to_swap):
        """ComfyUI int8 ConvRot checkpoints load natively when unswapped; block-swapped loads decode to INT8."""
        from fizgig.mage_flow.model import is_convrot, load_mage_flow_dit
        if not is_convrot(path):
            return None
        swap = min(int(blocks_to_swap or 0), self.max_blocks_to_swap())
        if swap <= 0:
            logger.info(f"[precision] {os.path.basename(path)}: int8 ConvRot, as stored")
            return load_mage_flow_dit(path, device=device).eval().requires_grad_(False), 0
        from fizgig.families import quant
        dit = load_mage_flow_dit(path, device="cpu", keep_convrot=False).eval().requires_grad_(False)
        quant.quantize(dit, self, "int8", device, store_device="cpu")
        self.enable_block_swap(dit, swap, device, False)
        logger.info(f"[precision] {os.path.basename(path)}: decoded to INT8 with {swap} blocks streamed")
        return dit, swap

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.transformer_blocks) if dit is not None else self.description.n_blocks) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    # ---- training previews on the Turbo checkpoint ----------------------------------------------
    def park_for_preview(self, dit, device):
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.families import quant
            torch.cuda.empty_cache()
            if quant.free_vram_gb() >= 6.5:
                return None
            quant.move(dit, "cpu")
            torch.cuda.empty_cache()
            return 0
        orig = int(dit.blocks_to_swap or 0)
        dit.enable_block_swap(len(dit.transformer_blocks) - 2, torch.device(device), True)
        dit.prepare_block_swap_before_forward()
        return orig

    def load_preview_checkpoint(self, path, device, int8=False):
        from fizgig.families import quant
        from fizgig.mage_flow.model import is_convrot
        torch.cuda.empty_cache()
        free = quant.free_vram_gb()
        size = os.path.getsize(path) / 1024 ** 3
        room = 3.5
        if is_convrot(path):
            precision, need = "int8", size + room
        else:
            precision = "int8" if int8 or free < size + room else "bf16"
            need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2 if not is_convrot(path) else size) / max(1, self.description.n_blocks)
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        if swapped:
            self.block_swap_mode(m, inference=True)
        logger.info(f"[sample] Turbo checkpoint: {precision}, {swapped} blocks streamed ({free:.1f} GB free)")
        return m.eval().requires_grad_(False), swapped

    def unpark_after_preview(self, dit, device, token):
        if token is None:
            return
        device = torch.device(device)
        if token > 0:
            dit.enable_block_swap(token, device, True)
            dit.move_to_device_except_swap_blocks(device)
        else:
            from fizgig.families import quant
            dit.disable_block_swap()
            quant.move(dit, device)
        dit.switch_block_swap_for_training()

    def ft_spec(self, dit):
        from fizgig.families.ft import FTSpec
        return FTSpec(
            blocks="transformer_blocks",
            components=("attn", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2"),
            overhead_gb=3.5,
            stream_base_gb=2.0,
            calib_mp=0.5,
            act_gb_per_mp=2.5,
        )

    def quant_target_names(self, dit):
        """Quantize every block Linear (attention, MLPs, and img_mod.1 / txt_mod.1) while LoRA trains attention + MLPs."""
        names = list(self.lora_target_names(dit))
        n = len(dit.transformer_blocks) if dit is not None else self.description.n_blocks
        existing = {m for m, _ in dit.named_modules()} if dit is not None else None
        for i in range(n):
            for mod in (f"transformer_blocks.{i}.img_mod.1", f"transformer_blocks.{i}.txt_mod.1"):
                if existing is None or mod in existing:
                    names.append(mod)
        return names

    def load_vae(self, path, device):
        from fizgig.mage_flow.vae import load_mage_vae
        return load_mage_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Qwen3-VL-4B text-only: bf16 (~7.2 GB without LM head/vision) when it fits, else INT8 (~4.2 GB)."""
        from fizgig.families.quant import free_vram_gb
        from fizgig.mage_flow.embedder import MageFlowTextEncoder
        return MageFlowTextEncoder(path, device=device, int8=free_vram_gb() < 10.5)

    supports_references = True

    def load_reference_text_encoder(self, path, device):
        from fizgig.families.quant import free_vram_gb
        from fizgig.mage_flow.embedder import MageFlowTextEncoder
        return MageFlowTextEncoder(path, device=device, int8=free_vram_gb() < 11.5, vision=True)

    def unload_text_encoder(self, te):
        te.unload()

    def compile_targets(self, dit):
        return dit.transformer_blocks

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        return [z.to(torch.bfloat16).cpu() for z in S.encode_image(vae, x)]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in te.encode(captions)]

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        from PIL import Image
        from fizgig.utils.image import flatten
        out = []
        for cap, imgs in zip(captions, references):
            pils = [flatten(Image.fromarray(np.ascontiguousarray(a))) for a in imgs]
            h = te.encode_with_references(cap, pils)
            out.append({"hidden_states": h.to(torch.bfloat16).cpu()})
        return out

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(generator, min_t: float, max_t: float) -> float:
        """Logit-normal shifted by static_shift=6.0 (schedule_mode='z-image'); ~3.7% below 0.5 before [min_t, max_t]."""
        t = torch.sigmoid(torch.randn(1, generator=generator)).item()
        t = S.time_shift(t, S.SHIFT)
        return min_t + (max_t - min_t) * t

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        device = latents.device
        h, w = latents.shape[-2:]
        n = h * w
        x0 = S.pack(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(device)
        t = self._sample_t(generator, min_t, max_t)
        xt = (1.0 - t) * x0 + t * noise
        enc = cond["hidden_states"].to(device)
        if enc.dim() == 2:
            enc = enc[None]
        enc_mask = cond.get("mask")
        if enc_mask is not None:
            enc_mask = enc_mask.to(device)
            if enc_mask.dim() == 1:
                enc_mask = enc_mask[None]

        shapes = [(1, h, w)]
        xin = xt
        if refs:
            ref_list = [r.float().to(device) for r in refs]
            ref_tokens = torch.cat([S.pack(r) for r in ref_list], dim=1)
            xin = torch.cat([xt, ref_tokens], dim=1)
            shapes = shapes + [(1, r.shape[-2], r.shape[-1]) for r in ref_list]

        tt = torch.tensor([t], device=device, dtype=torch.bfloat16)
        base = prior_base(prior, loss_mask, lambda: dit(xin.to(torch.bfloat16), enc.to(torch.bfloat16),
                                                         tt, [shapes], enc_mask)[:, :n])
        pred = dit(xin.to(torch.bfloat16), enc.to(torch.bfloat16), tt, [shapes], enc_mask)[:, :n]
        target = noise - x0

        wgt = None
        if diff_ref is not None and diff_weight > 0.0:
            d = S.pack((latents.float() - diff_ref.to(device).float()).abs()).mean(dim=-1)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean(dim=1, keepdim=True).clamp_min(1e-8)
            wgt = torch.where(dm > 1e-6, wgt, torch.ones_like(wgt))
        mw = None if loss_mask is None else S.pack(loss_mask.to(device).float()).mean(dim=-1)
        if wgt is None and mw is None:
            return F.mse_loss(pred.float(), target), {"t": t}
        loss = weighted_loss((pred.float() - target).pow(2).mean(dim=-1), wgt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred.float() - base).pow(2).mean(dim=-1), mw, wgt)
        return loss, {"t": t}

    def stack_conds(self, conds):
        """Pad variable-length captions to a multiple of 16 with a boolean attention mask."""
        hs = [c["hidden_states"][0] if c["hidden_states"].dim() == 3 else c["hidden_states"] for c in conds]
        L = max(h.shape[0] for h in hs)
        L += -L % 16
        return {
            "hidden_states": torch.stack([torch.cat([h, h.new_zeros(L - h.shape[0], h.shape[1])]) for h in hs]),
            "mask": torch.stack([torch.arange(L, device=h.device) < h.shape[0] for h in hs]),
        }

    def training_loss_batch(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, loss_mask=None):
        device = latents.device
        h, w = latents.shape[-2:]
        n = h * w
        x0 = S.pack(latents.float())
        noises, ts = [], []
        for i in range(x0.shape[0]):
            noises.append(torch.randn(x0[i:i + 1].shape, generator=generator))
            ts.append(self._sample_t(generator, min_t, max_t))
        noise = torch.cat(noises).to(device)
        t = torch.tensor(ts, device=device, dtype=torch.float32).view(-1, 1, 1)
        xt = (1.0 - t) * x0 + t * noise
        enc = cond["hidden_states"].to(device)
        enc_mask = cond["mask"].to(device)
        tt = torch.tensor(ts, device=device, dtype=torch.bfloat16)
        pred = dit(xt.to(torch.bfloat16), enc.to(torch.bfloat16), tt, [[(1, h, w)]], enc_mask)[:, :n]
        mw = None if loss_mask is None else S.pack(loss_mask.to(device).float()).mean(dim=-1)
        per = picture_losses((pred.float() - (noise - x0)).pow(2).mean(dim=-1), mw)
        return per.mean(), {"t": sum(ts) / len(ts), "per": per, "ts": [float(v) for v in ts]}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        h, w = latents.shape[-2:]
        x0 = S.pack(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        t = self._sample_t(generator, min_t, max_t)
        return {"xt": (1.0 - t) * x0 + t * noise, "t": t, "hw": (h, w)}

    def map_target(self, latents, state):
        return (state["xt"].float() - S.pack(latents.float()).to(state["xt"].device)) / float(state["t"])

    def trigger_text(self, tokenizer, caption):
        from fizgig.mage_flow.embedder import TEMPLATE_T2I
        dropped = TEMPLATE_T2I.split("{}", 1)[0]
        return TEMPLATE_T2I.replace("{}", caption, 1), dropped

    def predict(self, dit, state, cond):
        h, w = state["hw"]
        n = h * w
        device = state["xt"].device
        enc = cond["hidden_states"].to(device)
        if enc.dim() == 2:
            enc = enc[None]
        enc_mask = cond.get("mask")
        if enc_mask is not None:
            enc_mask = enc_mask.to(device)
            if enc_mask.dim() == 1:
                enc_mask = enc_mask[None]
        tt = torch.tensor([state["t"]], device=device, dtype=torch.bfloat16)
        return dit(state["xt"].to(torch.bfloat16), enc.to(torch.bfloat16), tt, [[(1, h, w)]], enc_mask)[:, :n]

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=5.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        neg = neg_cond["hidden_states"] if (neg_cond is not None and cfg > 1.0) else None
        neg_mask = neg_cond.get("mask") if neg is not None else None
        opts = dict(options)
        shift = float(opts.get("shift", S.SHIFT))
        if sigmas is not None and len(sigmas) != steps:
            sigmas = None
        return S.sample(
            dit,
            cond["hidden_states"],
            height,
            width,
            steps=steps,
            seed=seed,
            cfg=cfg,
            neg_emb=neg,
            device=device,
            sigmas=sigmas,
            shift=shift,
            noise=noise,
            on_step=on_step,
            text_mask=cond.get("mask"),
            neg_mask=neg_mask,
            ref_latents=refs or None,
            renormalization=bool(opts.get("renormalization", False)),
        )

    def pad_conditioning(self, conds):
        L = max(c["hidden_states"].shape[0] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"]
            pad = L - h.shape[0]
            out.append({
                "hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]),
            })
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        img = S.decode(vae, latents)[0].float().clamp(-1, 1)
        px = ((img.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
        return Image.fromarray(px)
