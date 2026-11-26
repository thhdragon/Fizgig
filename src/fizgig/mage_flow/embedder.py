"""Mage-Flow text conditioning: Qwen3-VL-4B, last_hidden_state (AFTER final RMSNorm), system/user-opening dropped.

Follows Microsoft Mage-Flow (source_reference.temp/models/modules/text_encoder.py, models/utils.py) and
ComfyUI (comfy/text_encoders/mage_flow.py):
  - Uses Qwen3-VL-4B-Instruct (`last_hidden_state`, WITH the final RMSNorm — unlike Qwen Image 2.1).
  - T2I template strips the leading 34 tokens (system + `<|im_start|>user\n`).
  - Edit template uses `Image 1: <|vision_start|><|image_pad|><|vision_end|>...{instruction}`, strips 64 tokens,
    and caps reference images' long edge at 384 px before Qwen3-VL vision preprocessing.
  - Loads single-file ComfyUI checkpoints (bf16 or fp8_scaled) or HF sharded text_encoder/ dirs.
  - Tokenizer/processor loads offline from the bundled `assets/qwen3vl_tokenizer/` (or a local dir next to the ckpt).
"""
from __future__ import annotations

import logging
import os

import torch
from PIL import Image

logger = logging.getLogger(__name__)

SYS_PROMPT_T2I = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT_T2I}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SYS_PROMPT_EDIT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)
TEMPLATE_EDIT = (
    f"<|im_start|>system\n{SYS_PROMPT_EDIT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VL_COND_LONG_EDGE = 384
MAX_SEQ_LENGTH = 2048


def _resize_long_edge(image: Image.Image, max_long_edge: int | None = VL_COND_LONG_EDGE) -> Image.Image:
    if max_long_edge is None or max_long_edge <= 0:
        return image
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    return image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BICUBIC)


def _resolve_te_dir(path: str) -> str:
    """Resolve a HF repo root (holding text_encoder/) or text_encoder dir or weights file."""
    if os.path.isdir(path):
        sub = os.path.join(path, "text_encoder")
        return sub if os.path.isdir(sub) else path
    return path


def _resolve_tokenizer_dir(model_path: str) -> str:
    """Offline tokenizer/processor dir: checkpoint-local override, HF text_encoder dir, or Fizgig's bundled copy."""
    from fizgig.krea2.embedder import _bundled_tokenizer_dir, _local_tokenizer_dir
    local = _local_tokenizer_dir(model_path)
    if os.path.isfile(os.path.join(local, "tokenizer.json")):
        return local
    resolved = _resolve_te_dir(model_path)
    if os.path.isdir(resolved) and os.path.isfile(os.path.join(resolved, "tokenizer.json")):
        return resolved
    return _bundled_tokenizer_dir()


def _load_te_state_dict(model_path: str, dtype: torch.dtype, prequant: bool) -> dict:
    from fizgig.krea2.embedder import _convert_comfyui_qwen3vl_state_dict
    from fizgig.krea2.safetensors_utils import ShardedSafeOpen, load_split_weights

    resolved = _resolve_te_dir(model_path)
    if os.path.isdir(resolved):
        sharded = ShardedSafeOpen(resolved)
        sd = {k: (sharded.get_tensor(k) if prequant else sharded.get_tensor(k).to(dtype)) for k in sharded.keys()}
    else:
        sd = load_split_weights(resolved, device="cpu", dtype=(None if prequant else dtype))
    return _convert_comfyui_qwen3vl_state_dict(sd)


class MageFlowTextEncoder:
    def __init__(self, model_path: str, tokenizer_dir: str | None = None, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, int8: bool = False, vision: bool = False):
        from accelerate import init_empty_weights
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from fizgig.krea2.embedder import QWEN3_VL_4B_INSTRUCT_CONFIG
        from fizgig.krea2.utils import (_COMFY_FP8_MARKER_SUFFIX, _FP8_SCALE_SUFFIX,
                                        _reshape_prequant_fp8_scale, is_prequantized_fp8)

        self.device, self.dtype = torch.device(device), dtype
        tok_src = tokenizer_dir or _resolve_tokenizer_dir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.processor = AutoProcessor.from_pretrained(tok_src, local_files_only=True) if vision else None

        config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(config)

        resolved = _resolve_te_dir(model_path)
        prequant = os.path.isfile(resolved) and is_prequantized_fp8(resolved)
        sd = _load_te_state_dict(model_path, dtype, prequant)

        if prequant:
            fixed = {}
            for k, v in sd.items():
                if k.endswith(_COMFY_FP8_MARKER_SUFFIX):
                    continue
                if k.endswith(_FP8_SCALE_SUFFIX):
                    fixed[k[:-len(_FP8_SCALE_SUFFIX)] + ".scale_weight"] = _reshape_prequant_fp8_scale(v).to(dtype)
                else:
                    fixed[k] = v
            sd = fixed
            from fizgig.krea2.fp8_optimization_utils import apply_fp8_monkey_patch
            apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        info = model.load_state_dict(sd, strict=False, assign=True)
        model.tie_weights()
        unexpected = list(info.unexpected_keys)
        missing = [k for k in info.missing_keys if k != "lm_head.weight"]
        if unexpected or missing:
            raise RuntimeError(f"Qwen3-VL-4B checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")

        model.lm_head = None
        if not vision:
            model.model.visual = None
        self.vision = vision
        del sd

        if int8 and not prequant:
            from fizgig.qwen_image21.embedder import _int8_weights
            n = _int8_weights(model, "language_model.layers.", self.device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")

        if dtype is not None and not prequant and not int8:
            model.to(dtype)
        self.model = model.to(self.device).eval().requires_grad_(False)

        self.drop_idx_t2i = len(self.tokenizer(TEMPLATE_T2I.split("{}")[0], add_special_tokens=False).input_ids)
        self.drop_idx_edit = len(self.tokenizer(TEMPLATE_EDIT.split("{}")[0], add_special_tokens=False).input_ids)

    @torch.no_grad()
    def encode(self, prompts) -> list[torch.Tensor]:
        """-> list of [L_i, 2560] tensors (real tokens after system/user prefix, on CPU)."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        texts = [TEMPLATE_T2I.format(p if p else " ") for p in prompts]
        tok = self.tokenizer(
            texts,
            padding=True,
            padding_side="left",
            max_length=MAX_SEQ_LENGTH + self.drop_idx_t2i,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.model(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        hs = out.last_hidden_state
        res = []
        for h, m in zip(hs, tok.attention_mask.bool()):
            res.append(h[m][self.drop_idx_t2i:].to("cpu"))
        return res

    @torch.no_grad()
    def encode_with_references(self, prompt: str, images: list[Image.Image]) -> torch.Tensor:
        """Edit conditioning: reference images capped to 384 px long edge, formatted with Image N: <vision> prefix.
        -> [L, 2560] tensor on CPU."""
        if not self.vision or self.processor is None:
            raise RuntimeError("this Qwen3-VL encoder was loaded without its vision tower (vision=False)")
        pils = [_resize_long_edge(im.convert("RGB"), VL_COND_LONG_EDGE) for im in images]
        prefix = "".join(f"Image {j + 1}: {EDIT_IMAGE_PLACEHOLDER}" for j in range(len(pils)))
        formatted = TEMPLATE_EDIT.format(prefix + (prompt if prompt else " "))
        inputs = self.processor(
            text=[formatted],
            images=pils,
            padding=True,
            return_tensors="pt",
        )
        kw = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "output_hidden_states": False,
            "use_cache": False,
        }
        if inputs.get("pixel_values") is not None:
            kw["pixel_values"] = inputs["pixel_values"].to(self.device, self.dtype)
            kw["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
        out = self.model.model(**kw)
        return out.last_hidden_state[0][self.drop_idx_edit:].to("cpu")

    def unload(self):
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
