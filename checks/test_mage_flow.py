"""Unit tests for the Mage-Flow family: sampling, DiT, VAE, driver batch/edit parity, block swap, and LoRA/LoKR."""
import copy
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from simgpu import simulated_gpu

from fizgig.families.lora import FamilyLoRA, TRAINABLE
from fizgig.families.registry import family_of_lora, get as get_family
from fizgig.mage_flow import sampling as S
from fizgig.mage_flow.model import MageFlowDiT, is_convrot, load_mage_flow_dit
from fizgig.mage_flow.vae import MageVAE, _DConvDenoiser, _DConvEncoder


def _tiny_dit(*, depth=4, dtype=torch.bfloat16):
    return MageFlowDiT(
        in_channels=128,
        out_channels=128,
        context_in_dim=64,
        hidden_size=128,
        num_heads=4,
        depth=depth,
        axes_dim=(8, 12, 12),
        patch_size=1,
        mlp_ratio=2.0,
    ).to(dtype)


def test_sampling_schedule_and_initial_noise():
    sched = S.sigma_schedule(30)
    assert sched.shape == (31,)
    assert sched[0].item() == pytest.approx(1.0, abs=1e-6)
    assert sched[-1].item() == 0.0
    # s = 0.5 with static shift 6.0 -> 3.0 / 3.5 = 6/7
    assert S.time_shift(0.5, 6.0) == pytest.approx(6.0 / 7.0, rel=1e-7)

    n1 = S.initial_noise(42, 64, 96)
    n2 = S.initial_noise(42, 64, 96)
    n3 = S.initial_noise(43, 64, 96)
    assert n1.shape == (1, 128, 4, 6)
    assert torch.equal(n1, n2)
    assert not torch.equal(n1, n3)


def test_dit_forward_t2i_edit_and_masks():
    dit = _tiny_dit(depth=2, dtype=torch.float32).eval()
    img = torch.randn(2, 16, 128)
    txt0 = torch.randn(5, 64)
    txt1 = torch.randn(7, 64)
    txt_pad = torch.stack([torch.cat([txt0, torch.zeros(2, 64)]), txt1])
    mask = torch.tensor([[True] * 5 + [False] * 2, [True] * 7])
    t = torch.tensor([0.3, 0.8])

    with torch.no_grad():
        o_batch = dit(img, txt_pad, t, [[(1, 4, 4)]], encoder_hidden_states_mask=mask)
        o0 = dit(img[:1], txt0[None], t[:1], [[(1, 4, 4)]])
        o1 = dit(img[1:], txt1[None], t[1:], [[(1, 4, 4)]])
        assert torch.allclose(o_batch[0], o0[0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(o_batch[1], o1[0], atol=1e-5, rtol=1e-4)

        # Edit: target first (16) + 2 refs (12 + 9)
        edit_in = torch.randn(1, 16 + 12 + 9, 128)
        o_edit = dit(edit_in, txt0[None], t[:1], [[(1, 4, 4), (1, 3, 4), (1, 3, 3)]])
        assert o_edit.shape == edit_in.shape

        # Prompt-travel float weights [0..1]
        weights = torch.ones(1, 5, dtype=torch.float32)
        weights[0, 3:] = 0.5
        o_w = dit(img[:1], txt0[None], t[:1], [[(1, 4, 4)]], encoder_hidden_states_mask=weights)
        assert o_w.shape == (1, 16, 128) and torch.isfinite(o_w).all()


def test_driver_batch_edit_and_sampling():
    desc = get_family("mage_flow")
    drv = desc.load_driver()
    dit = _tiny_dit(depth=2, dtype=torch.bfloat16).eval()

    cond0 = {"hidden_states": torch.randn(1, 5, 64, dtype=torch.bfloat16)}
    cond1 = {"hidden_states": torch.randn(1, 8, 64, dtype=torch.bfloat16)}
    lat0 = torch.randn(1, 128, 4, 4, dtype=torch.bfloat16)
    lat1 = torch.randn(1, 128, 4, 4, dtype=torch.bfloat16)

    with torch.no_grad():
        g0 = torch.Generator().manual_seed(123)
        l0, _ = drv.training_loss(dit, lat0, cond0, g0)
        l1, _ = drv.training_loss(dit, lat1, cond1, g0)
        g_b = torch.Generator().manual_seed(123)
        stacked = drv.stack_conds([cond0, cond1])
        assert stacked["hidden_states"].shape[1] % 16 == 0
        lb, _ = drv.training_loss_batch(dit, torch.cat([lat0, lat1]), stacked, g_b)
        assert abs(0.5 * (l0 + l1).item() - lb.item()) < 2e-2

        # Edit training with ref + diff_ref + loss_mask
        ref = [torch.randn(1, 128, 3, 4, dtype=torch.bfloat16)]
        le, _ = drv.training_loss(
            dit, lat0, cond0, torch.Generator().manual_seed(9),
            refs=ref, diff_ref=lat1, diff_weight=0.5, loss_mask=torch.ones(1, 1, 4, 4),
        )
        assert torch.isfinite(le)

        # Predict + sample (T2I CFG and Edit)
        st = drv.noise_latents(lat0, torch.Generator().manual_seed(1))
        assert drv.map_target(lat0, st).shape == (1, 16, 128)
        assert drv.predict(dit, st, cond0).shape == (1, 16, 128)

        c_t2i = {"hidden_states": torch.randn(5, 64, dtype=torch.bfloat16)}
        neg = {"hidden_states": torch.randn(4, 64, dtype=torch.bfloat16)}
        out = drv.generate(dit, c_t2i, 64, 64, steps=2, seed=0, cfg=5.0, neg_cond=neg)
        assert out.shape == (1, 128, 4, 4) and torch.isfinite(out).all()
        out_edit = drv.generate(dit, c_t2i, 64, 64, steps=2, seed=0, cfg=5.0, neg_cond=neg, refs=ref)
        assert out_edit.shape == (1, 128, 4, 4) and torch.isfinite(out_edit).all()

    padded = drv.pad_conditioning([c_t2i, {"hidden_states": torch.randn(8, 64, dtype=torch.bfloat16)}])
    assert padded[0]["mask"].sum().item() == 5 and padded[0]["hidden_states"].shape[0] == 8
    full_txt, dropped = drv.trigger_text(object(), "a red cube")
    assert "a red cube" in full_txt and dropped in full_txt
    assert len(drv.quant_target_names(dit)) > 0
    assert drv.ft_spec(dit).blocks == "transformer_blocks"


def test_vae_posterior_mode():
    vae = MageVAE.__new__(MageVAE)
    torch.nn.Module.__init__(vae)
    vae.latent_channels = 8
    vae.downsample_factor = 16
    vae.sample_posterior = False
    vae.dconv_encoder = _DConvEncoder(
        z_ch=8, hidden_size=32, num_blocks=1, patch_size=16, mlp_ratio=2.0, head_size=32, num_head_blocks=1,
    )
    vae.decoder_model = _DConvDenoiser(
        patch_size=16, hidden_size=32, hidden_size_x=8, mlp_ratio=2.0, num_blocks=2, num_cond_blocks=1, bottleneck_dim=8,
    )
    vae._freeze_adaln_cache()
    vae.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        z1 = vae.encode(x)
        z2 = vae.encode(x)
        assert z1.shape == (1, 8, 2, 2)
        assert torch.equal(z1, z2)
        rec = vae.decode(z1)
        assert rec.shape == x.shape


@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_lora_and_lokr_roundtrip_and_block_swap(kind):
    desc = get_family("mage_flow")
    drv = desc.load_driver()
    base = _tiny_dit(depth=4, dtype=torch.bfloat16).requires_grad_(False)
    ref_dit = copy.deepcopy(base)
    swap_dit = copy.deepcopy(base)

    cond = {"hidden_states": torch.randn(1, 5, 64, dtype=torch.bfloat16)}
    lat = torch.randn(1, 128, 4, 4, dtype=torch.bfloat16)

    with simulated_gpu():
        ref_dit.to("cuda")
        drv.enable_gradient_checkpointing(ref_dit, True)
        torch.manual_seed(2)
        net_ref = FamilyLoRA(ref_dit, drv, device="cuda")
        net_ref.add_trainable(4, 4, kind=kind, factor=4)
        for p in net_ref.parameters():
            p.data = p.data.to("cuda")
            torch.nn.init.normal_(p, std=0.05)

        drv.enable_block_swap(swap_dit, 2, "cuda")
        drv.enable_gradient_checkpointing(swap_dit, True)
        torch.manual_seed(2)
        net_swap = FamilyLoRA(swap_dit, drv, device="cuda")
        net_swap.add_trainable(4, 4, kind=kind, factor=4)
        for p in net_swap.parameters():
            p.data = p.data.to("cuda")
            torch.nn.init.normal_(p, std=0.05)

        ref_dit.train()
        swap_dit.train()
        l_ref, _ = drv.training_loss(ref_dit, lat.to("cuda"), cond, torch.Generator().manual_seed(7))
        l_ref.backward()
        l_swp, _ = drv.training_loss(swap_dit, lat.to("cuda"), cond, torch.Generator().manual_seed(7))
        l_swp.backward()

        assert torch.allclose(l_ref.cpu(), l_swp.cpu(), atol=1e-4, rtol=1e-3)
        for pr, ps in zip(net_ref.parameters(), net_swap.parameters()):
            if pr.grad is not None or ps.grad is not None:
                assert torch.allclose(pr.grad.cpu(), ps.grad.cpu(), atol=1e-3, rtol=1e-2)


def test_lora_file_detection_and_convrot_roundtrip(tmp_path):
    desc = get_family("mage_flow")
    drv = desc.load_driver()
    for kind in ("lora", "lokr"):
        dit = _tiny_dit(depth=2, dtype=torch.bfloat16)
        spec = FamilyLoRA(dit, drv)
        spec.add_trainable(rank=4, alpha=4.0, kind=kind, factor=4)
        sd, _ = spec.bake([TRAINABLE])
        p = tmp_path / f"{kind}.safetensors"
        save_file(sd, str(p))
        assert family_of_lora(str(p)) is desc

    # Synthetic ComfyUI int8 ConvRot checkpoint detection + dequantized load
    from fizgig.minimax.convrot import ConvRotInt8Linear, quantize_int8_convrot
    src = _tiny_dit(depth=2, dtype=torch.bfloat16)
    sd = {}
    for k, v in src.state_dict().items():
        if k.startswith("transformer_blocks.0.") and k.endswith(".weight") and v.ndim == 2:
            q, sc = quantize_int8_convrot(v.float(), rot=64)
            base = k[:-len(".weight")]
            sd[k] = q
            sd[base + ".weight_scale"] = sc.reshape(-1, 1)
            sd[base + ".comfy_quant"] = torch.tensor(
                list(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 64}'),
                dtype=torch.uint8,
            )
        else:
            sd[k] = v
    ckpt = tmp_path / "convrot.safetensors"
    save_file(sd, str(ckpt))
    assert is_convrot(str(ckpt))
    cfg = dict(
        in_channels=128, out_channels=128, context_in_dim=64,
        hidden_size=128, num_heads=4, depth=2, axes_dim=(8, 12, 12), patch_size=1, mlp_ratio=2.0,
    )
    loaded = load_mage_flow_dit(str(ckpt), device="cpu", dtype=torch.bfloat16, config=cfg, keep_convrot=True)
    assert isinstance(loaded.transformer_blocks[0].img_mlp.net[2], ConvRotInt8Linear)
    deq = load_mage_flow_dit(str(ckpt), device="cpu", dtype=torch.bfloat16, config=cfg, keep_convrot=False)
    assert isinstance(deq.transformer_blocks[0].img_mlp.net[2], torch.nn.Linear)
