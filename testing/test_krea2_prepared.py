"""Focused CPU parity checks; run in an installed AI Toolkit/PyTorch environment."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from extensions_built_in.diffusion_models.krea2.src import mmdit
from extensions_built_in.diffusion_models.krea2.src.pipeline import (
    pad_text_features,
    predict_velocity,
    prepare_velocity_context,
)


class _LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.randn(2, base.in_features) * 0.01)
        self.up = nn.Parameter(torch.randn(base.out_features, 2) * 0.01)

    def forward(self, value):
        return self.base(value) + (value @ self.down.T) @ self.up.T


def _model():
    model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(
        features=32, tdim=16, txtdim=16, heads=2, kvheads=1,
        multiplier=1, layers=2, patch=2, channels=4,
        txtlayers=2, txtheads=2, txtkvheads=1,
    ))
    model.requires_grad_(False)
    for blocks in (model.blocks, model.txtfusion.layerwise_blocks, model.txtfusion.refiner_blocks):
        for block in blocks:
            block.attn.wq = _LoRALinear(block.attn.wq)
    return model


class PreparedKreaTest(unittest.TestCase):
    def test_cached_geometry_matches_full_head_outputs_and_lora_gradients(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(8)
                model = _model()
                self.assertTrue(mmdit._can_slice_last_layer(model.last))
                if checkpointing:
                    model.enable_gradient_checkpointing()
                latents = torch.randn(2, 4, 4, 8)
                text, mask = pad_text_features(
                    [torch.randn(5, 32), torch.randn(3, 32)],
                    torch.device("cpu"), torch.float32,
                )
                prepared = prepare_velocity_context(model, latents, text, mask)
                params = {name: p for name, p in model.named_parameters() if p.requires_grad}
                target = torch.randn_like(latents)
                timestep = torch.tensor([0.6, 0.3])
                for _ in range(2):
                    # Disable pruning only in the reference call, reproducing
                    # the original full-head path without changing production API.
                    with patch.object(mmdit, "_can_slice_last_layer", return_value=False):
                        expected = predict_velocity(model, latents, timestep, text, mask)
                    ((expected - target) ** 2).mean().backward()
                    gradients = {name: p.grad.clone() for name, p in params.items()}
                    for group in ("blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."):
                        self.assertTrue(any(bool(g.abs().sum() > 0) for name, g in gradients.items() if name.startswith(group)))
                    model.zero_grad(set_to_none=True)
                    actual = predict_velocity(model, latents, timestep, text, mask, prepared_context=prepared)
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    ((actual - target) ** 2).mean().backward()
                    for name, param in params.items():
                        torch.testing.assert_close(param.grad, gradients[name], atol=1e-5, rtol=1e-5)
                    with torch.no_grad():
                        for param in params.values():
                            param.add_(param.grad, alpha=-0.01)
                    model.zero_grad(set_to_none=True)
                    # Cache reuse must still consume fresh noisy image values.
                    latents = torch.randn_like(latents)

    def test_head_pruning_preserves_reference_outputs_and_patched_head_rng(self):
        torch.manual_seed(12)
        model = _model()
        self.assertTrue(mmdit._can_slice_last_layer(model.last))
        latents = torch.randn(1, 4, 4, 8)
        text = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5, dtype=torch.long)
        timestep = torch.tensor([0.7])
        refs = [[torch.randn(4, 2, 2), torch.randn(4, 4, 2)]]
        for isolate in (False, True):
            with self.subTest(isolate_refs=isolate), torch.no_grad():
                with patch.object(mmdit, "_can_slice_last_layer", return_value=False):
                    expected = predict_velocity(model, latents, timestep, text, mask, ref_latents=refs, isolate_refs=isolate)
                actual = predict_velocity(model, latents, timestep, text, mask, ref_latents=refs, isolate_refs=isolate)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        # External adapters patch Linear.forward without placing their dropout
        # inside the original module's children. Preserve full-row RNG draws.
        original_forward = model.last.linear.forward
        model.last.linear.forward = lambda value: original_forward(F.dropout(value, p=0.25, training=True))
        self.assertFalse(mmdit._can_slice_last_layer(model.last))
        torch.manual_seed(19)
        with patch.object(mmdit, "_can_slice_last_layer", return_value=False):
            expected = predict_velocity(model, latents, timestep, text, mask)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(19)
        actual = predict_velocity(model, latents, timestep, text, mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)

    def test_geometry_rejects_stale_sources_and_equal_token_count_grid_changes(self):
        for change in ("mask", "text", "grid", "position", "references"):
            with self.subTest(change=change):
                model = _model()
                latents = torch.randn(1, 4, 4, 8)
                text = torch.randn(1, 5, 32)
                mask = torch.ones(1, 5, dtype=torch.long)
                prepared = prepare_velocity_context(model, latents, text, mask)
                kwargs = {}
                if change == "mask":
                    mask[0, -1] = 0
                elif change == "text":
                    text.add_(0.1)
                elif change == "grid":
                    latents = torch.randn(1, 4, 8, 4)
                elif change == "position":
                    model.posemb.ntk = 2.0
                else:
                    kwargs["ref_latents"] = [[torch.randn(4, 2, 2)]]
                with self.assertRaises(ValueError):
                    predict_velocity(model, latents, torch.tensor([0.5]), text, mask, prepared_context=prepared, **kwargs)


if __name__ == "__main__":
    unittest.main()
