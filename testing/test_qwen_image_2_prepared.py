"""Small CPU parity checks; run in an installed AI Toolkit/PyTorch environment."""

import unittest

import torch
from torch import nn

from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
    pack_latents,
    prepare_run_context,
    run_transformer,
    unpack_latents,
)
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import (
    QwenImage21Transformer2DModel,
    _qwenimage21_prefix_segments,
    _qwenimage21_segment_masks,
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
    model = QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=12, axes_dims_rope=(2, 2, 4),
    )
    model.requires_grad_(False)
    for block in model.transformer_blocks:
        block.attn.to_q = _LoRALinear(block.attn.to_q)
    return model


class PreparedQwenTest(unittest.TestCase):
    def test_prepared_output_and_lora_gradients_match_full_sequence(self):
        for references in (0, 2):
            for checkpointing in (False, True):
                with self.subTest(references=references, checkpointing=checkpointing):
                    torch.manual_seed(3)
                    model = _model()
                    if checkpointing:
                        model.enable_gradient_checkpointing()
                    latents = torch.randn(1, 4, 2, 2)
                    text = torch.randn(1, 6, 12)
                    valid = torch.tensor([[True, True, True, True, True, False]])
                    slots = torch.zeros(1, 6, dtype=torch.bool)
                    slots[:, 1:1 + references] = True
                    shapes = [(2, 2)] * references
                    conditions = torch.randn(1, references * 4, 4) if references else None
                    context = prepare_run_context(model, text, valid, slots, target_shape=(2, 2), condition_shapes=shapes)
                    cached_text = context.transformer_context.projected_text.clone()
                    timestep = torch.tensor([0.6])
                    target = torch.randn_like(latents)
                    params = [p for p in model.parameters() if p.requires_grad]

                    for _ in range(2):
                        # General full-sequence API is the reference for pruning
                        # the final head and for cache reuse after a weight update.
                        packed = pack_latents(latents)
                        if conditions is not None:
                            packed = torch.cat([conditions, packed], dim=1)
                        full = model(
                            packed, text, timestep,
                            [[*[(1, h, w) for h, w in shapes], (1, 2, 2)]],
                            torch.cat([slots, torch.ones(1, 1, dtype=torch.bool)], dim=1),
                            encoder_hidden_states_mask=valid, return_dict=False,
                        )[0]
                        expected = unpack_latents(full[:, -4:], 2, 2)
                        ((expected - target) ** 2).mean().backward()
                        gradients = [p.grad.clone() for p in params]
                        model.zero_grad(set_to_none=True)
                        actual = run_transformer(
                            model, latents, timestep, text, valid, slots,
                            condition_latents=conditions, condition_shapes=shapes,
                            prepared_context=context,
                        )
                        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                        ((actual - target) ** 2).mean().backward()
                        for param, expected_grad in zip(params, gradients):
                            torch.testing.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(context.transformer_context.projected_text, cached_text)
                        with torch.no_grad():
                            for param in params:
                                param.add_(param.grad, alpha=-0.01)
                        model.zero_grad(set_to_none=True)

    def test_shared_segment_masks_match_block_causal_attention(self):
        ids = torch.tensor([-1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 2, 2, 2, 2])
        valid = torch.ones(2, 15, dtype=torch.bool)
        valid[0, 10] = False
        segments = _qwenimage21_prefix_segments(ids, 11)
        masks = _qwenimage21_segment_masks(segments, valid, torch.device("cpu"))
        positions = torch.arange(15)
        # Independent definition: earlier keys or the same image block, with
        # invalid keys removed. Adjacent references must stay distinct blocks.
        allowed = ((positions[:, None] >= positions[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)))
        allowed = allowed[None] & valid[:, None, :]
        for (start, end, _), mask in zip(segments, masks):
            torch.testing.assert_close(mask.expand(2, 1, end - start, end)[:, 0], allowed[:, start:end, :end])
        torch.testing.assert_close(masks[-1].expand(2, 1, 4, 15)[:, 0], allowed[:, 11:])

    def test_prepared_context_rejects_changed_masks_or_projection(self):
        for change in ("mask", "projection"):
            with self.subTest(change=change):
                model = _model()
                text = torch.randn(1, 4, 12)
                valid = torch.ones(1, 4, dtype=torch.bool)
                slots = torch.zeros_like(valid)
                context = prepare_run_context(model, text, valid, slots, target_shape=(2, 2))
                with torch.no_grad():
                    if change == "mask":
                        valid[0, -1] = False
                    else:
                        model.txt_in.in_layer.weight.add_(0.1)
                with self.assertRaises(ValueError):
                    run_transformer(model, torch.randn(1, 4, 2, 2), torch.tensor([0.5]), text, valid, slots, prepared_context=context)


if __name__ == "__main__":
    unittest.main()
