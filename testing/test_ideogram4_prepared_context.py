"""Parity tests for Ideogram 4's split static/dynamic transformer forward.

Run directly::

    PYTHONPATH=. python -m unittest -v testing.test_ideogram4_prepared_context
"""

import copy
import unittest

import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Config,
    Ideogram4Transformer2DModel,
)


def _legacy_forward(
    model,
    *,
    llm_features,
    x,
    t,
    position_ids,
    segment_ids,
    indicator,
):
    """Reproduce the transformer forward before context preparation was split."""
    _, _, in_channels = x.shape
    assert in_channels == model.config.in_channels

    param_dtype = model.input_proj.weight.dtype
    x = x.to(param_dtype)
    t = t.to(param_dtype)
    llm_features = llm_features.to(param_dtype)

    indicator = indicator.to(torch.long)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
    output_image_mask = (
        (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
    )

    llm_features = llm_features * llm_token_mask
    x = x * output_image_mask
    x = model.input_proj(x) * output_image_mask

    t_cond = model.t_embedding(t)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(model.adaln_proj(t_cond))

    llm_features = model.llm_cond_norm(llm_features)
    llm_features = model.llm_cond_proj(llm_features) * llm_token_mask

    h = x + llm_features
    h = h + model.embed_image_indicator(
        (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
    )

    cos, sin = model.rotary_emb(position_ids)
    cos = cos.to(h.dtype)
    sin = sin.to(h.dtype)
    attn_mask = (
        segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    ).unsqueeze(1)

    for layer in model.layers:
        h = layer(h, attn_mask, cos, sin, adaln_input, None)

    return model.final_layer(h, c=adaln_input).to(torch.float32)


def _tensor(shape, start, end, *, device, dtype):
    return torch.linspace(
        start,
        end,
        steps=torch.tensor(shape).prod().item(),
        device=device,
        dtype=torch.float32,
    ).reshape(shape).to(dtype)


def _inputs(config, *, device, dtype):
    batch_size = 1
    num_text_tokens = 3
    gh = gw = 2
    num_image_tokens = gh * gw
    seq_len = num_text_tokens + num_image_tokens

    x = torch.zeros(
        batch_size, seq_len, config.in_channels, device=device, dtype=dtype
    )
    x[:, num_text_tokens:] = _tensor(
        (batch_size, num_image_tokens, config.in_channels),
        -0.4,
        0.5,
        device=device,
        dtype=dtype,
    )
    llm_features = torch.zeros(
        batch_size, seq_len, config.llm_features_dim, device=device, dtype=dtype
    )
    llm_features[:, :num_text_tokens] = _tensor(
        (batch_size, num_text_tokens, config.llm_features_dim),
        -0.25,
        0.35,
        device=device,
        dtype=dtype,
    )

    indicator = torch.full(
        (batch_size, seq_len), OUTPUT_IMAGE_INDICATOR, device=device
    )
    indicator[:, :num_text_tokens] = LLM_TOKEN_INDICATOR
    segment_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    text_pos = torch.arange(num_text_tokens, device=device).view(1, -1, 1)
    text_pos = text_pos.expand(batch_size, -1, 3)
    h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1)
    image_pos = image_pos + IMAGE_POSITION_OFFSET
    position_ids = torch.cat(
        [text_pos, image_pos.unsqueeze(0).expand(batch_size, -1, -1)], dim=1
    )
    t = torch.tensor([0.37], device=device, dtype=dtype)
    return x, llm_features, t, position_ids, segment_ids, indicator


class Ideogram4PreparedContextParityTest(unittest.TestCase):
    def _assert_parity(self, device, dtype, compare_update):
        torch.manual_seed(19)
        config = Ideogram4Config(
            emb_dim=24,
            num_layers=2,
            num_heads=3,
            intermediate_size=32,
            adanln_dim=8,
            in_channels=4,
            llm_features_dim=12,
            rope_theta=10_000,
            mrope_section=(1, 1, 1),
        )
        current = Ideogram4Transformer2DModel(config).to(device=device, dtype=dtype)
        legacy = copy.deepcopy(current)

        current_inputs = _inputs(config, device=device, dtype=dtype)
        legacy_inputs = tuple(value.detach().clone() for value in current_inputs)
        current_x, current_llm, current_t = current_inputs[:3]
        legacy_x, legacy_llm, legacy_t = legacy_inputs[:3]
        current_x.requires_grad_(True)
        current_llm.requires_grad_(True)
        current_t.requires_grad_(True)
        legacy_x.requires_grad_(True)
        legacy_llm.requires_grad_(True)
        legacy_t.requires_grad_(True)

        current_output = current(
            llm_features=current_llm,
            x=current_x,
            t=current_t,
            position_ids=current_inputs[3],
            segment_ids=current_inputs[4],
            indicator=current_inputs[5],
        )
        legacy_output = _legacy_forward(
            legacy,
            llm_features=legacy_llm,
            x=legacy_x,
            t=legacy_t,
            position_ids=legacy_inputs[3],
            segment_ids=legacy_inputs[4],
            indicator=legacy_inputs[5],
        )
        self.assertTrue(torch.equal(current_output, legacy_output))

        target = _tensor(
            current_output.shape,
            -0.1,
            0.2,
            device=device,
            dtype=torch.float32,
        )
        current_loss = F.mse_loss(current_output, target)
        legacy_loss = F.mse_loss(legacy_output, target)
        self.assertTrue(torch.equal(current_loss, legacy_loss))

        current_loss.backward()
        legacy_loss.backward()

        for current_value, legacy_value in (
            (current_x.grad, legacy_x.grad),
            (current_llm.grad, legacy_llm.grad),
            (current_t.grad, legacy_t.grad),
        ):
            self.assertTrue(torch.equal(current_value, legacy_value))

        current_parameters = dict(current.named_parameters())
        legacy_parameters = dict(legacy.named_parameters())
        self.assertEqual(current_parameters.keys(), legacy_parameters.keys())
        for name in current_parameters:
            self.assertTrue(
                torch.equal(
                    current_parameters[name].grad,
                    legacy_parameters[name].grad,
                ),
                name,
            )

        if compare_update:
            current_optimizer = torch.optim.AdamW(
                current.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                foreach=False,
                fused=False,
            )
            legacy_optimizer = torch.optim.AdamW(
                legacy.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                foreach=False,
                fused=False,
            )
            current_optimizer.step()
            legacy_optimizer.step()
            for name in current_parameters:
                self.assertTrue(
                    torch.equal(
                        current_parameters[name],
                        legacy_parameters[name],
                    ),
                    name,
                )

    def test_cpu_float32_matches_legacy_forward_and_update(self):
        self._assert_parity("cpu", torch.float32, compare_update=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_matches_legacy_forward_and_gradients(self):
        self._assert_parity("cuda", torch.bfloat16, compare_update=False)

    def test_prepared_context_keeps_gradient_history(self):
        config = Ideogram4Config(
            emb_dim=24,
            num_layers=1,
            num_heads=3,
            intermediate_size=32,
            adanln_dim=8,
            in_channels=4,
            llm_features_dim=12,
            rope_theta=10_000,
            mrope_section=(1, 1, 1),
        )
        model = Ideogram4Transformer2DModel(config)
        inputs = _inputs(config, device="cpu", dtype=torch.float32)
        llm_features = inputs[1].requires_grad_(True)
        context = model.prepare_context(
            llm_features=llm_features,
            position_ids=inputs[3],
            segment_ids=inputs[4],
            indicator=inputs[5],
        )
        self.assertTrue(context.projected_llm_features.requires_grad)


if __name__ == "__main__":
    unittest.main()
