"""Parity tests for Ideogram 4 static sequence packing.

Run directly::

    PYTHONPATH=. python -m unittest -v testing.test_ideogram4_packed_context
"""

import copy
import unittest

import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
    pack_latent_tokens,
    predict_velocity,
    predict_velocity_with_context,
    prepare_packed_context,
)
from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Config,
    Ideogram4Transformer2DModel,
)


def _tensor(shape, start, end, *, device, dtype):
    return torch.linspace(
        start,
        end,
        steps=torch.tensor(shape).prod().item(),
        device=device,
        dtype=torch.float32,
    ).reshape(shape).to(dtype)


def _legacy_pack(llm_features, text_mask, gh, gw):
    device = llm_features.device
    b, num_text_tokens, llm_dim = llm_features.shape
    num_image_tokens = gh * gw
    seq_len = num_text_tokens + num_image_tokens
    text_mask_bool = text_mask.to(device) > 0
    text_mask_long = text_mask_bool.long()

    llm_full = torch.cat(
        [
            llm_features,
            torch.zeros(
                b,
                num_image_tokens,
                llm_dim,
                device=device,
                dtype=llm_features.dtype,
            ),
        ],
        dim=1,
    )
    indicator = torch.zeros(b, seq_len, dtype=torch.long, device=device)
    indicator[:, :num_text_tokens] = text_mask_long * LLM_TOKEN_INDICATOR
    indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    segment_ids = torch.ones(b, seq_len, dtype=torch.long, device=device)
    segment_ids[:, :num_text_tokens] = torch.where(
        text_mask_bool,
        torch.ones_like(text_mask_long),
        torch.full_like(text_mask_long, SEQUENCE_PADDING_INDICATOR),
    )

    text_pos = (text_mask_long.cumsum(dim=-1) - 1).clamp(min=0)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)
    h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1)
    image_pos = image_pos + IMAGE_POSITION_OFFSET
    position_ids = torch.cat(
        [text_pos_3d, image_pos.unsqueeze(0).expand(b, -1, -1)], dim=1
    )
    return llm_full, position_ids, segment_ids, indicator


def _legacy_latent_tokens(latents, num_text_tokens):
    b, c, gh, gw = latents.shape
    image_tokens = latents.permute(0, 2, 3, 1).reshape(b, gh * gw, c)
    return torch.cat(
        [
            torch.zeros(
                b,
                num_text_tokens,
                c,
                device=latents.device,
                dtype=image_tokens.dtype,
            ),
            image_tokens,
        ],
        dim=1,
    )


class Ideogram4PackedContextParityTest(unittest.TestCase):
    def _assert_parity(self, device, dtype, compare_update):
        torch.manual_seed(23)
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

        batch_size, num_text_tokens, gh, gw = 2, 3, 2, 3
        current_llm = _tensor(
            (batch_size, num_text_tokens, config.llm_features_dim),
            -0.3,
            0.4,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        legacy_llm = current_llm.detach().clone().requires_grad_(True)
        text_mask = torch.tensor(
            [[1, 1, 1], [1, 1, 0]], device=device, dtype=torch.long
        )
        current_latents = _tensor(
            (batch_size, config.in_channels, gh, gw),
            -0.45,
            0.55,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        legacy_latents = current_latents.detach().clone().requires_grad_(True)
        timestep = torch.tensor([0.25, 0.7], device=device, dtype=dtype)

        context = prepare_packed_context(current_llm, text_mask, gh, gw)
        legacy_context = _legacy_pack(legacy_llm, text_mask, gh, gw)
        for current_value, legacy_value in zip(
            (
                context.llm_features,
                context.position_ids,
                context.segment_ids,
                context.indicator,
            ),
            legacy_context,
        ):
            self.assertTrue(torch.equal(current_value, legacy_value))
        self.assertTrue(
            torch.equal(
                pack_latent_tokens(current_latents, num_text_tokens),
                _legacy_latent_tokens(current_latents, num_text_tokens),
            )
        )

        current_output = predict_velocity_with_context(
            current,
            current_latents,
            timestep,
            context,
        )
        legacy_x = _legacy_latent_tokens(legacy_latents, num_text_tokens)
        legacy_full_output = legacy(
            llm_features=legacy_context[0],
            x=legacy_x,
            t=1.0 - timestep,
            position_ids=legacy_context[1],
            segment_ids=legacy_context[2],
            indicator=legacy_context[3],
        )
        legacy_output = -legacy_full_output[:, num_text_tokens:].reshape(
            batch_size, gh, gw, config.in_channels
        ).permute(0, 3, 1, 2)
        self.assertTrue(torch.equal(current_output, legacy_output))

        target = _tensor(
            current_output.shape,
            -0.2,
            0.25,
            device=device,
            dtype=torch.float32,
        )
        current_loss = F.mse_loss(current_output, target)
        legacy_loss = F.mse_loss(legacy_output, target)
        self.assertTrue(torch.equal(current_loss, legacy_loss))
        current_loss.backward()
        legacy_loss.backward()
        self.assertTrue(torch.equal(current_latents.grad, legacy_latents.grad))
        self.assertTrue(torch.equal(current_llm.grad, legacy_llm.grad))

        current_parameters = dict(current.named_parameters())
        legacy_parameters = dict(legacy.named_parameters())
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
                current.parameters(), lr=1e-3, foreach=False, fused=False
            )
            legacy_optimizer = torch.optim.AdamW(
                legacy.parameters(), lr=1e-3, foreach=False, fused=False
            )
            current_optimizer.step()
            legacy_optimizer.step()
            for name in current_parameters:
                self.assertTrue(
                    torch.equal(current_parameters[name], legacy_parameters[name]),
                    name,
                )

    def test_cpu_float32_packing_matches_legacy_update(self):
        self._assert_parity("cpu", torch.float32, compare_update=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_packing_matches_legacy_gradients(self):
        self._assert_parity("cuda", torch.bfloat16, compare_update=False)

    def test_context_rejects_wrong_latent_geometry(self):
        config = Ideogram4Config(
            emb_dim=24,
            num_layers=1,
            num_heads=3,
            intermediate_size=32,
            adanln_dim=8,
            in_channels=4,
            llm_features_dim=12,
            mrope_section=(1, 1, 1),
        )
        model = Ideogram4Transformer2DModel(config)
        llm = torch.zeros(1, 3, config.llm_features_dim)
        mask = torch.ones(1, 3, dtype=torch.long)
        context = prepare_packed_context(llm, mask, 2, 3)
        # Same token count, transposed geometry: positions must not be reused.
        wrong_latents = torch.zeros(1, config.in_channels, 3, 2)
        with self.assertRaises(ValueError):
            predict_velocity_with_context(
                model,
                wrong_latents,
                torch.tensor([0.5]),
                context,
            )

    def test_public_wrapper_matches_explicit_context_path(self):
        config = Ideogram4Config(
            emb_dim=24,
            num_layers=1,
            num_heads=3,
            intermediate_size=32,
            adanln_dim=8,
            in_channels=4,
            llm_features_dim=12,
            mrope_section=(1, 1, 1),
        )
        model = Ideogram4Transformer2DModel(config)
        llm = torch.zeros(1, 3, config.llm_features_dim)
        mask = torch.ones(1, 3, dtype=torch.long)
        latents = torch.zeros(1, config.in_channels, 2, 2)
        timestep = torch.tensor([0.5])
        context = prepare_packed_context(llm, mask, 2, 2)
        direct = predict_velocity_with_context(model, latents, timestep, context)
        wrapped = predict_velocity(model, latents, timestep, llm, mask)
        self.assertTrue(torch.equal(direct, wrapped))


if __name__ == "__main__":
    unittest.main()
