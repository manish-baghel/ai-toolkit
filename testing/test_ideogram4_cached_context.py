"""Training parity for reusable Ideogram 4 velocity contexts.

Run directly::

    PYTHONPATH=. python -m unittest -v testing.test_ideogram4_cached_context
"""

import copy
import unittest

import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
    predict_velocity,
    predict_velocity_with_prepared_context,
    prepare_velocity_context,
)
from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
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


def _config():
    return Ideogram4Config(
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


def _freeze_outside_blocks(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("layers."))


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


class Ideogram4CachedContextParityTest(unittest.TestCase):
    def _assert_two_step_parity(self, device, dtype, compare_update):
        torch.manual_seed(29)
        config = _config()
        fresh = Ideogram4Transformer2DModel(config).to(device=device, dtype=dtype)
        cached = copy.deepcopy(fresh)
        _freeze_outside_blocks(fresh)
        _freeze_outside_blocks(cached)

        llm_features = _tensor(
            (1, 3, config.llm_features_dim),
            -0.3,
            0.4,
            device=device,
            dtype=dtype,
        )
        text_mask = torch.ones(1, 3, dtype=torch.long, device=device)
        prepared = prepare_velocity_context(
            cached,
            llm_features,
            text_mask,
            2,
            3,
            detach=True,
        )

        context = prepared.transformer_context
        for value in (
            context.projected_llm_features,
            context.output_image_mask,
            context.image_indicator_embedding,
            context.cos,
            context.sin,
            context.attn_mask,
        ):
            if value is not None:
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)

        fresh_optimizer = torch.optim.AdamW(
            _trainable_parameters(fresh),
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )
        cached_optimizer = torch.optim.AdamW(
            _trainable_parameters(cached),
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )

        for step in range(2):
            fresh_optimizer.zero_grad(set_to_none=True)
            cached_optimizer.zero_grad(set_to_none=True)
            fresh_latents = _tensor(
                (1, config.in_channels, 2, 3),
                -0.45 + step * 0.03,
                0.55 + step * 0.03,
                device=device,
                dtype=dtype,
            ).requires_grad_(True)
            cached_latents = fresh_latents.detach().clone().requires_grad_(True)
            timestep = torch.tensor(
                [0.25 + step * 0.2], device=device, dtype=dtype
            )

            fresh_output = predict_velocity(
                fresh,
                fresh_latents,
                timestep,
                llm_features,
                text_mask,
            )
            cached_output = predict_velocity_with_prepared_context(
                cached,
                cached_latents,
                timestep,
                prepared,
            )
            self.assertTrue(torch.equal(fresh_output, cached_output))

            target = _tensor(
                fresh_output.shape,
                -0.2,
                0.25,
                device=device,
                dtype=torch.float32,
            )
            fresh_loss = F.mse_loss(fresh_output, target)
            cached_loss = F.mse_loss(cached_output, target)
            self.assertTrue(torch.equal(fresh_loss, cached_loss))
            fresh_loss.backward()
            cached_loss.backward()
            self.assertTrue(torch.equal(fresh_latents.grad, cached_latents.grad))

            fresh_parameters = _trainable_parameters(fresh)
            cached_parameters = _trainable_parameters(cached)
            for fresh_parameter, cached_parameter in zip(
                fresh_parameters, cached_parameters
            ):
                self.assertTrue(
                    torch.equal(fresh_parameter.grad, cached_parameter.grad)
                )

            if compare_update:
                fresh_optimizer.step()
                cached_optimizer.step()
                for fresh_parameter, cached_parameter in zip(
                    fresh_parameters, cached_parameters
                ):
                    self.assertTrue(torch.equal(fresh_parameter, cached_parameter))
                    fresh_state = fresh_optimizer.state[fresh_parameter]
                    cached_state = cached_optimizer.state[cached_parameter]
                    self.assertEqual(fresh_state.keys(), cached_state.keys())
                    for key in fresh_state:
                        self.assertTrue(
                            torch.equal(fresh_state[key], cached_state[key]), key
                        )

    def test_cpu_context_reuse_matches_two_training_updates(self):
        self._assert_two_step_parity("cpu", torch.float32, compare_update=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_context_reuse_matches_gradients(self):
        self._assert_two_step_parity("cuda", torch.bfloat16, compare_update=False)

    def test_static_work_runs_once_and_dynamic_work_runs_per_prediction(self):
        config = _config()
        model = Ideogram4Transformer2DModel(config)
        llm_features = torch.zeros(1, 3, config.llm_features_dim)
        text_mask = torch.ones(1, 3, dtype=torch.long)
        counts = {"llm": 0, "input": 0, "block": 0}
        handles = [
            model.llm_cond_proj.register_forward_hook(
                lambda *_: counts.__setitem__("llm", counts["llm"] + 1)
            ),
            model.input_proj.register_forward_hook(
                lambda *_: counts.__setitem__("input", counts["input"] + 1)
            ),
            model.layers[0].register_forward_hook(
                lambda *_: counts.__setitem__("block", counts["block"] + 1)
            ),
        ]
        try:
            prepared = prepare_velocity_context(
                model, llm_features, text_mask, 2, 3, detach=True
            )
            for timestep in (0.25, 0.75):
                predict_velocity_with_prepared_context(
                    model,
                    torch.zeros(1, config.in_channels, 2, 3),
                    torch.tensor([timestep]),
                    prepared,
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, {"llm": 1, "input": 2, "block": 2})

    def test_prepared_context_validates_backend_and_oriented_geometry(self):
        config = _config()
        model = Ideogram4Transformer2DModel(config)
        llm_features = torch.zeros(1, 3, config.llm_features_dim)
        text_mask = torch.ones(1, 3, dtype=torch.long)
        prepared = prepare_velocity_context(
            model, llm_features, text_mask, 2, 3, detach=True
        )
        with self.assertRaises(ValueError):
            predict_velocity_with_prepared_context(
                model,
                torch.zeros(1, config.in_channels, 3, 2),
                torch.tensor([0.5]),
                prepared,
            )
        prepared.attention_backend = "different"
        with self.assertRaises(ValueError):
            predict_velocity_with_prepared_context(
                model,
                torch.zeros(1, config.in_channels, 2, 3),
                torch.tensor([0.5]),
                prepared,
            )


if __name__ == "__main__":
    unittest.main()
