import unittest
from unittest import mock

import torch

from toolkit.kohya_lora import LoRAModule as KohyaLoRAModule
from toolkit.lora_special import LoRAModule
from toolkit.lycoris_special import LoConSpecialModule
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule
from toolkit.network_mixins import (
    ToolkitNetworkMixin,
    _is_fixed_unit_multiplier,
    broadcast_and_multiply,
)


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0


class _MultiplierNetwork(ToolkitNetworkMixin):
    network_type = "lora"

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.unet_loras = []


def _linear(device=None, dtype=None):
    return torch.nn.Linear(8, 8, bias=False, device=device, dtype=dtype)


class AdapterScaleTest(unittest.TestCase):
    def test_adapters_keep_float_metadata_and_nonpersistent_runtime_buffer(self):
        network = _Network()
        modules = [
            LoRAModule(
                "lora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            KohyaLoRAModule(
                "kohya_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
            ),
            LoConSpecialModule(
                "locon_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            DoRAModule(
                "dora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            LokrModule(
                "lokr_scale",
                _linear(),
                lora_dim=2,
                alpha=torch.tensor(4, dtype=torch.bfloat16),
                network=network,
            ),
        ]

        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertIs(type(module.scale), float)
                self.assertEqual(module._runtime_scale.item(), module.scale)
                self.assertNotIn("_runtime_scale", module.state_dict())
                self.assertFalse(module._runtime_scale.requires_grad)

    def test_extract_weight_synchronizes_runtime_scale(self):
        for alpha in (4, 8):
            with self.subTest(alpha=alpha):
                module = LoRAModule(
                    "extract_scale",
                    _linear(),
                    lora_dim=4,
                    alpha=torch.tensor(alpha, dtype=torch.bfloat16),
                    network=_Network(),
                )
                self.assertEqual(module._has_fixed_unit_scale, alpha == 4)
                runtime_scale = module._runtime_scale
                down = torch.randn(2, 8)
                up = torch.randn(8, 2)

                with mock.patch(
                    "toolkit.network_mixins.extract_linear",
                    return_value=(down, up, 2, None),
                ):
                    module.extract_weight(extract_mode="fixed", extract_mode_param=2)

                self.assertIs(module._runtime_scale, runtime_scale)
                self.assertEqual(module.scale, 1.0)
                self.assertEqual(module._runtime_scale.item(), 1.0)
                self.assertFalse(module._has_fixed_unit_scale)

    def test_unit_scale_fast_path_returns_projection_without_multiplying(self):
        module = LoRAModule(
            "unit_scale",
            _linear(),
            lora_dim=4,
            alpha=4,
            network=_Network(),
        )
        projections = []
        hook = module.lora_up.register_forward_hook(
            lambda _module, _args, output: projections.append(output)
        )
        try:
            output = module._call_forward(torch.randn(2, 8))
        finally:
            hook.remove()

        self.assertTrue(module._has_fixed_unit_scale)
        self.assertIs(output, projections[0])

    def test_runtime_scale_updates_disable_unit_fast_path(self):
        module = LoRAModule(
            "updated_scale",
            _linear(),
            lora_dim=4,
            alpha=4,
            network=_Network(),
        )
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 8)
        unit_output = module._call_forward(value).detach()
        runtime_scale = module._runtime_scale

        self.assertTrue(module._has_fixed_unit_scale)
        for scale in (0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                module._set_runtime_scale(scale)
                self.assertFalse(module._has_fixed_unit_scale)
                self.assertIs(module._runtime_scale, runtime_scale)
                self.assertEqual(module.scale, scale)
                self.assertEqual(module._runtime_scale.item(), scale)
                torch.testing.assert_close(
                    module._call_forward(value), unit_output * scale
                )

    def test_compiled_unit_scale_switches_to_runtime_buffer_once(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        module = LoRAModule(
            "compiled_unit_scale",
            _linear(),
            lora_dim=4,
            alpha=4,
            network=_Network(),
        )
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 8)
        unit_output = module._call_forward(value).detach()
        graphs = []

        def backend(graph, _example_inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(module._call_forward, backend=backend, fullgraph=True)
        torch.testing.assert_close(compiled(value), unit_output)
        self.assertEqual(len(graphs), 1)

        # Leaving the fixed-unit shortcut changes the graph once. All later
        # scale changes use the same runtime buffer without recompilation.
        for scale in (0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                module._set_runtime_scale(scale)
                torch.testing.assert_close(compiled(value), unit_output * scale)
                self.assertEqual(len(graphs), 2)

    def test_unit_scale_fast_path_preserves_rank_dropout_compensation(self):
        module = LoRAModule(
            "rank_dropout_scale",
            _linear(),
            lora_dim=4,
            alpha=4,
            rank_dropout=0.5,
            network=_Network(),
        )
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 8)
        unit_output = module.lora_up(module.lora_down(value))

        # Keep every rank so the dropout compensation itself is observable.
        with mock.patch(
            "toolkit.network_mixins.torch.rand", return_value=torch.ones(2, 4)
        ):
            torch.testing.assert_close(module._call_forward(value), unit_output * 2)
        module.eval()
        torch.testing.assert_close(module._call_forward(value), unit_output)

    def test_trainable_scalar_never_uses_unit_fast_path(self):
        module = LoConSpecialModule(
            "trainable_scale",
            _linear(),
            lora_dim=4,
            alpha=4,
            network=_Network(),
        )
        with torch.no_grad():
            module.scalar.fill_(0.25)
        module._set_runtime_scale(1.0)
        value = torch.randn(2, 8)
        expected = module.lora_up(module.lora_down(value)) * module.scalar

        self.assertFalse(getattr(module, "_has_fixed_unit_scale", False))
        torch.testing.assert_close(module._call_forward(value), expected)

    def test_fixed_unit_multiplier_detection_uses_host_values_only(self):
        for multiplier in (1, 1.0, [1.0, 1.0], [[1.0], [1.0]]):
            with self.subTest(multiplier=multiplier):
                self.assertTrue(_is_fixed_unit_multiplier(multiplier))
        for multiplier in (0.5, [], [1.0, 0.5], torch.ones(2)):
            with self.subTest(multiplier=multiplier):
                self.assertFalse(_is_fixed_unit_multiplier(multiplier))

    def test_multiplier_updates_switch_identity_fast_path(self):
        network = _MultiplierNetwork()
        original = _linear()
        module = LoRAModule(
            "multiplier_scale", original, lora_dim=4, alpha=4, network=network
        )
        module.org_forward = original.forward
        network.unet_loras = [module]
        network._update_torch_multiplier()
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(4, 8)
        base = original(value)
        delta = module._call_forward(value)

        for multiplier, is_one, expected in (
            (1.0, True, [1.0, 1.0, 1.0, 1.0]),
            ([1.0, 1.0], True, [1.0, 1.0, 1.0, 1.0]),
            ([0.5, 1.0], False, [0.5, 0.5, 1.0, 1.0]),
            ([1.0, 1.0], True, [1.0, 1.0, 1.0, 1.0]),
        ):
            with self.subTest(multiplier=multiplier):
                network.multiplier = multiplier
                self.assertEqual(network._multiplier_is_one, is_one)
                with mock.patch(
                    "toolkit.network_mixins.broadcast_and_multiply",
                    wraps=broadcast_and_multiply,
                ) as multiply:
                    output = module(value)
                self.assertEqual(multiply.call_count, 0 if is_one else 1)
                torch.testing.assert_close(
                    output, base + delta * torch.tensor(expected).unsqueeze(-1)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dynamic_compile_stays_cuda_and_scale_updates_do_not_recompile(self):
        torch.manual_seed(0)
        torch._dynamo.reset()
        from torch._inductor import metrics

        metrics.reset()
        network = _Network()
        network.torch_multiplier = torch.ones(1, device="cuda")
        original = _linear(device="cuda", dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = LoRAModule(
            "compiled_scale",
            original,
            lora_dim=4,
            alpha=torch.tensor(8, dtype=torch.bfloat16),
            network=network,
        ).to("cuda")
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        eager = module(value)
        eager.square().mean().backward()
        eager_down_grad = module.lora_down.weight.grad.detach().clone()
        eager_up_grad = module.lora_up.weight.grad.detach().clone()
        module.zero_grad(set_to_none=True)

        compiled = torch.compile(module, fullgraph=False, dynamic=True)
        actual = compiled(value)
        actual.square().mean().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, eager, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(
            module.lora_down.weight.grad, eager_down_grad, rtol=2e-2, atol=5e-2
        )
        torch.testing.assert_close(
            module.lora_up.weight.grad, eager_up_grad, rtol=2e-2, atol=5e-2
        )
        self.assertEqual(
            getattr(metrics, "generated_cpp_vec_kernel_count", 0), 0
        )
        self.assertEqual(module._runtime_scale.device.type, "cuda")

        base = original(value)
        original_delta = actual - base
        generated_kernels = metrics.generated_kernel_count
        module._set_runtime_scale(0.5)
        updated = compiled(value)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            updated - base, original_delta * 0.25, rtol=2e-2, atol=5e-2
        )
        self.assertEqual(metrics.generated_kernel_count, generated_kernels)


if __name__ == "__main__":
    unittest.main()
