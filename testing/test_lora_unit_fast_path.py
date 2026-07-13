"""Parity coverage for the standard LoRA unit-scale training path.

Run directly because AI Toolkit's ``testing`` directory contains a mixture of
standalone scripts that are not safe for automatic test discovery::

    PYTHONPATH=. python -m unittest -v testing.test_lora_unit_fast_path
"""

import unittest

import torch
import torch.nn.functional as F

from toolkit.lora_special import LoRAModule


class _FakeNetwork:
    def __init__(self, multiplier):
        self.network_type = "lora"
        self.is_active = True
        self.is_lorm = False
        self.is_merged_in = False
        self._multiplier = multiplier
        self.torch_multiplier = torch.tensor(multiplier, dtype=torch.float32).reshape(-1)


def _deterministic_weight(shape, start, end):
    return torch.linspace(start, end, steps=torch.tensor(shape).prod().item()).reshape(shape)


def _build_lora(alpha=3, multiplier=1.0):
    base = torch.nn.Linear(5, 4, bias=False)
    base.weight.requires_grad_(False)
    with torch.no_grad():
        base.weight.copy_(_deterministic_weight(base.weight.shape, -0.3, 0.4))

    network = _FakeNetwork(multiplier)
    lora = LoRAModule(
        "test_lora",
        base,
        multiplier=multiplier,
        lora_dim=3,
        alpha=alpha,
        network=network,
    )
    with torch.no_grad():
        lora.lora_down.weight.copy_(
            _deterministic_weight(lora.lora_down.weight.shape, -0.2, 0.25)
        )
        lora.lora_up.weight.copy_(
            _deterministic_weight(lora.lora_up.weight.shape, -0.15, 0.3)
        )
    lora.apply_to()
    return base, lora, network


def _legacy_forward(x, base_weight, down_weight, up_weight, alpha, multiplier):
    """Reproduce ToolkitModuleMixin's pre-optimization arithmetic exactly."""
    base_output = F.linear(x, base_weight)
    lora_output = F.linear(F.linear(x, down_weight), up_weight)

    scalar = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = (alpha / down_weight.shape[0]) * scalar
    lora_output = lora_output * scale

    multiplier_tensor = torch.tensor(multiplier, device=x.device, dtype=x.dtype).reshape(-1)
    if lora_output.shape[0] != multiplier_tensor.shape[0]:
        repeats = lora_output.shape[0] // multiplier_tensor.shape[0]
        multiplier_tensor = multiplier_tensor.repeat_interleave(repeats)
    while multiplier_tensor.dim() < lora_output.dim():
        multiplier_tensor = multiplier_tensor.unsqueeze(-1)

    return base_output + (lora_output * multiplier_tensor).to(base_output.dtype)


class LoRAUnitFastPathParityTest(unittest.TestCase):
    def _assert_training_step_parity(self, alpha, multiplier, batch_size=1, exact=True):
        torch.manual_seed(7)
        base, lora, _ = _build_lora(alpha=alpha, multiplier=multiplier)

        actual_x = _deterministic_weight((batch_size, 7, 5), -0.4, 0.5).requires_grad_(True)
        reference_x = actual_x.detach().clone().requires_grad_(True)
        target = _deterministic_weight((batch_size, 7, 4), -0.25, 0.35)

        reference_down = torch.nn.Parameter(lora.lora_down.weight.detach().clone())
        reference_up = torch.nn.Parameter(lora.lora_up.weight.detach().clone())

        actual_optimizer = torch.optim.AdamW(
            [lora.lora_down.weight, lora.lora_up.weight],
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )
        reference_optimizer = torch.optim.AdamW(
            [reference_down, reference_up],
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )

        actual_output = base(actual_x)
        reference_output = _legacy_forward(
            reference_x,
            base.weight.detach(),
            reference_down,
            reference_up,
            alpha,
            multiplier,
        )
        actual_loss = F.mse_loss(actual_output, target)
        reference_loss = F.mse_loss(reference_output, target)

        compare = torch.equal if exact else lambda left, right: torch.allclose(
            left, right, rtol=1e-6, atol=1e-7
        )
        self.assertTrue(compare(actual_output.detach(), reference_output.detach()))
        self.assertTrue(compare(actual_loss.detach(), reference_loss.detach()))

        actual_loss.backward()
        reference_loss.backward()

        self.assertTrue(compare(actual_x.grad, reference_x.grad))
        self.assertTrue(compare(lora.lora_down.weight.grad, reference_down.grad))
        self.assertTrue(compare(lora.lora_up.weight.grad, reference_up.grad))

        actual_optimizer.step()
        reference_optimizer.step()

        self.assertTrue(compare(lora.lora_down.weight, reference_down))
        self.assertTrue(compare(lora.lora_up.weight, reference_up))

        for actual_parameter, reference_parameter in zip(
            [lora.lora_down.weight, lora.lora_up.weight],
            [reference_down, reference_up],
        ):
            actual_state = actual_optimizer.state[actual_parameter]
            reference_state = reference_optimizer.state[reference_parameter]
            self.assertEqual(actual_state.keys(), reference_state.keys())
            for key in actual_state:
                self.assertTrue(compare(actual_state[key], reference_state[key]))

    def test_unit_scale_and_multiplier_are_exact(self):
        self._assert_training_step_parity(alpha=3, multiplier=1.0, exact=True)

    def test_non_unit_scale_retains_legacy_behavior(self):
        self._assert_training_step_parity(alpha=1.5, multiplier=1.0, exact=True)

    def test_non_unit_multiplier_retains_legacy_behavior(self):
        self._assert_training_step_parity(alpha=3, multiplier=0.5, exact=True)

    def test_per_sample_multiplier_retains_legacy_behavior(self):
        self._assert_training_step_parity(
            alpha=3,
            multiplier=[1.0, 0.5],
            batch_size=2,
            exact=True,
        )


if __name__ == "__main__":
    unittest.main()
