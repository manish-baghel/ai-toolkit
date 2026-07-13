import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.ideogram4.ideogram4 import (
    _load_component_state_dict,
    _validate_predequantized_bf16_state_dict,
)


class Ideogram4Bf16CheckpointTest(unittest.TestCase):
    def test_validator_accepts_only_fully_bfloat16_floating_weights(self):
        _validate_predequantized_bf16_state_dict(
            {
                "weight": torch.ones(2, 3, dtype=torch.bfloat16),
                "indices": torch.ones(2, dtype=torch.int64),
            }
        )
        with self.assertRaises(ValueError):
            _validate_predequantized_bf16_state_dict(
                {
                    "weight": torch.ones(2, 3, dtype=torch.bfloat16),
                    "weight_scale": torch.ones(2),
                }
            )
        with self.assertRaises(ValueError):
            _validate_predequantized_bf16_state_dict(
                {"weight": torch.ones(2, 3, dtype=torch.float32)}
            )

    def test_local_single_file_loads_on_requested_device(self):
        self._assert_local_load("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_local_single_file_loads_directly_on_cuda(self):
        self._assert_local_load("cuda")

    def _assert_local_load(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp) / "transformer"
            component_dir.mkdir()
            expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            save_file(
                {"weight": expected},
                component_dir / "diffusion_pytorch_model.safetensors",
            )
            loaded = _load_component_state_dict(
                tmp,
                "transformer",
                "diffusion_pytorch_model",
                device=device,
            )
            self.assertEqual(loaded["weight"].device.type, device)
            self.assertTrue(torch.equal(loaded["weight"].cpu(), expected))

    def test_local_shards_load_on_requested_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp) / "transformer"
            component_dir.mkdir()
            first_name = "diffusion_pytorch_model-00001-of-00002.safetensors"
            second_name = "diffusion_pytorch_model-00002-of-00002.safetensors"
            save_file(
                {"first": torch.ones(2, dtype=torch.bfloat16)},
                component_dir / first_name,
            )
            save_file(
                {"second": torch.zeros(2, dtype=torch.bfloat16)},
                component_dir / second_name,
            )
            (component_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "first": first_name,
                            "second": second_name,
                        },
                    }
                )
            )
            loaded = _load_component_state_dict(
                tmp,
                "transformer",
                "diffusion_pytorch_model",
                device="cpu",
            )
            self.assertEqual(set(loaded), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
