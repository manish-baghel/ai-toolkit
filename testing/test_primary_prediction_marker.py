import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds


class _Embeds:
    def to(self, *args, **kwargs):
        return self


class _Model:
    def __init__(self):
        self.kwargs = None

    def predict_noise(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros_like(kwargs["latents"])


class _Scheduler:
    def scale_model_input(self, latent, timestep):
        return latent


class _DeviceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class _BaseModelHarness(BaseModel):
    def get_noise_prediction(
        self,
        latent_model_input,
        timestep,
        text_embeddings,
        is_primary_pred=False,
        **kwargs,
    ):
        self.received_primary_marker = is_primary_pred
        return torch.zeros_like(latent_model_input)


class PrimaryPredictionMarkerTest(unittest.TestCase):
    def test_trainer_forwards_primary_prediction_marker(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.device_torch = torch.device("cpu")
        trainer.train_config = SimpleNamespace(
            dtype="float32",
            cfg_scale=1.0,
            do_guidance_loss=False,
            cfg_rescale=1.0,
            bypass_guidance_embedding=False,
        )
        trainer.sd = _Model()

        trainer.predict_noise(
            noisy_latents=torch.zeros(1, 4, 2, 2),
            timesteps=torch.ones(1),
            conditional_embeds=_Embeds(),
            is_primary_pred=True,
        )

        self.assertIs(trainer.sd.kwargs["is_primary_pred"], True)

    def test_base_model_only_forwards_marker_to_supporting_model_hook(self):
        model = _BaseModelHarness.__new__(_BaseModelHarness)
        model.model = _DeviceModule()
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.noise_scheduler = _Scheduler()

        model.predict_noise(
            latents=torch.zeros(1, 4, 2, 2),
            text_embeddings=PromptEmbeds(torch.zeros(1, 2, 3)),
            timestep=torch.ones(1),
            guidance_scale=1.0,
            is_primary_pred=True,
        )

        self.assertIs(model.received_primary_marker, True)


if __name__ == "__main__":
    unittest.main()
