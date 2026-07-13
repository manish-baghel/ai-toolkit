"""Lifecycle and parity coverage for Ideogram 4's prepared training cache."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
    pad_text_features,
    prepare_velocity_context,
)
from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    Ideogram4Config,
    Ideogram4Transformer2DModel,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import NetworkConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.lora_special import LoRASpecialNetwork


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


def _model_shell(transformer, device, dtype):
    model = Ideogram4Model.__new__(Ideogram4Model)
    model.model = transformer
    model.device_torch = torch.device(device)
    model.torch_dtype = dtype
    model.use_prepared_training_cache = True
    model._prepared_training_cache = {}
    model._prepared_text_embedding_cache = {}
    return model


class _LoRABaseModel:
    use_old_lokr_format = False

    @staticmethod
    def get_transformer_block_names():
        return ["layers"]


def _attach_lora(transformer, device):
    network_config = NetworkConfig(
        type="lora",
        linear=4,
        linear_alpha=4,
        transformer_only=True,
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=4,
        multiplier=1.0,
        alpha=4,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type="lora",
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=["Ideogram4Transformer2DModel"],
        base_model=_LoRABaseModel(),
    )
    network.apply_to(None, transformer, False, True)
    network.force_to(device, torch.float32)
    network._update_torch_multiplier()
    network.is_active = True
    return network


class _LoaderFileItem:
    def __init__(self):
        self._encoded_latent = torch.ones(4, 2, 3)
        self._cached_first_frame_latent = None
        self._cached_audio_latent = None
        self._ideogram4_prepared_cache_key = "cache-key"
        self.mutable_metadata = ["original"]
        self.caption_loaded = False

    def load_caption(self, caption_dict):
        self.caption_loaded = True

    def load_and_process_image(self, transform):
        raise AssertionError("prepared items must not load image data")


class _CacheFileItem:
    def __init__(self, latent_path, text_path, latent):
        self._latent_path = latent_path
        self._text_path = text_path
        self._latent = latent

    def get_latent_path(self, recalculate=False):
        return self._latent_path

    def get_text_embedding_path(self, recalculate=False):
        return self._text_path

    def get_latent(self):
        return self._latent


def _dto_file_item():
    item = SimpleNamespace(
        is_latent_cached=True,
        dataset_config=SimpleNamespace(load_image_when_caching_latents=False),
        num_frames=1,
        extra_values=[],
        audio_data=None,
        _cached_first_frame_latent=None,
        _cached_audio_latent=None,
        control_tensor=None,
        control_tensor_list=None,
        inpaint_tensor=None,
        clip_image_tensor=None,
        mask_tensor=None,
        unaugmented_tensor=None,
        unconditional_tensor=None,
        clip_image_embeds=None,
        clip_image_embeds_unconditional=None,
        prompt_embeds=None,
        audio_tensor=None,
        loss_multiplier=1.0,
    )
    item.get_latent = lambda: (_ for _ in ()).throw(
        AssertionError("prepared batches must not read the item latent")
    )
    item.cleanup = lambda: None
    return item


class Ideogram4TrainingCacheTest(unittest.TestCase):
    def test_production_shape_configuration_is_cache_eligible(self):
        transformer = Ideogram4Transformer2DModel(_config())
        transformer.requires_grad_(False)
        model = _model_shell(transformer, "cpu", torch.float32)
        model.model_config = SimpleNamespace(
            compile=False,
            layer_offloading=False,
        )
        network = _attach_lora(transformer, "cpu")
        dataset_config = SimpleNamespace(
            cache_latents=True,
            cache_text_embeddings=True,
            load_image_when_caching_latents=False,
            random_crop=False,
            random_scale=False,
            augments=[],
            augmentations=None,
            control_path=None,
            inpaint_path=None,
            mask_path=None,
            unconditional_path=None,
            clip_image_path=None,
            use_short_captions=False,
            shuffle_tokens=False,
            random_triggers=[],
        )
        dataset = SimpleNamespace(
            dataset_config=dataset_config,
            file_list=[
                SimpleNamespace(
                    is_latent_cached=True,
                    is_text_embedding_cached=True,
                )
            ],
        )
        data_loader = SimpleNamespace(
            dataset=SimpleNamespace(datasets=[dataset])
        )
        train_config = SimpleNamespace(
            batch_size=1,
            gradient_accumulation=1,
            train_text_encoder=False,
            do_cfg=False,
            do_random_cfg=False,
            short_and_long_captions=False,
            short_and_long_captions_encoder_split=False,
            single_item_batching=False,
            prompt_dropout_prob=0.0,
            adapter_assist_name_or_path=None,
        )

        reasons = model._prepared_training_cache_incompatibilities(
            data_loader=data_loader,
            data_loader_reg=None,
            train_config=train_config,
            network_config=network.network_config,
            network=network,
            adapter=None,
            embedding=None,
            decorator=None,
        )

        self.assertEqual(reasons, [])

    def test_prepared_loader_copy_does_not_clone_or_retain_cpu_latent(self):
        source = _LoaderFileItem()
        dataset = SimpleNamespace(file_list=[source], caption_dict={})

        copied = AiToolkitDataset._get_single_item(dataset, 0)

        self.assertIsNot(copied, source)
        self.assertIsNone(copied._encoded_latent)
        self.assertIsNotNone(source._encoded_latent)
        self.assertIsNot(copied.mutable_metadata, source.mutable_metadata)
        self.assertTrue(copied.caption_loaded)

    def test_batch_borrows_model_owned_inputs_without_loading_or_mutating_them(self):
        latent = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
        prompt_embeds = AdvancedPromptEmbeds(
            text_embeds=[torch.ones(3, 12)]
        )
        context = object()
        original_latent = latent.clone()
        original_prompt = prompt_embeds.text_embeds[0].clone()

        batch = DataLoaderBatchDTO(
            file_items=[_dto_file_item()],
            prepared_training_inputs={
                "latents": latent,
                "prompt_embeds": prompt_embeds,
                "context": context,
            },
        )

        self.assertIs(batch.latents, latent)
        self.assertIs(batch.prompt_embeds, prompt_embeds)
        self.assertIs(batch.prepared_training_context, context)
        batch.cleanup()
        self.assertTrue(torch.equal(latent, original_latent))
        self.assertTrue(torch.equal(prompt_embeds.text_embeds[0], original_prompt))

    def test_cache_build_deduplicates_text_and_publishes_atomically(self):
        transformer = Ideogram4Transformer2DModel(_config())
        transformer.requires_grad_(False)
        model = _model_shell(transformer, "cpu", torch.float32)
        items = [
            _CacheFileItem("latent-a", "text-a", torch.zeros(4, 2, 3)),
            _CacheFileItem("latent-b", "text-a", torch.ones(4, 3, 2)),
        ]
        data_loader = SimpleNamespace(
            dataset=SimpleNamespace(
                datasets=[SimpleNamespace(file_list=items)]
            )
        )
        prompt_embeds = AdvancedPromptEmbeds(
            text_embeds=[torch.zeros(3, _config().llm_features_dim)]
        )

        with patch.object(
            model,
            "_prepared_training_cache_incompatibilities",
            return_value=[],
        ), patch.object(
            AdvancedPromptEmbeds, "load", return_value=prompt_embeds
        ) as load_prompt:
            model.prepare_training_cache(
                data_loader=data_loader,
                data_loader_reg=None,
                train_config=None,
                network_config=None,
                network=None,
                adapter=None,
                embedding=None,
                decorator=None,
            )

        self.assertEqual(load_prompt.call_count, 1)
        self.assertEqual(len(model._prepared_training_cache), 2)
        self.assertEqual(len(model._prepared_text_embedding_cache), 1)
        self.assertTrue(
            all(
                hasattr(item, "_ideogram4_prepared_cache_key") for item in items
            )
        )
        for prepared in model._prepared_training_cache.values():
            self.assertEqual(prepared.latent.device.type, "cpu")
            for value in (
                prepared.context.transformer_context.projected_llm_features,
                prepared.context.transformer_context.output_image_mask,
                prepared.context.transformer_context.image_indicator_embedding,
                prepared.context.transformer_context.cos,
                prepared.context.transformer_context.sin,
            ):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
        model.clear_prepared_training_cache()
        self.assertFalse(model.use_prepared_training_cache)
        self.assertEqual(model._prepared_training_cache, {})
        self.assertTrue(
            all(
                not hasattr(item, "_ideogram4_prepared_cache_key")
                for item in items
            )
        )

    def _assert_prediction_parity(self, device, dtype):
        torch.manual_seed(41)
        config = _config()
        transformer = Ideogram4Transformer2DModel(config).to(
            device=device, dtype=dtype
        )
        model = _model_shell(transformer, device, dtype)
        features = torch.linspace(
            -0.3,
            0.4,
            steps=3 * config.llm_features_dim,
            device=device,
            dtype=torch.float32,
        ).reshape(3, config.llm_features_dim).to(dtype)
        prompt_embeds = AdvancedPromptEmbeds(text_embeds=[features])
        llm_features, text_mask = pad_text_features(
            prompt_embeds.text_embeds, torch.device(device), dtype
        )
        prepared = prepare_velocity_context(
            transformer, llm_features, text_mask, 2, 3, detach=True
        )
        batch = SimpleNamespace(prepared_training_context=prepared)
        latents = torch.linspace(
            -0.5,
            0.5,
            steps=config.in_channels * 2 * 3,
            device=device,
            dtype=torch.float32,
        ).reshape(1, config.in_channels, 2, 3).to(dtype)
        timestep = torch.tensor([370.0], device=device)

        fresh = model.get_noise_prediction(
            latents,
            timestep,
            prompt_embeds,
            batch=batch,
            is_primary_pred=False,
        )
        cached = model.get_noise_prediction(
            latents,
            timestep,
            prompt_embeds,
            batch=batch,
            is_primary_pred=True,
        )
        self.assertTrue(torch.equal(fresh, cached))

    def _assert_lora_update_parity(self, device, dtype):
        torch.manual_seed(47)
        config = _config()
        fresh_transformer = Ideogram4Transformer2DModel(config).to(
            device=device, dtype=dtype
        )
        cached_transformer = copy.deepcopy(fresh_transformer)
        fresh_transformer.requires_grad_(False)
        cached_transformer.requires_grad_(False)
        torch.manual_seed(53)
        fresh_network = _attach_lora(fresh_transformer, device)
        torch.manual_seed(53)
        cached_network = _attach_lora(cached_transformer, device)
        cached_network.load_state_dict(fresh_network.state_dict())

        features = torch.linspace(
            -0.25,
            0.35,
            steps=3 * config.llm_features_dim,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 3, config.llm_features_dim).to(dtype)
        text_mask = torch.ones(1, 3, dtype=torch.long, device=device)
        prepared = prepare_velocity_context(
            cached_transformer, features, text_mask, 2, 3, detach=True
        )
        fresh_latents = torch.linspace(
            -0.45,
            0.55,
            steps=config.in_channels * 2 * 3,
            device=device,
            dtype=torch.float32,
        ).reshape(1, config.in_channels, 2, 3).to(dtype)
        cached_latents = fresh_latents.clone()
        timestep = torch.tensor([0.37], device=device, dtype=dtype)

        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
            predict_velocity,
            predict_velocity_with_prepared_context,
        )

        fresh_output = predict_velocity(
            fresh_transformer,
            fresh_latents,
            timestep,
            features,
            text_mask,
        )
        cached_output = predict_velocity_with_prepared_context(
            cached_transformer,
            cached_latents,
            timestep,
            prepared,
        )
        self.assertTrue(torch.equal(fresh_output, cached_output))
        target = torch.linspace(
            -0.2,
            0.3,
            steps=fresh_output.numel(),
            device=device,
            dtype=torch.float32,
        ).reshape_as(fresh_output)
        fresh_loss = torch.nn.functional.mse_loss(fresh_output, target)
        cached_loss = torch.nn.functional.mse_loss(cached_output, target)
        self.assertTrue(torch.equal(fresh_loss, cached_loss))
        fresh_loss.backward()
        cached_loss.backward()

        fresh_parameters = dict(fresh_network.named_parameters())
        cached_parameters = dict(cached_network.named_parameters())
        self.assertEqual(fresh_parameters.keys(), cached_parameters.keys())
        for name in fresh_parameters:
            self.assertTrue(
                torch.equal(
                    fresh_parameters[name].grad,
                    cached_parameters[name].grad,
                ),
                name,
            )

        fresh_optimizer = torch.optim.AdamW(
            fresh_parameters.values(),
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )
        cached_optimizer = torch.optim.AdamW(
            cached_parameters.values(),
            lr=1e-3,
            weight_decay=1e-4,
            foreach=False,
            fused=False,
        )
        fresh_optimizer.step()
        cached_optimizer.step()
        for name in fresh_parameters:
            self.assertTrue(
                torch.equal(fresh_parameters[name], cached_parameters[name]), name
            )

    def test_cpu_prediction_is_exact(self):
        self._assert_prediction_parity("cpu", torch.float32)

    def test_cpu_lora_gradients_and_update_are_exact(self):
        self._assert_lora_update_parity("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_prediction_is_exact(self):
        self._assert_prediction_parity("cuda", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_lora_gradients_and_update_are_exact(self):
        self._assert_lora_update_parity("cuda", torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
