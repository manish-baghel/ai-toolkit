"""Model-owned, fixed-caption inputs for small Qwen LoRA training datasets."""

from dataclasses import dataclass

import torch

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.lora_special import LoRAModule, LoRASpecialNetwork
from toolkit.print import print_acc

from .pipeline import prepare_run_context


@dataclass(frozen=True)
class QwenTrainingContext:
    run_context: object
    reference_posteriors: tuple
    reference_shapes: tuple


class QwenPreparedTrainingCache:
    """Own GPU tensors for one training run; never cache trainable block outputs."""

    def __init__(self, model, *, data_loader, data_loader_reg, train_config,
                 network_config, network, adapter, embedding, decorator):
        from toolkit.data_loader import get_dataloader_datasets

        datasets = list(get_dataloader_datasets(data_loader)) if data_loader is not None else []
        transformer = unwrap_model(model.transformer)
        reasons = []
        if not datasets or data_loader_reg is not None:
            reasons.append("one training dataset stream without regularization is required")
        if train_config.batch_size != 1 or any(ds.batch_size != 1 for ds in datasets):
            reasons.append("batch_size must be 1")
        if train_config.gradient_accumulation != 1:
            reasons.append("gradient_accumulation must be 1")
        for name in (
            "train_text_encoder", "do_cfg", "do_random_cfg", "single_item_batching",
            "short_and_long_captions", "short_and_long_captions_encoder_split",
            "do_blank_stabilization", "diff_output_preservation",
            "blank_prompt_preservation", "do_guidance_loss", "do_prior_divergence",
            "correct_pred_norm",
        ):
            if getattr(train_config, name, False):
                reasons.append(f"{name} must be disabled")
        if train_config.prompt_dropout_prob or train_config.adapter_assist_name_or_path:
            reasons.append("prompt dropout and adapter assistance are unsupported")
        if train_config.loss_type == "mean_flow" or train_config.timestep_type == "next_sample":
            reasons.append("multi-prediction training requires the ordinary input path")
        if any(value is not None for value in (adapter, embedding, decorator)):
            reasons.append("adapters, embeddings, and decorators are unsupported")
        if model.model_config.compile or model.model_config.layer_offloading or model.model_config.low_vram:
            reasons.append("compile, layer_offloading, and low_vram must be disabled")
        if (
            network_config is None or network_config.type.lower() != "lora"
            or not network_config.transformer_only or type(network) is not LoRASpecialNetwork
        ):
            reasons.append("a standard transformer-only LoRA is required")
        else:
            block_modules = {id(module) for block in transformer.transformer_blocks for module in block.modules()}
            if network.text_encoder_loras or not network.unet_loras or any(
                type(lora) is not LoRAModule or id(lora.orig_module_ref()) not in block_modules
                for lora in network.unet_loras
            ):
                reasons.append("all LoRA modules must be inside transformer blocks")

        for dataset in datasets:
            config = dataset.dataset_config
            if not config.cache_text_embeddings or any(
                not item.is_latent_cached or not item.is_text_embedding_cached
                for item in dataset.file_list
            ):
                reasons.append("target latents and text embeddings must already be cached")
            if config.caption_dropout_rate or config.token_dropout_rate:
                reasons.append("caption_dropout_rate and token_dropout_rate must be 0")
            for name in (
                "random_crop", "random_scale", "shuffle_tokens", "random_triggers",
                "augments", "augmentations", "use_short_captions", "is_reg", "prior_reg",
                "load_image_when_caching_latents", "cache_tensors_to_disk",
                "control_from_same_folder", "controls", "inpaint_path", "mask_path",
                "unconditional_path", "clip_image_path", "alpha_mask",
            ):
                if getattr(config, name, None):
                    reasons.append(f"dataset {name} is unsupported")
            if any(
                item.num_frames != 1 or getattr(item, "dopsd_self_ref", False)
                or getattr(item, "control_video_paths", None)
                for item in dataset.file_list
            ):
                reasons.append("only still images without D-OPSD or video controls are supported")
        if reasons:
            raise ValueError("Cannot enable Qwen prepared training cache: " + "; ".join(dict.fromkeys(reasons)))

        self.entries = {}
        self.file_items = []
        prompt_cache = {}
        vae_device = model.vae.device
        print_acc("Preparing Qwen training inputs on GPU")
        try:
            with torch.no_grad():
                for dataset in datasets:
                    for item in dataset.file_list:
                        # Repeated files may have independently sampled target
                        # latents and different reference preprocessing. A path
                        # hash must not collapse them to the same training input.
                        key = id(item)
                        if key in self.entries:
                            continue
                        latent = item.get_latent().unsqueeze(0).to(model.device_torch, dtype=model.torch_dtype)
                        path = item.get_text_embedding_path()
                        if path not in prompt_cache:
                            prompt_cache[path] = AdvancedPromptEmbeds.load(path).to(model.device_torch, dtype=model.torch_dtype)
                        prompt = prompt_cache[path]
                        padded = model.pad_prompt_embeds(prompt)
                        posteriors, shapes = (), ()
                        if bool(padded[2].any()):
                            posteriors, shapes = self._prepare_references(model, item, latent)
                        context = QwenTrainingContext(
                            prepare_run_context(
                                transformer, *padded, target_shape=tuple(latent.shape[-2:]),
                                condition_shapes=shapes,
                            ),
                            posteriors,
                            shapes,
                        )
                        self.entries[key] = {"latents": latent, "prompt_embeds": prompt, "context": context}
                        self.file_items.append(item)
        finally:
            # A frozen reference VAE is only needed during preparation. Do not
            # make the cache keep it on GPU if the trainer had offloaded it.
            if model.vae.device != vae_device:
                model.vae.to(vae_device)

        # Publish keys only after the entire cache has been prepared successfully.
        for item in self.file_items:
            item._qwen_image_2_prepared_cache_key = id(item)
        print_acc(f"Prepared {len(self.entries)} Qwen inputs ({len(prompt_cache)} unique captions)")

    @staticmethod
    def _prepare_references(model, item, latent):
        try:
            # Use the ordinary loader's EXIF, alpha-compositing and sizing rules.
            item.load_control_image()
            controls = item.control_tensor_list
            if controls is None:
                controls = [] if item.control_tensor is None else [item.control_tensor]
            samples = model._prepare_control_images(
                model._normalize_control_images([controls], 1),
                target_pixels=model._target_pixels((
                    latent.shape[3] * model.vae_scale_factor,
                    latent.shape[2] * model.vae_scale_factor,
                )),
            )
            posteriors = tuple(
                model._encode_image_distribution(
                    [image[0].to(model.device_torch) * 2 - 1],
                    device=model.device_torch, dtype=model.torch_dtype,
                )
                for image in samples[0]
            )
            shapes = tuple(tuple(posterior.mean.shape[-2:]) for posterior in posteriors)
            return posteriors, shapes
        finally:
            item.cleanup_control()

    def get_inputs(self, file_items):
        if len(file_items) != 1:
            raise ValueError("Qwen prepared training cache requires batch_size=1")
        key = getattr(file_items[0], "_qwen_image_2_prepared_cache_key", None)
        if key not in self.entries:
            raise RuntimeError("Qwen prepared training input is missing")
        return self.entries[key]

    def clear(self):
        for item in self.file_items:
            if hasattr(item, "_qwen_image_2_prepared_cache_key"):
                del item._qwen_image_2_prepared_cache_key
        self.file_items.clear()
        self.entries.clear()
