"""Opt-in GPU inputs for fixed-caption, small-dataset Krea training."""

from dataclasses import dataclass

import torch

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.print import print_acc

from .pipeline import pad_text_features, prepare_velocity_context


@dataclass(frozen=True)
class KreaTrainingContext:
    features: torch.Tensor
    text_mask: torch.Tensor
    velocity_context: object


class KreaPreparedTrainingCache:
    """Cache raw inputs and geometry, never trainable text-fusion outputs."""

    def __init__(self, model, *, data_loader, data_loader_reg, train_config,
                 network_config, network, adapter, embedding, decorator):
        from toolkit.data_loader import get_dataloader_datasets

        datasets = list(get_dataloader_datasets(data_loader)) if data_loader is not None else []
        reasons = []
        if not datasets or data_loader_reg is not None:
            reasons.append("a training dataset stream without regularization is required")
        if train_config.batch_size != 1 or any(ds.batch_size != 1 for ds in datasets):
            reasons.append("batch_size must be 1")
        if train_config.gradient_accumulation != 1 or train_config.gradient_accumulation_steps != 1:
            reasons.append("gradient_accumulation and gradient_accumulation_steps must be 1")
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
            network_config is None or network is None
            or network_config.type.lower() not in {"lora", "lokr"}
            or not network_config.transformer_only
        ):
            reasons.append("a transformer-only LoRA or LoKr is required")

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
                "unconditional_path", "clip_image_path", "clip_image_from_same_folder",
                "alpha_mask",
            ):
                if getattr(config, name, None):
                    reasons.append(f"dataset {name} is unsupported")
            for item in dataset.file_list:
                if item.num_frames != 1 or any(getattr(item, name, False) for name in (
                    "dopsd_self_ref", "control_video_paths", "has_clip_image",
                    "has_mask_image", "has_inpaint_image", "has_unconditional",
                    "is_video", "is_audio_model",
                )) or item._cached_first_frame_latent is not None:
                    reasons.append("only still images without auxiliary image/video inputs are supported")
        if reasons:
            raise ValueError("Cannot enable Krea prepared training cache: " + "; ".join(dict.fromkeys(reasons)))

        self.entries = {}
        self.file_items = []
        prompt_cache = {}
        transformer = unwrap_model(model.transformer)
        print_acc("Preparing Krea training inputs on GPU")
        with torch.no_grad():
            for dataset in datasets:
                for item in dataset.file_list:
                    # Repeated paths can have independently sampled target
                    # latents. Retain each original dataset item's latent.
                    key = id(item)
                    if key in self.entries:
                        continue
                    latent = item.get_latent().unsqueeze(0).to(model.device_torch, dtype=model.torch_dtype)
                    path = item.get_text_embedding_path()
                    if path not in prompt_cache:
                        prompt = AdvancedPromptEmbeds.load(path).to(model.device_torch, dtype=model.torch_dtype)
                        features, mask = pad_text_features(prompt.text_embeds, model.device_torch, model.torch_dtype)
                        prompt_cache[path] = (prompt, features, mask)
                    prompt, features, mask = prompt_cache[path]
                    # Edits retain the ordinary geometry/reference path: control
                    # images are loaded per draw and their VAE posterior is
                    # sampled afresh by get_noise_prediction, just as before.
                    geometry = None if model.is_edit else prepare_velocity_context(
                        transformer, latent, features, mask, isolate_refs=model.kv_cache,
                    )
                    context = KreaTrainingContext(features, mask, geometry)
                    self.entries[key] = {"latents": latent, "prompt_embeds": prompt, "context": context}
                    self.file_items.append(item)

        # Failed preparation must not leave a partially enabled loader path.
        for item in self.file_items:
            item._krea2_prepared_cache_key = id(item)
        print_acc(f"Prepared {len(self.entries)} Krea inputs ({len(prompt_cache)} unique captions)")

    def get_inputs(self, file_items):
        if len(file_items) != 1:
            raise ValueError("Krea prepared training cache requires batch_size=1")
        key = getattr(file_items[0], "_krea2_prepared_cache_key", None)
        if key not in self.entries:
            raise RuntimeError("Krea prepared training input is missing")
        return self.entries[key]

    def clear(self):
        for item in self.file_items:
            if hasattr(item, "_krea2_prepared_cache_key"):
                del item._krea2_prepared_cache_key
        self.file_items.clear()
        self.entries.clear()
