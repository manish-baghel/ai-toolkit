import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import yaml
from safetensors.torch import load_file, save_file

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.models.base_model import BaseModel
from toolkit.lora_special import LoRAModule, LoRASpecialNetwork
from toolkit.basic import flush
from toolkit.print import print_acc
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.ideogram_caption import digest_caption_string
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from toolkit.metadata import get_meta_for_safetensors
from toolkit.memory_management import MemoryManager
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from optimum.quanto import freeze, QTensor

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from transformers import AutoModel, AutoTokenizer

from .src.transformer import Ideogram4Config, Ideogram4Transformer2DModel
from .src.vae import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from .src.latent_norm import get_latent_norm
from .src.pipeline import (
    Ideogram4Pipeline,
    get_qwen3_vl_features,
    pad_text_features,
    patchify_latents,
    predict_velocity,
    predict_velocity_with_prepared_context,
    prepare_velocity_context,
    unpatchify_latents,
)


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Weight-only FP8 (e4m3) Linear weights carry a per-output-channel float32 scale
# saved alongside as ``<name>.weight_scale``. Folding it back gives bf16 weights.
FP8_SCALE_SUFFIX = ".weight_scale"

# The text encoder is frozen, stock Qwen3-VL-8B-Instruct.
QWEN3_VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN", None)


@dataclass
class Ideogram4PreparedTrainingInput:
    latent: torch.Tensor
    prompt_embeds: AdvancedPromptEmbeds
    context: object


def _dequantize_fp8_state_dict(
    state_dict: dict,
    dtype: torch.dtype,
    device: torch.device,
    low_vram: bool,
) -> dict:
    """Fold weight-only FP8 scales back into the weights, casting to ``dtype``.

    Linear weights stored as float8 with a sibling ``.weight_scale`` are
    reconstructed as ``weight_fp8.to(float32) * scale[:, None]``. Everything else
    is simply cast to ``dtype`` (non-floating tensors are left untouched). If the
    checkpoint isn't quantized this is just a dtype cast.

    The fold/cast runs on ``device`` (GPU is much faster than CPU). With
    ``low_vram=True`` each tensor is moved to ``device``, processed, then moved
    back to CPU so the whole bf16 model never sits on the GPU at once; otherwise
    the dequantized tensors are left on ``device`` ready to load.
    """
    work_device = torch.device(device)

    def _finish(t: torch.Tensor) -> torch.Tensor:
        return t.to("cpu") if low_vram else t

    num_fp8 = sum(1 for k in state_dict if k.endswith(FP8_SCALE_SUFFIX))
    if num_fp8 > 0:
        print_acc(f"    dequantizing {num_fp8} fp8 weights -> {dtype} on {work_device}")
    else:
        print_acc(f"    casting weights -> {dtype} on {work_device}")

    out = {}
    for key, tensor in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            continue
        scale_key = key + "_scale"
        if key.endswith(".weight") and scale_key in state_dict:
            w = tensor.to(work_device, torch.float32)
            scale = state_dict[scale_key].to(work_device, torch.float32)
            out[key] = _finish((w * scale.unsqueeze(1)).to(dtype))
        elif tensor.is_floating_point():
            out[key] = _finish(tensor.to(work_device, dtype))
        else:
            out[key] = tensor
    return out


def _validate_predequantized_bf16_state_dict(state_dict: dict) -> None:
    scale_keys = [key for key in state_dict if key.endswith(FP8_SCALE_SUFFIX)]
    if scale_keys:
        raise ValueError(
            "Predequantized Ideogram transformer still contains FP8 scale tensors"
        )
    wrong_dtype_keys = [
        key
        for key, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    ]
    if wrong_dtype_keys:
        raise ValueError(
            "Predequantized Ideogram transformer contains non-bfloat16 weights: "
            + ", ".join(wrong_dtype_keys[:5])
        )


def _load_component_state_dict(
    base: str,
    subfolder: str,
    basename: str,
    *,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a component's weights whether local or on the hub, sharded or single."""
    index_name = f"{basename}.safetensors.index.json"
    single_name = f"{basename}.safetensors"

    # Local directory layout: <base>/<subfolder>/<file>
    local_dir = os.path.join(base, subfolder)
    if os.path.isdir(local_dir):
        index_path = os.path.join(local_dir, index_name)
        if os.path.exists(index_path):
            return _load_sharded(
                local_dir, index_path, is_local=True, device=device
            )
        return load_file(os.path.join(local_dir, single_name), device=str(device))

    # Hub repo layout: <subfolder>/<file>
    prefix = f"{subfolder}/" if subfolder else ""
    try:
        index_path = huggingface_hub.hf_hub_download(
            repo_id=base, filename=f"{prefix}{index_name}", token=HF_TOKEN
        )
        return _load_sharded(
            base,
            index_path,
            is_local=False,
            prefix=prefix,
            device=device,
        )
    except EntryNotFoundError:
        single_path = huggingface_hub.hf_hub_download(
            repo_id=base, filename=f"{prefix}{single_name}", token=HF_TOKEN
        )
        return load_file(single_path, device=str(device))


def _load_sharded(
    base,
    index_path,
    is_local,
    prefix="",
    device: torch.device | str = "cpu",
) -> dict:
    import json

    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    num_shards = len(shard_files)
    for i, shard in enumerate(shard_files):
        if is_local:
            shard_path = os.path.join(base, shard)
        else:
            print_acc(f"    downloading shard {i + 1}/{num_shards}: {shard}")
            shard_path = huggingface_hub.hf_hub_download(
                repo_id=base, filename=f"{prefix}{shard}", token=HF_TOKEN
            )
        print_acc(f"    loading shard {i + 1}/{num_shards}: {shard}")
        state_dict.update(load_file(shard_path, device=str(device)))
    return state_dict


class Ideogram4Model(BaseModel):
    arch = "ideogram4"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.use_old_lokr_format = False
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Ideogram4Transformer2DModel"]

        self.patch_size = 2
        self.vae_scale_factor = 8
        # Safety cap on caption token length (truncation only). Captions are stored
        # per-sample at their natural length and padded to the batch max at the
        # model call, so this is just an upper bound for very long JSON prompts.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 3072)
        )
        self.use_prepared_training_cache = bool(
            self.model_config.model_kwargs.get(
                "cache_prepared_training_inputs", False
            )
        )
        self._prepared_training_cache: dict[
            str, Ideogram4PreparedTrainingInput
        ] = {}
        self._prepared_text_embedding_cache: dict[
            str, AdvancedPromptEmbeds
        ] = {}
        self._prepared_training_file_items: List[object] = []

        self._latent_shift = None
        self._latent_scale = None

        # Optional LoRA that is only switched on during the unconditional (negative)
        # CFG pass. Loaded from model_config.unconditional_lora_path if set; stays
        # inactive everywhere else (training, conditional pass).
        self.unconditional_lora: Optional[LoRASpecialNetwork] = None

    @property
    def text_embedding_space_version(self):
        # we changed the embeddings. invalidate cache.
        return self.arch + "_te_v2"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 8 for the VAE downsample, 2 for the patch size.
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_text_encoder(self, base: str):
        dtype = self.torch_dtype
        # The text encoder is frozen, stock Qwen3-VL-8B-Instruct. The ideogram repo
        # only ships an fp8 copy of it, so load the public bf16 model directly --
        # faster and higher precision than dequantizing the fp8 weights.
        te_path = self.model_config.model_kwargs.get("text_encoder_path", QWEN3_VL_PATH)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")

        tokenizer = AutoTokenizer.from_pretrained(te_path, token=HF_TOKEN)
        text_encoder = AutoModel.from_pretrained(
            te_path, torch_dtype=dtype, token=HF_TOKEN
        )
        flush()

        text_encoder.eval()
        text_encoder.requires_grad_(False)
        return tokenizer, text_encoder

    def _load_transformer(self, base: str):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer")
        use_predequantized_bf16 = bool(
            self.model_config.model_kwargs.get(
                "predequantized_transformer_bf16", False
            )
        )
        if use_predequantized_bf16 and dtype != torch.bfloat16:
            raise ValueError(
                "predequantized_transformer_bf16 requires model dtype bfloat16"
            )

        transformer_config = Ideogram4Config()
        with torch.device("meta"):
            transformer = Ideogram4Transformer2DModel(transformer_config)

        self.print_and_status_update("  - fetching transformer weights")
        load_device = "cpu"
        if use_predequantized_bf16 and not self.model_config.low_vram:
            load_device = self.device_torch
        state_dict = _load_component_state_dict(
            base,
            "transformer",
            "diffusion_pytorch_model",
            device=load_device,
        )
        if use_predequantized_bf16:
            self.print_and_status_update("  - using baked bfloat16 transformer weights")
            _validate_predequantized_bf16_state_dict(state_dict)
        else:
            self.print_and_status_update("  - dequantizing transformer weights")
            state_dict = _dequantize_fp8_state_dict(
                state_dict, dtype, self.device_torch, self.model_config.low_vram
            )
        self.print_and_status_update("  - loading transformer state dict")
        transformer.load_state_dict(state_dict, assign=True)
        del state_dict
        flush()

        # inv_freq is a non-persistent buffer absent from the checkpoint; rebuild
        # it now that the module is off the meta device.
        head_dim = transformer_config.emb_dim // transformer_config.num_heads
        inv_freq = 1.0 / (
            transformer_config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        transformer.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        return transformer

    def _load_vae(self, base: str):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading VAE")
        vae_sd = _load_component_state_dict(base, "vae", "diffusion_pytorch_model")
        vae_sd = convert_diffusers_state_dict(vae_sd)
        vae = AutoEncoder(AutoEncoderParams())
        vae.load_state_dict(vae_sd)
        del vae_sd
        vae.to(self.vae_device_torch, dtype=dtype)
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def load_unconditional_lora(self, transformer: Ideogram4Transformer2DModel):
        """Load the unconditional-pass LoRA and leave it applied but inactive.

        The adapter is wired into the transformer via ``apply_to`` (no merge) so
        the pipeline can flip ``is_active`` on for the unconditional CFG pass only.
        It never affects the conditional pass or training, where it stays inactive.
        """
        lora_path = self.model_config.unconditional_lora_path
        self.print_and_status_update(f"Loading unconditional LoRA from {lora_path}")

        if not os.path.exists(lora_path):
            # assume it is a "repo/owner/filename.safetensors" hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Unconditional LoRA path {lora_path} is not a valid local path "
                    "or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id, filename=filename, token=HF_TOKEN
                )
                self.model_config.unconditional_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download unconditional LoRA from {lora_path}: {e}"
                )

        # Detect the LoRA rank from the first down-projection weight in the file.
        lora_state_dict = load_file(lora_path)
        lora_dim = None
        for key, value in lora_state_dict.items():
            if key.endswith("lora_A.weight") or key.endswith("lora_down.weight"):
                lora_dim = int(value.shape[0])
                break
        if lora_dim is None:
            raise ValueError(
                f"Could not determine LoRA rank from {lora_path}: no lora_A/lora_down "
                "weights found."
            )

        # transformer_only=False so every nn.Linear in the model is targeted (not
        # just the transformer blocks) -- the extraction script factors all linears,
        # so the adapter must wrap all of them to load every key.
        network_config = NetworkConfig(
            type="lora",
            linear=lora_dim,
            linear_alpha=lora_dim,
            transformer_only=False,
        )
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=lora_dim,
            multiplier=1.0,
            alpha=lora_dim,
            # train_unet just gates module creation here; the network is applied,
            # kept inactive, and never trained (the pipeline only toggles is_active).
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type="lora",
            transformer_only=False,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            # base_model_ref lets load_weights run convert_lora_weights_before_load
            # so saved "diffusion_model." keys map back to "transformer.".
            base_model=self,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_path)
        network.eval()

        # Inactive by default; the pipeline flips this on only for the uncond pass.
        network.is_active = False
        self.unconditional_lora = network
        self.print_and_status_update("Unconditional LoRA loaded (inactive)")

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Ideogram4 model")
        base = self.model_config.name_or_path

        transformer = self._load_transformer(base)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[
                    transformer.rotary_emb.inv_freq,
                    transformer.input_proj,
                    transformer.llm_cond_proj,
                ],
            )
        elif self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            # quantize_model leaves the model on CPU; make sure it lands on device.
            transformer.to(self.device_torch)
        flush()

        tokenizer, text_encoder = self._load_text_encoder(base)
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            text_encoder.to(self.device_torch)
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
        elif self.model_config.low_vram:
            self.print_and_status_update("Moving text encoder to CPU")
            text_encoder.to("cpu")
        else:
            self.print_and_status_update("Moving text encoder to device")
            text_encoder.to(self.device_torch)
        flush()

        vae = self._load_vae(base)

        self.noise_scheduler = Ideogram4Model.get_train_scheduler()

        shift, scale = get_latent_norm()
        self._latent_shift = shift.view(1, -1, 1, 1)
        self._latent_scale = scale.view(1, -1, 1, 1)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer
        self.pipeline = Ideogram4Pipeline(self)

        if self.model_config.unconditional_lora_path is not None:
            self.load_unconditional_lora(transformer)

        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Ideogram4Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Ideogram4Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        )[0]
        return img

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def _prepared_training_cache_incompatibilities(
        self,
        *,
        data_loader,
        data_loader_reg,
        train_config,
        network_config,
        network,
        adapter,
        embedding,
        decorator,
    ) -> List[str]:
        reasons = []
        transformer = unwrap_model(self.transformer)
        transformer_block_module_ids = {
            id(module)
            for block in transformer.layers
            for module in block.modules()
        }
        if data_loader is None:
            reasons.append("a training dataset is required")
        if data_loader_reg is not None:
            reasons.append("regularization datasets are not supported")
        if train_config.batch_size != 1:
            reasons.append("batch_size must be 1")
        if train_config.gradient_accumulation != 1:
            reasons.append("gradient_accumulation must be 1")
        if train_config.train_text_encoder:
            reasons.append("text encoder training must be disabled")
        if train_config.do_cfg or train_config.do_random_cfg:
            reasons.append("training CFG must be disabled")
        if train_config.short_and_long_captions:
            reasons.append("short_and_long_captions must be disabled")
        if train_config.short_and_long_captions_encoder_split:
            reasons.append("caption encoder splitting must be disabled")
        if train_config.single_item_batching:
            reasons.append("single_item_batching must be disabled")
        if train_config.prompt_dropout_prob != 0.0:
            reasons.append("prompt_dropout_prob must be 0")
        if train_config.adapter_assist_name_or_path is not None:
            reasons.append("adapter assistance is not supported")
        if adapter is not None or embedding is not None or decorator is not None:
            reasons.append("adapters, embeddings, and decorators are not supported")
        if self.model_config.compile:
            reasons.append("torch.compile must be disabled")
        if self.model_config.layer_offloading:
            reasons.append("layer offloading must be disabled")

        if network_config is None or network is None:
            reasons.append("a LoRA network is required")
        else:
            if network_config.type.lower() != "lora":
                reasons.append("only the standard LoRA network is supported")
            if not network_config.transformer_only:
                reasons.append("network.transformer_only must be true")
            if type(network) is not LoRASpecialNetwork:
                reasons.append("only LoRASpecialNetwork is supported")
            else:
                if network.text_encoder_loras:
                    reasons.append("text encoder LoRA modules are not supported")
                if not network.unet_loras:
                    reasons.append("the LoRA network has no transformer modules")
                if any(
                    type(lora) is not LoRAModule
                    or lora.orig_module_ref() is None
                    or id(lora.orig_module_ref()) not in transformer_block_module_ids
                    for lora in network.unet_loras
                ):
                    reasons.append("all LoRA modules must be inside transformer layers")

        static_modules = (
            transformer.llm_cond_norm,
            transformer.llm_cond_proj,
            transformer.embed_image_indicator,
            transformer.rotary_emb,
        )
        if any(
            parameter.requires_grad
            for module in static_modules
            for parameter in module.parameters()
        ):
            reasons.append("prepared conditioning modules must be frozen")

        if data_loader is not None:
            from toolkit.data_loader import get_dataloader_datasets

            for dataset in get_dataloader_datasets(data_loader):
                config = dataset.dataset_config
                if not config.cache_latents:
                    reasons.append("all datasets must cache latents in memory")
                if not config.cache_text_embeddings:
                    reasons.append("all datasets must cache text embeddings")
                if config.load_image_when_caching_latents:
                    reasons.append("raw image loading with cached latents is unsupported")
                if config.random_crop or config.random_scale:
                    reasons.append("random crop and random scale are unsupported")
                if config.shuffle_tokens or config.random_triggers:
                    reasons.append("randomized caption processing is unsupported")
                if config.augments or config.augmentations:
                    reasons.append("dataset augmentations are unsupported")
                if any(
                    getattr(config, name, None) is not None
                    for name in (
                        "control_path",
                        "inpaint_path",
                        "mask_path",
                        "unconditional_path",
                        "clip_image_path",
                    )
                ):
                    reasons.append("auxiliary dataset images are unsupported")
                if config.use_short_captions:
                    reasons.append("short captions are unsupported")
                if any(
                    not item.is_latent_cached or not item.is_text_embedding_cached
                    for item in dataset.file_list
                ):
                    reasons.append("dataset caches must be populated before preparation")

        return list(dict.fromkeys(reasons))

    def prepare_training_cache(
        self,
        *,
        data_loader,
        data_loader_reg,
        train_config,
        network_config,
        network,
        adapter,
        embedding,
        decorator,
    ) -> None:
        if not self.use_prepared_training_cache:
            return

        reasons = self._prepared_training_cache_incompatibilities(
            data_loader=data_loader,
            data_loader_reg=data_loader_reg,
            train_config=train_config,
            network_config=network_config,
            network=network,
            adapter=adapter,
            embedding=embedding,
            decorator=decorator,
        )
        if reasons:
            raise ValueError(
                "Cannot enable Ideogram prepared training cache: "
                + "; ".join(reasons)
            )

        from toolkit.data_loader import get_dataloader_datasets

        transformer = unwrap_model(self.transformer)
        prepared_cache = {}
        text_embedding_cache = {}
        assignments = []
        print_acc("Preparing Ideogram training inputs on GPU")

        with torch.no_grad():
            for dataset in get_dataloader_datasets(data_loader):
                for file_item in dataset.file_list:
                    latent_path = file_item.get_latent_path(recalculate=True)
                    text_embedding_path = file_item.get_text_embedding_path(
                        recalculate=True
                    )
                    cache_key = hashlib.sha256(
                        f"{latent_path}\0{text_embedding_path}".encode("utf-8")
                    ).hexdigest()
                    assignments.append((file_item, cache_key))
                    if cache_key in prepared_cache:
                        continue

                    prompt_embeds = text_embedding_cache.get(text_embedding_path)
                    if prompt_embeds is None:
                        prompt_embeds = AdvancedPromptEmbeds.load(
                            text_embedding_path
                        ).to(self.device_torch, dtype=self.torch_dtype)
                        text_embedding_cache[text_embedding_path] = prompt_embeds

                    latent = file_item.get_latent().unsqueeze(0).to(
                        self.device_torch, dtype=self.torch_dtype
                    )
                    _, _, gh, gw = latent.shape
                    llm_features, text_mask = pad_text_features(
                        prompt_embeds.text_embeds,
                        self.device_torch,
                        self.torch_dtype,
                    )
                    context = prepare_velocity_context(
                        transformer,
                        llm_features,
                        text_mask,
                        gh,
                        gw,
                        detach=True,
                    )
                    prepared_cache[cache_key] = Ideogram4PreparedTrainingInput(
                        latent=latent,
                        prompt_embeds=prompt_embeds,
                        context=context,
                    )

        # Publish only after every item prepared successfully. Until this point
        # the dataloader remains on its original CPU-cache path.
        self._prepared_training_cache = prepared_cache
        self._prepared_text_embedding_cache = text_embedding_cache
        self._prepared_training_file_items = [
            file_item for file_item, _ in assignments
        ]
        for file_item, cache_key in assignments:
            file_item._ideogram4_prepared_cache_key = cache_key
        print_acc(
            f"Prepared {len(prepared_cache)} Ideogram training inputs "
            f"({len(text_embedding_cache)} unique text embeddings) on GPU"
        )

    def get_prepared_training_inputs(self, file_items):
        if not self.use_prepared_training_cache:
            return None
        if len(file_items) != 1:
            raise ValueError("Ideogram prepared training cache requires batch_size=1")
        cache_key = getattr(
            file_items[0], "_ideogram4_prepared_cache_key", None
        )
        if cache_key is None or cache_key not in self._prepared_training_cache:
            raise RuntimeError("Ideogram prepared training input is missing")
        prepared = self._prepared_training_cache[cache_key]
        return {
            "latents": prepared.latent,
            "prompt_embeds": prepared.prompt_embeds,
            "context": prepared.context,
        }

    def clear_prepared_training_cache(self) -> None:
        for file_item in self._prepared_training_file_items:
            if hasattr(file_item, "_ideogram4_prepared_cache_key"):
                del file_item._ideogram4_prepared_cache_key
        self._prepared_training_file_items.clear()
        self._prepared_training_cache.clear()
        self._prepared_text_embedding_cache.clear()
        self.use_prepared_training_cache = False

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 128, gh, gw)
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch=None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        t01 = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t01.dim() == 0:
            t01 = t01.unsqueeze(0)
        if t01.shape[0] != latent_model_input.shape[0]:
            t01 = t01.expand(latent_model_input.shape[0])

        prepared_context = (
            getattr(batch, "prepared_training_context", None)
            if is_primary_pred and batch is not None
            else None
        )
        if prepared_context is not None:
            pred = predict_velocity_with_prepared_context(
                unwrap_model(self.transformer),
                latent_model_input.to(self.device_torch),
                t01,
                prepared_context,
            )
        else:
            # Pad the per-sample caption features to the batch max here.
            llm_features, text_mask = pad_text_features(
                text_embeddings.text_embeds, self.device_torch, self.torch_dtype
            )

            pred = predict_velocity(
                self.transformer,
                latent_model_input.to(self.device_torch),
                t01,
                llm_features,
                text_mask,
            )
        return pred

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        device = self.text_encoder.device

        # Encode each caption at its natural length (no cross-sample padding) and
        # store one feature tensor per batch item. Padding to a common length is
        # deferred to the model call, so caching a prompt only stores its real
        # length -- important for the long structured (JSON) captions.
        features_list = []
        for p in prompt:
            # Digest the prompt: migrate any old-format Ideogram caption into the
            # current schema and serialize it compact (the form the renderer wants).
            # Plain-text prompts pass straight through unchanged.
            p = digest_caption_string(p)
            messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_length,
            )["input_ids"]
            if len(ids) == 0:
                ids = [self.tokenizer.eos_token_id or 0]

            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(token_ids)
            pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

            features = get_qwen3_vl_features(
                self.text_encoder, token_ids, attention_mask, pos_2d
            )  # (1, Lt, D)
            features_list.append(features[0].to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=features_list)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)

        if isinstance(image_list, list):
            images = torch.stack(image_list, dim=0)
        else:
            images = image_list
        images = images.to(device, dtype=dtype)

        ae_channels = self.vae.params.z_channels
        moments = self.vae.encoder(images)
        mean = moments[:, :ae_channels]

        patched = patchify_latents(mean, self.patch_size)
        shift = self._latent_shift.to(patched.device, patched.dtype)
        scale = self._latent_scale.to(patched.device, patched.dtype)
        latents = (patched - shift) / scale
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)

        latents = latents.to(device, dtype=dtype)
        shift = self._latent_shift.to(device, dtype)
        scale = self._latent_scale.to(device, dtype)
        patched = latents * scale + shift
        z = unpatchify_latents(patched, self.patch_size)
        images = self.vae.decoder(z)
        return images

    # ------------------------------------------------------------------
    # Saving / misc
    # ------------------------------------------------------------------
    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: Ideogram4Transformer2DModel = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)
        meta = get_meta_for_safetensors(meta, name="ideogram4")
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "ideogram4"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
