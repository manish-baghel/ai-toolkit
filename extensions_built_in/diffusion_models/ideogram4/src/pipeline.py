"""Packing / sampling helpers for Ideogram 4.

This module holds the glue that turns image latents + Qwen3-VL text features into
the single packed sequence the transformer consumes, plus a minimal flow-matching
sampling pipeline used to render preview images during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from transformers.masking_utils import create_causal_mask

from .transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)

_LOGSNR_MIN = -15.0
_LOGSNR_MAX = 18.0


def _logit_normal_schedule(
    u: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Reference Ideogram time schedule, where 0 is noise and 1 is clean."""
    u = torch.as_tensor(u, dtype=torch.float64)
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    return t.clamp(t_min, t_max)


def get_ideogram4_sigmas(
    num_steps: int,
    width: int,
    height: int,
    mu: float = 0.0,
    std: float = 1.75,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the resolution-aware sigma schedule used by ComfyUI/Ideogram."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if std <= 0:
        raise ValueError("std must be positive")

    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
    sigmas = (1.0 - _logit_normal_schedule(u, mean, std)).flip(0)
    sigmas[-1] = 0.0
    return sigmas.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Latent (un)patchification.
#
# The VAE produces (B, ae_ch=32, H/8, W/8) latents. The transformer works on
# tokens of dim ae_ch * patch**2 = 128. We store the patchified latent in a 4-D
# (B, 128, gh, gw) layout so the rest of ai-toolkit (noise, add_noise, loss) can
# treat it like an ordinary image latent. The channel ordering here matches the
# reference Ideogram 4 decode exactly: 128 = (patch_h, patch_w, ae_ch) with ae_ch
# the fastest-varying axis.
# ---------------------------------------------------------------------------


def patchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch, H8, W8) -> (B, ae_ch * patch**2, gh, gw)."""
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    # -> (B, ph, pw, ae_ch, gh, gw) then merge (ph, pw, ae_ch) -> channels
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


def unpatchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch * patch**2, gh, gw) -> (B, ae_ch, H8, W8)."""
    b, c, gh, gw = z.shape
    ph = pw = patch_size
    ae_ch = c // (ph * pw)
    z = z.view(b, ph, pw, ae_ch, gh, gw)
    # -> (B, ae_ch, gh, ph, gw, pw) then merge spatial
    z = z.permute(0, 3, 4, 1, 5, 2).reshape(b, ae_ch, gh * ph, gw * pw)
    return z


# ---------------------------------------------------------------------------
# Qwen3-VL hidden-state extraction.
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_qwen3_vl_features(
    text_encoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3-VL and concat the hidden states from the activation layers.

    Returns a (B, L, hidden_size * num_layers) tensor (in the encoder's dtype),
    zeroed at non-text (padding) positions.
    """
    language_model = text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    batch_size, seq_len = token_ids.shape
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # (B, L, H, num_taps)
    stacked = stacked.reshape(batch_size, seq_len, -1)

    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked


# ---------------------------------------------------------------------------
# Packing + velocity prediction.
# ---------------------------------------------------------------------------


@dataclass
class Ideogram4PackedContext:
    """Latent/timestep-independent inputs for a packed Ideogram sequence."""

    llm_features: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    indicator: torch.Tensor
    num_text_tokens: int
    num_image_tokens: int
    gh: int
    gw: int


def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample (Lt_i, D) features into a batch.

    Captions are stored at their natural length (one tensor per batch item) and
    only padded to the batch max here, right before the model call. Returns
    ``(features (B, Lt, D), attention_mask (B, Lt))``; the mask is 1 for real
    tokens and 0 for padding (which the transformer masks out anyway).
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        features[i, :n] = f.to(device, dtype)
        mask[i, :n] = 1
    return features, mask


def prepare_packed_context(
    llm_features: torch.Tensor,
    text_mask: torch.Tensor,
    gh: int,
    gw: int,
) -> Ideogram4PackedContext:
    """Build the static [text | image] conditioning and sequence geometry."""
    device = llm_features.device
    b, num_text_tokens, llm_dim = llm_features.shape
    num_image_tokens = gh * gw
    seq_len = num_text_tokens + num_image_tokens

    # The mask may arrive as a float (PromptEmbeds.to casts it to the embed
    # dtype); work in long so cumsum positions stay exact for long prompts.
    text_mask_bool = text_mask.to(device) > 0
    text_mask_long = text_mask_bool.long()

    # LLM features: image region is zero.
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

    # Indicator: real text -> 3, image -> 2, text pad -> 0.
    indicator = torch.zeros(b, seq_len, dtype=torch.long, device=device)
    indicator[:, :num_text_tokens] = text_mask_long * LLM_TOKEN_INDICATOR
    indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    # Segment ids: real text + image -> 1, text pad -> -1.
    segment_ids = torch.ones(b, seq_len, dtype=torch.long, device=device)
    segment_ids[:, :num_text_tokens] = torch.where(
        text_mask_bool,
        torch.ones_like(text_mask_long),
        torch.full_like(text_mask_long, SEQUENCE_PADDING_INDICATOR),
    )

    # Position ids (t, h, w). Text uses relative positions; image positions use
    # a large offset so they cannot collide with text positions.
    text_pos = (text_mask_long.cumsum(dim=-1) - 1).clamp(min=0)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)

    h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    image_pos_3d = image_pos.unsqueeze(0).expand(b, -1, -1)
    position_ids = torch.cat([text_pos_3d, image_pos_3d], dim=1)

    return Ideogram4PackedContext(
        llm_features=llm_full,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
        num_text_tokens=num_text_tokens,
        num_image_tokens=num_image_tokens,
        gh=gh,
        gw=gw,
    )


def pack_latent_tokens(
    latents: torch.Tensor,
    num_text_tokens: int,
) -> torch.Tensor:
    """Pack dynamic image latents after a zeroed text-token prefix."""
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


def predict_velocity(
    transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,  # (B, 128, gh, gw)
    t: torch.Tensor,  # (B,) toolkit flow time in [0, 1] (1 = pure noise)
    llm_features: torch.Tensor,  # (B, Lt, llm_dim)
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
) -> torch.Tensor:
    """Run the transformer on the packed [text | image] sequence.

    ``t`` is in the ai-toolkit flow-matching convention: ``t=1`` is pure noise,
    ``t=0`` is clean, and the returned velocity is ``noise - clean`` (matching the
    toolkit scheduler / loss target).

    Ideogram's transformer uses the opposite convention internally (``t=1`` is
    clean) and predicts ``clean - noise``, so we feed it ``1 - t`` and negate its
    output. Returns the velocity reshaped to the (B, 128, gh, gw) latent layout.
    """
    _, _, gh, gw = latents.shape
    packed_context = prepare_packed_context(llm_features, text_mask, gh, gw)
    return predict_velocity_with_context(transformer, latents, t, packed_context)


def predict_velocity_with_context(
    transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,
    t: torch.Tensor,
    packed_context: Ideogram4PackedContext,
) -> torch.Tensor:
    """Predict velocity while reusing an already packed static context."""
    b, c, gh, gw = latents.shape
    if packed_context.gh != gh or packed_context.gw != gw:
        raise ValueError("Packed Ideogram context does not match latent geometry")
    if packed_context.llm_features.shape[0] != b:
        raise ValueError("Packed Ideogram context does not match latent batch size")

    x = pack_latent_tokens(latents, packed_context.num_text_tokens)

    # Flip into the model's time convention (t=1 -> clean).
    model_t = 1.0 - t

    out = transformer(
        llm_features=packed_context.llm_features,
        x=x,
        t=model_t,
        position_ids=packed_context.position_ids,
        segment_ids=packed_context.segment_ids,
        indicator=packed_context.indicator,
    )

    image_velocity = out[:, packed_context.num_text_tokens:]  # (B, Li, 128)
    image_velocity = image_velocity.reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    # Model predicts clean - noise; negate to return toolkit velocity (noise - clean).
    return -image_velocity


# ---------------------------------------------------------------------------
# Minimal sampling pipeline (for training previews).
# ---------------------------------------------------------------------------


class Ideogram4Pipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the Ideogram4Model so we can reuse its encode/decode and
        # latent helpers without duplicating state.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        patch = model.patch_size

        schedule_mu = float(
            model.model_config.model_kwargs.get("ideogram_schedule_mu", 0.0)
        )
        schedule_std = float(
            model.model_config.model_kwargs.get("ideogram_schedule_std", 1.75)
        )
        sigmas = get_ideogram4_sigmas(
            num_inference_steps,
            width,
            height,
            mu=schedule_mu,
            std=schedule_std,
            device=device,
        )

        ae_scale = model.vae_scale_factor  # 8
        gh = height // (ae_scale * patch)
        gw = width // (ae_scale * patch)
        latent_channels = transformer.config.in_channels

        # Ideogram uses asymmetric CFG: the unconditional branch is image-only
        # (no text tokens) with zeroed text features -- it does NOT run a negative
        # prompt through the text encoder. So we ignore unconditional_embeds and
        # build an empty (0-length) text sequence for the uncond pass below.
        do_cfg = guidance_scale > 1.0

        if latents is None:
            shape = (1, latent_channels, gh, gw)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        latents = latents.to(device, dtype=torch.float32)
        latents = latents * sigmas[0]

        cond_feats, cond_mask = pad_text_features(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg:
            # Image-only unconditional: zero-length text sequence. predict_velocity
            # then produces an image-token-only forward pass with zeroed llm
            # features, matching the reference's asymmetric CFG.
            batch_size = latents.shape[0]
            text_dim = cond_feats.shape[-1]
            uncond_feats = torch.zeros(
                batch_size, 0, text_dim, device=device, dtype=dtype
            )
            uncond_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # The unconditional LoRA (if present) must be active *only* on the
        # unconditional pass. We force it off before each conditional pass since the
        # outer sampling context (``with network:``) may switch it on globally.
        uncond_lora = getattr(model, "unconditional_lora", None)

        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            t01 = sigma.expand(latents.shape[0])
            if uncond_lora is not None:
                uncond_lora.is_active = False
            v_cond = predict_velocity(
                transformer, latents.to(dtype), t01, cond_feats, cond_mask
            )
            if do_cfg:
                if uncond_lora is not None:
                    uncond_lora.is_active = True
                try:
                    v_uncond = predict_velocity(
                        transformer, latents.to(dtype), t01, uncond_feats, uncond_mask
                    )
                finally:
                    if uncond_lora is not None:
                        uncond_lora.is_active = False
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + v.to(torch.float32) * (sigma_next - sigma)

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
