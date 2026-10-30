# Qwen Image 2.1 training performance

The standard path now shares SDPA segment masks across transformer blocks and
gradient-checkpoint recomputations. The toolkit wrapper requests only target
image outputs, pruning the final normalization/projection of discarded prefix
tokens when the output head is unpatched and deterministic. The transformer's
default full-sequence API is unchanged.

For small LoRA datasets, enable the optional prepared GPU cache. Merge these
settings into the existing training configuration:

```yaml
network:
  type: lora
  transformer_only: true
train:
  batch_size: 1
  gradient_accumulation: 1
  train_text_encoder: false
  cache_text_embeddings: true
model:
  arch: qwen_image_2
  low_vram: false
  layer_offloading: false
  compile: false
  model_kwargs:
    cache_prepared_training_inputs: true
datasets:
  - folder_path: /path/to/images
    cache_latents: true
    caption_dropout_rate: 0
```

Preparation retains each target item's sampled latent, GPU prompt embeddings,
frozen text projections, rotary positions, and attention layout/masks. Metadata
copies avoid copying the original CPU latent on every iteration. The cache is
owned by the training run and cleared before final saving/sampling. Its VRAM
cost scales with dataset size and caption/reference lengths.

Editing references use the ordinary control-image preprocessing. Their frozen
VAE posteriors are prepared once and sampled afresh on each prediction, keeping
the previous sample/normalization order. Repeated target files remain separate
items because their cached target samples may differ. Trainable transformer
block activations and layer KV tensors are never cached across optimizer steps.

Prepared caching requires fixed captions and references, standard block-only
LoRA, and no CFG training, preservation/teacher modes, augmentation, masks, or
auxiliary adapters. Unsupported combinations raise an explanatory error.
Unlike the Ideogram fast path, Qwen rejects nonzero caption dropout instead of
silently ignoring it. Leaving the flag off preserves the ordinary training
path, including caption dropout, larger batches, and other training modes.

`testing/test_qwen_image_2_prepared.py` contains three focused CPU checks for
output/LoRA-gradient parity (including checkpointing and optimizer updates),
block-causal mask equivalence, and stale-cache rejection. They require an
installed AI Toolkit/PyTorch environment; no speedup or GPU parity has been
measured locally.
