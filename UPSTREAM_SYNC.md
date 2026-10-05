# Upstream integration — 2026-10-05

Integration branch: `ostris-main`.

- Fork starting point: `6a760187aef5c4a578cc3e28f275f5ed87a8ba19`.
- Upstream snapshot: `ecee894ed2b1f3716d9d7326693061ec1a3105bb`.
- Common ancestor: `64663c8575b7a9ac3b946611724f76d015c941bc`.
- Divergence: 16 fork commits and 313 upstream commits.
- Fork delta: 14 files, +927/-151. Upstream delta: 424 files,
  +62,500/-10,380. Thirteen of our files overlap upstream changes.
- Five textual conflicts: `BaseSDTrainProcess.py`, both data-loader files,
  `lora_special.py`, and `SimpleJob.tsx`.

## Resolution decisions

| Area | Resolution |
| --- | --- |
| Ideogram static conditioning and sequence packing | Preserve the fork implementation. Upstream has no equivalent prepared GPU cache. The pipeline is unchanged from the fork; all 26 existing transformer methods are preserved. |
| Model loading | Adopt upstream's shared VAE, Qwen loader, model mixin, and post-load placement/quantization APIs. Preserve the baked BF16 validation/direct-device loading option and FP8 dequantization helpers. |
| LoRA scaling | Adopt upstream's nonpersistent runtime scale buffer and removal of the fixed scalar tensor. Retain the fork's fixed-unit-scale and all-ones-multiplier shortcuts. |
| Dynamic scale updates | Invalidate the unit shortcut when `_set_runtime_scale` changes an existing adapter, retaining the same runtime buffer. Otherwise a unit-to-nonunit change can silently ignore the new scale. Extraction also invalidates the shortcut. |
| Data loading | Keep upstream's module-level collator, persistent workers, dataset batch overrides, and DTO batching for ordinary loads. Prepared GPU inputs use a separate main-process collator with zero workers and retain their original latent/embedding/context references. |
| Primary prediction and embeddings | Keep primary-prediction forwarding and prepared-embedding reuse without per-step cloning. Auxiliary predictions retain their separate conditioning path. |
| Training progress | Keep deferred metric materialization and `progress_every`. Adopt upstream's compatible completed-step count, smoothing, and pause changes. |
| Checkpoints | Keep `save_start_step` and its offset cadence. Upstream's `sample_start_step` remains an independent sampling gate. Final saving remains unconditional. |
| Timestep bias | Keep `content_u2` (U²) alongside upstream content/style behavior and keep its option in the reorganized UI. |

The prepared loader no longer accesses `_cached_audio_latent`, which upstream
removed in favor of DTO extras. Ordinary batching uses upstream `DTO.stack`.
Prepared-cache validation also rejects newly available combinations it cannot
serve correctly: dataset batches above one, raw-tensor caching, and DOP/D-OPSD
conditioning. Ordinary training can still use upstream's implementations.

## Caption dropout

The prepared GPU cache deliberately retains the fork's fixed-caption behavior.
Upstream's cached-caption dropout fix applies to the ordinary loader; selecting
an alternate embedding alone would not update the prepared transformer context.
A nonzero `caption_dropout_rate` now produces a notice during preparation rather
than implying it is active on this path.

The outer `train_dev.py` currently combines prepared caching with a rate of
0.01, so its effective dropout remains zero, as before this merge. Supporting
dropout on the prepared path would require paired embedding/context variants;
this integration does not introduce that training-behavior change.

## Modal dependencies

The outer project's `modal-image-requirements.txt` is maintained separately from
this submodule. Its changed requirements were aligned with upstream: Diffusers
`c943837899b16cbae2f619b8dd4f7bb6f07dd81a`, Hub 1.23.0,
Setuptools >=77.0.3, TorchCodec 0.15.0, FastAPI, and Uvicorn. The unused
k-diffusion requirement was removed. Existing outer-only SciPy/TorchAudio
requirements and the previous invisible-watermark omission were retained.

The Diffusers revision matters even when training only Ideogram: model discovery
eagerly imports the diffusion-model package, including new model APIs. Commit
the outer dependency changes together with its updated submodule pointer when
adopting this branch. No model downloads or remote training jobs were started.

## Verification and remaining runtime validation

Completed locally:

- Parsed all 523 unique tracked Python files using Python 3.10 syntax rules.
- Compiled the changed TSX entry with Bun (dependencies externalized).
- Checked checkpoint cadence, progress/log materialization, scale-update
  invalidation, prepared-reference reuse/cleanup, metadata copies, and
  ordinary/prepared collation using extracted production definitions with
  lightweight dependency substitutes. These exercise control flow, not tensor
  or gradient numerics.
- Compared the preserved pipeline, transformer methods, and weight-loading
  helpers against the original fork.
- Checked the fork-specific diff for whitespace errors and the merge for
  conflict markers.
- Extended upstream's LoRA scale tests for the preserved fast paths and runtime
  scale transitions.

The tensor tests could not run: local Python lacks PyTorch, and the CPU PyTorch
download host was blocked even with an escalation request. GPU numerical parity,
throughput, peak VRAM, and an actual training/checkpoint cycle remain unverified.

Before promoting this integration to production, use the rebuilt training
environment to run `pip check`, import the trainer, verify that `ideogram4`
resolves via `get_model_class`, and run:

```bash
python -m unittest testing.test_lora_compile_scalars -v
```

Then run a short H200 canary with fixed inputs/noise/timesteps, compare prepared
and ordinary prediction/loss/LoRA gradients, and exercise checkpoint save/resume.
Compare step time and peak VRAM with the fork starting point; source preservation
alone does not establish runtime performance parity.
