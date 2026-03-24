# variational-3DGS Notes

## Purpose

Use this file as the quick handoff for training behavior in this folder, especially the probability-driven level-of-detail scheduling added in `train.py`.

## Start here

If the task is about LoD scheduling, probability loss, or multi-resolution camera usage, begin with:

- [`train.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train.py)
- [`train_vanilla.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_vanilla.py)
- [`train_bidirectional_lod.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_bidirectional_lod.py)
- [`scene/__init__.py`](/mnt/share/nas/christo/splatting/variational-3dgs/scene/__init__.py)
- [`utils/camera_utils.py`](/mnt/share/nas/christo/splatting/variational-3dgs/utils/camera_utils.py)
- [`gaussian_renderer/__init__.py`](/mnt/share/nas/christo/splatting/variational-3dgs/gaussian_renderer/__init__.py)
- [`docs/probability_lod_scheduler.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/probability_lod_scheduler.md)
- [`docs/naive_lod_scheduler.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/naive_lod_scheduler.md)
- [`docs/bidirectional_probability_lod_scheduler.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/bidirectional_probability_lod_scheduler.md)
- [`docs/caveats.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/caveats.md)

## Current training behavior

- The training loop now builds a multi-resolution camera stack from `--resolution_scales`.
- The intended default order is `2 4 8`, where:
  - `2` is the finest training scale
  - `4` is intermediate
  - `8` is the coarsest
- Training itself runs coarse-to-fine by reversing that list internally:
  - start on `8`
  - promote to `4`
  - promote to `2`

## How LoD promotion works

- One training camera index is still sampled and reused across all scales for the optimization step.
- The active optimization image comes from the current LoD scale.
- The uncertainty signal is always measured on the finest configured scale, which is the first element of `--resolution_scales`.
- The current implementation uses a fixed probe set of training views instead of a single random probe view.
- For each scheduler check, the code renders all probe views at the finest scale and averages their probability losses.
- The uncertainty probe now uses stable `model_id`-driven ensemble members rather than fresh uncontrolled resampling on every probe render.
- The rendered probability loss uses:
  - `forward_k_times(...)`
  - `nll_kernel_density(...)`
- The averaged probability loss is smoothed with an EMA.
- The scheduler does not trust a one-shot baseline anymore:
  - it collects several early probe values
  - takes the median as the baseline
  - only then starts promotion checks
- LoD promotion happens when the EMA drops below a ratio of that warmup baseline.

## Relevant CLI controls

- `--resolution_scales`
- `--match_resolution`
- `--probability_regularizer_weight`
- `--probability_lod_thresholds`
- `--probability_lod_interval`
- `--probability_lod_min_iterations`
- `--probability_loss_ema_alpha`
- `--probability_lod_probe_num_views`
- `--probability_lod_baseline_warmup_probes`

Example with defaults:

- `--resolution_scales 2 4 8`
- `--probability_regularizer_weight 1.0`
- `--probability_lod_thresholds 0.5 0.3`
- `--probability_lod_probe_num_views 4`
- `--probability_lod_baseline_warmup_probes 5`

This means:

- start at scale `8`
- estimate the baseline from the median of the first `5` averaged probe measurements
- each probe measurement averages `4` fixed finest-scale views
- move to scale `4` when probability-loss EMA is below `0.5 * baseline`
- move to scale `2` when probability-loss EMA is below `0.3 * baseline`

If `--match_resolution` is enabled:

- lower LoD scales are still loaded from downsampled images
- but they are resized back to the finest training resolution before creating the `Camera`
- the effective difference between LoD levels becomes blur/detail rather than image size

## Important implementation detail

- The probability-loss probe is not the same tensor as the optimization loss used for backprop each iteration.
- The optimization loss still uses:
  - image reconstruction term
  - KL-based variational regularizers
- The KL regularizer block can be globally scaled with `--probability_regularizer_weight`.
- The LoD scheduler uses a separate uncertainty-aware probe based on finest-scale ensemble rendering.

## How to Read Probe Metrics

- `probability_probe_loss` is a calibration signal, not a reconstruction signal.
- It depends on both mean error and predicted uncertainty.
- It can increase even while photometric loss and PSNR improve if the model becomes overconfident.
- `probability_probe_loss_std` measures ensemble spread on the fixed probe set.
- A rise in `probability_probe_loss_std` can sometimes reduce `probability_probe_loss` if the model was previously underestimating uncertainty.
- Higher probe spread is not automatically better; the best probe score happens when uncertainty matches the remaining residual error.

## Current Logging Outputs

- WandB train logs include:
  - `train/photometric_loss`
  - `train/total_loss`
  - `train/kl_loss`
  - `train/probability_probe_loss`
  - `train/probability_probe_loss_std`
  - `train/lod_scale`
- `train_bidirectional_lod.py` also logs:
  - `train/kl_scale_loss`
  - `train/probability_probe_loss_ema`
  - `train/probability_probe_loss_best_ema`
  - `train/lod_mode`
- TensorBoard uses the same naming for the probe metrics and also logs `probability_probe_loss_ema`.
- Each run output directory now also contains:
  - `train_metrics.csv`
  - `eval_metrics.csv`
- `train_metrics.csv` is the most complete per-iteration record and includes:
  - `kl_scale_loss`
  - `kl_xyz_loss`
  - `kl_opacity_loss`
  - `probability_regularizer`
  - `probability_probe_loss`
  - `probability_probe_loss_std`
  - `probability_probe_loss_ema`
- Final render-set summaries are still appended to the dataset-level evaluation CSV.

## Draft bidirectional LoD behavior

- `train_bidirectional_lod.py` is a draft alternative scheduler and does not change `train.py`.
- The draft policy starts on the finest configured scale first.
- It continues to measure uncertainty on the finest configured scale.
- If the finest-scale probability-loss EMA worsens beyond a configured ratio of the best EMA so far, the routine enters a recovery mode:
  - jump to the coarsest configured scale
  - then promote back toward finer scales as finest-scale probability loss improves
- Recovery is considered complete only after the finest-scale EMA improves enough again.
- Validation and reporting still use the finest configured scale.

## Naive LoD behavior

- `train_vanilla.py` is a deterministic baseline with no probability-driven switching.
- It still uses coarse-to-fine ordering by reversing `--resolution_scales`.
- Promotions happen at fixed stage boundaries:
  - `stage_idx = floor((iteration - 1) / --naive_lod_stage_iterations)`
- Example:
  - `--resolution_scales 2 4 8 --naive_lod_stage_iterations 5000`
  - scale `8` for iterations `1..5000`
  - scale `4` for iterations `5001..10000`
  - scale `2` for iterations `10001+`
- It uses the same training loss, CSV outputs, evaluation path, and optional `--match_resolution` behavior as the other entrypoints.

## Files and functions to check before editing

- `_build_viewpoint_stacks(...)`
- `_refill_viewpoint_indices(...)`
- `_prepare_resolution_scales(...)`
- `_select_probability_probe_indices(...)`
- `_validate_probability_lod_thresholds(...)`
- `_maybe_update_lod_scale(...)`
- `_compute_probability_probe_loss(...)`
- `_active_member_ids(...)`
- `_stable_member_uniform(...)`
- `_stable_member_normal(...)`
- `_maybe_update_bidirectional_lod_scale(...)`
- `training(...)`
- `training_report(...)`
- `render_set(...)`

## Caveats

- The probability probe uses `forward_k_times(...)`, so it is more expensive than a normal single render.
- It currently runs every `--probability_lod_interval` iterations.
- It now probes multiple fixed views per scheduler check, which improves robustness but increases probe cost.
- The scheduler is more stable than the old one-shot / one-view version, but it is still driven by a limited probe set rather than full-scene validation.
- Evaluation and final rendering now use the finest configured scale instead of assuming scale `1.0`.

## Runner defaults

- `run_exp.py` now defines three regular-training variants:
  - `baseline`
  - `lod`
  - `matched-lod`
- `baseline` runs do not use LoD and only use `--probability_regularizer_weight 1.0`.
- `lod` and `matched-lod` sweep:
  - `--probability_regularizer_weight 1.0`
  - `--probability_regularizer_weight 0.5`
  - `--probability_regularizer_weight 0.1`
- `matched-lod` enables `--match_resolution`, so coarse levels are blurred-but-same-size rather than smaller tensors.
- `run_exp.py` forwards the scheduler robustness settings explicitly:
  - `--probability_lod_interval 50`
  - `--probability_lod_min_iterations 1000`
  - `--probability_loss_ema_alpha 0.2`
  - `--probability_lod_probe_num_views 4`
  - `--probability_lod_baseline_warmup_probes 5`
- `run_exp_bi.py` uses the draft bidirectional routine with:
  - `--probability_lod_interval 50`
  - `--probability_lod_min_iterations 1000`
  - `--probability_loss_ema_alpha 0.1`
  - `--probability_lod_probe_num_views 4`
  - `--probability_lod_increase_ratio 1.05`
  - `--probability_lod_increase_patience 2`
  - `MATCH_RESOLUTION = False` in the runner by default
