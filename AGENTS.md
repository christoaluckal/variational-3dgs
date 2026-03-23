# variational-3DGS Notes

## Purpose

Use this file as the quick handoff for training behavior in this folder, especially the probability-driven level-of-detail scheduling added in `train.py`.

## Start here

If the task is about LoD scheduling, probability loss, or multi-resolution camera usage, begin with:

- [`train.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train.py)
- [`train_bidirectional_lod.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_bidirectional_lod.py)
- [`scene/__init__.py`](/mnt/share/nas/christo/splatting/variational-3dgs/scene/__init__.py)
- [`utils/camera_utils.py`](/mnt/share/nas/christo/splatting/variational-3dgs/utils/camera_utils.py)
- [`gaussian_renderer/__init__.py`](/mnt/share/nas/christo/splatting/variational-3dgs/gaussian_renderer/__init__.py)
- [`docs/probability_lod_scheduler.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/probability_lod_scheduler.md)
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
- `--probability_lod_thresholds`
- `--probability_lod_interval`
- `--probability_lod_min_iterations`
- `--probability_loss_ema_alpha`
- `--probability_lod_probe_num_views`
- `--probability_lod_baseline_warmup_probes`

Example with defaults:

- `--resolution_scales 2 4 8`
- `--probability_lod_thresholds 0.5 0.3`
- `--probability_lod_probe_num_views 4`
- `--probability_lod_baseline_warmup_probes 5`

This means:

- start at scale `8`
- estimate the baseline from the median of the first `5` averaged probe measurements
- each probe measurement averages `4` fixed finest-scale views
- move to scale `4` when probability-loss EMA is below `0.5 * baseline`
- move to scale `2` when probability-loss EMA is below `0.3 * baseline`

## Important implementation detail

- The probability-loss probe is not the same tensor as the optimization loss used for backprop each iteration.
- The optimization loss still uses:
  - image reconstruction term
  - KL-based variational regularizers
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
  - `train/kl_scale_loss`
  - `train/probability_probe_loss`
  - `train/probability_probe_loss_std`
  - `train/lod_scale`
- TensorBoard uses the same naming for the probe metrics and also logs `probability_probe_loss_ema`.
- Each run output directory now also contains:
  - `train_metrics.csv`
  - `eval_metrics.csv`
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

- `run_exp.py` now forwards the scheduler robustness settings explicitly:
  - `--probability_lod_interval 50`
  - `--probability_lod_min_iterations 1000`
  - `--probability_loss_ema_alpha 0.1`
  - `--probability_lod_probe_num_views 4`
  - `--probability_lod_baseline_warmup_probes 5`
