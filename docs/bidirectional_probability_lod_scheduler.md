# Bidirectional Probability-Loss LoD Scheduler

## Goal

This document describes the draft alternative training routine in:

- [`train_bidirectional_lod.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_bidirectional_lod.py)

It is intentionally separate from the current scheduler in `train.py`.

The draft policy changes the LoD behavior to:

1. start training on the finest configured scale
2. monitor probability loss on that same finest scale
3. if finest-scale probability loss gets worse, fall back to coarser supervision
4. progressively reintroduce finer detail
5. validate recovery from the finest scale

## Resolution semantics

With:

- `--resolution_scales 2 4 8`

the draft assumes:

- `2` is finest
- `8` is coarsest

Unlike the current `train.py` scheduler, the draft does not reverse the list for the initial training stage.

## Core interaction

- One training camera index is sampled.
- The same index is reused across all configured scales.
- The optimization loss is computed at the currently active scale.
- The uncertainty probe is always computed on the finest configured scale.
- The same `--match_resolution` option exists here too, so lower LoD levels can optionally be blurrier views at the finest training tensor size.
- The probe uses:
  - `forward_k_times(...)`
  - `nll_kernel_density(...)`
- `forward_k_times(...)` now evaluates stable `model_id`-driven ensemble members, so repeated probe checks are more comparable over time.
- The finest-scale probability loss is smoothed with an EMA.
- The backpropagated KL regularizer block is still separate from the probe and can be scaled with `--probability_regularizer_weight`.

## State machine

The draft scheduler has two modes.

### 1. Finest mode

- Active optimization scale is the finest configured scale.
- The scheduler tracks:
  - current EMA of finest-scale probability loss
  - best EMA seen so far
- If the EMA rises above `best_ema * probability_lod_increase_ratio` for enough consecutive checks, the scheduler treats that as degradation.

Default degradation controls in the draft:

- `--probability_lod_increase_ratio 1.05`
- `--probability_lod_increase_patience 2`

That means the draft waits for the finest-scale EMA to exceed the best EMA by 5% for two consecutive scheduler checks before triggering recovery.

### 2. Recovering mode

- The routine jumps to the coarsest configured scale.
- It stores the EMA at recovery entry as the recovery anchor.
- It keeps measuring probability loss from the finest scale.
- As the finest-scale EMA improves relative to the recovery anchor, it promotes training toward finer scales.

With defaults for three scales:

- `--probability_recovery_thresholds 0.9 0.75`

the draft transitions look like:

- `8 -> 4` when `EMA <= recovery_anchor * 0.75`
- `4 -> 2` when `EMA <= recovery_anchor * 0.9`

Once back on the finest scale, recovery is considered validated only when the finest-scale EMA is at or below the recovery target.

## Why the probe stays on the finest scale

The draft keeps the uncertainty probe on the finest configured scale for the same reason as the existing scheduler:

- the model ultimately has to succeed on the sharpest supervision
- coarse-scale confidence alone is not enough evidence that detail has recovered

This also matches the requested behavior of validating improvement from the finest resolution.

## What changes relative to `train.py`

- `train.py` is one-way coarse-to-fine.
- `train_bidirectional_lod.py` is fine-first, then coarse-to-fine only when degradation is detected.
- `train.py` uses a warmup baseline built from early fixed-view probe measurements and checks whether EMA falls below baseline-scaled thresholds.
- `train_bidirectional_lod.py` uses:
  - best-so-far EMA to detect degradation
  - recovery-entry EMA as the anchor for reintroducing finer levels

## Logging and Outputs

The draft routine follows the same general logging structure as `train.py`:

- WandB logs training loss and probe metrics
- per-run CSV files are written to the output directory
  - `train_metrics.csv`
  - `eval_metrics.csv`

The bidirectional train log additionally tracks controller state:

- `train/lod_mode`
- `train/probability_probe_loss_ema`
- `train/probability_probe_loss_best_ema`

This is important when analyzing whether a fallback was driven by real degradation or controller noise.

## Caveats

- This is a draft scheduler, not the established default behavior.
- Densification and pruning still operate on the active training render, so changing scales mid-training can alter those signals.
- The probability probe remains relatively expensive because it uses `forward_k_times(...)`.
- The bidirectional runner does not currently sweep `--probability_regularizer_weight`; that argument still exists in the training script and defaults to `1.0` unless passed explicitly.
- Threshold defaults are heuristic and would likely need tuning per dataset.

## Recommended files to compare

- [`train.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train.py)
- [`train_bidirectional_lod.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_bidirectional_lod.py)
- [`docs/probability_lod_scheduler.md`](/mnt/share/nas/christo/splatting/variational-3dgs/docs/probability_lod_scheduler.md)
