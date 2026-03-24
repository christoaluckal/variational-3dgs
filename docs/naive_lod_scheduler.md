# Naive LoD Scheduler

## Goal

This document describes the deterministic LoD baseline in:

- [`train_vanilla.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train_vanilla.py)

It is a simple coarse-to-fine schedule with no probability-driven controller.

## Core idea

The active training scale is chosen only by iteration count.

Let:

- `lod_scales = reversed(resolution_scales)`
- `N = --naive_lod_stage_iterations`

Then the stage for zero-based training step `t` is:

- `stage_idx = min(floor(t / N), len(lod_scales) - 1)`

That means promotion happens at fixed boundaries `N * i`.

## Example

With:

- `--resolution_scales 2 4 8`
- `--naive_lod_stage_iterations 5000`

the LoD order is:

1. scale `8` for steps `0..4999`
2. scale `4` for steps `5000..9999`
3. scale `2` for steps `10000..14999`

In one-based iteration numbers, that is:

1. iterations `1..5000` at scale `8`
2. iterations `5001..10000` at scale `4`
3. iterations `10001+` at scale `2`

After the final promotion, training stays on the finest configured scale.

## What this variant does not do

Unlike [`train.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train.py), the naive scheduler:

- does not compute a probability probe for LoD switching
- does not use an EMA
- does not use thresholds
- does not depend on fixed probe views
- does not react to uncertainty spikes or drops

The optimization loss is still the usual training loss:

- photometric reconstruction
- KL-based variational regularization

The KL block can still be scaled with:

- `--probability_regularizer_weight`

## Resolution semantics

The first element of `--resolution_scales` is still the finest training scale.

If `--match_resolution` is enabled:

- coarse levels are loaded from lower-resolution source images
- then upsampled back to the finest training tensor size
- the LoD difference becomes blur/detail rather than tensor shape

## Evaluation and logging

`train_vanilla.py` keeps the current output stack:

- finest-scale evaluation via the shared reporting path
- `train_metrics.csv`
- `eval_metrics.csv`
- W&B logging for training loss, KL loss, LoD scale, and stage index

Since naive LoD does not use the uncertainty controller:

- `probability_probe_loss`
- `probability_probe_loss_std`
- `probability_probe_loss_ema`

are left empty in the training CSV for this variant.

## Why use it

This variant is useful as a control experiment.

It answers:

- how much of the observed behavior comes from coarse-to-fine staging alone
- how much comes from the probability-driven controller in [`train.py`](/mnt/share/nas/christo/splatting/variational-3dgs/train.py)

It is therefore the simplest LoD baseline in this folder.
