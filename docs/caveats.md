# LoD Scheduling with Uncertainty Signals in Gaussian Splatting

## Overview

This document outlines design considerations, caveats, and practical guidelines for using uncertainty-driven Level-of-Detail (LoD) scheduling in Gaussian splatting training.

Two approaches are relevant in this folder:

1. **Coarse-to-fine (unidirectional)**  
   Start at low resolution and promote when uncertainty decreases.

2. **Bidirectional (experimental draft)**  
   Allow fallback to lower resolution when uncertainty increases.

The bidirectional approach is documented as a draft in `train_bidirectional_lod.py` and has not yet been validated as the default training policy.

---

# Core Concept

Training is performed at the **active LoD scale**, but scheduling decisions are driven by a **finest-scale ensemble uncertainty probe**.

This introduces a separation:

- **Optimization signal** → active-scale reconstruction + regularization
- **Control signal** → finest-scale uncertainty probe (`forward_k_times(...)` + `nll_kernel_density(...)`)

This separation is powerful but introduces risks.

---

# Caveats

## 1. Probe ≠ Training Objective

The uncertainty signal is not directly optimized.

### Implications
- Improvements in training loss may not align with uncertainty.
- Uncertainty may improve due to smoothing, not better reconstruction.
- Scheduler becomes an external controller.

### Risk
Switching decisions may not correlate with final quality.

---

## 2. Baseline Sensitivity (Regular Mode)

The regular scheduler is anchored to an **early warmup baseline** rather than a one-shot first probe.

### Implications
- The warmup median is more robust than a one-shot baseline.
- But the chosen warmup window still anchors all future promotions.

### Risk
- Too-early or too-late promotions
- Strong seed sensitivity
- Thresholds that still need retuning when scenes differ substantially

---

## 3. Limited-Sample Noise in the Controller

Uncertainty is estimated via a finite ensemble render on a fixed probe set.

### Implications
- The controller is more stable than pure fresh MC resampling.
- But it still depends on a small number of probe views and a finite number of ensemble members.
- EMA smoothing only partially helps.

### Risk
- False triggers (especially dangerous for bidirectional fallback)
- Sensitivity to sampling and probe interval

---

## 4. Viewpoint Bias

Scheduler decisions no longer depend on a single sampled view, but they still depend on a **small fixed probe set**.

### Implications
- Hard views can still dominate if the probe set is not representative.
- Control can become probe-set-dependent rather than scene-dependent.

### Risk
- Inconsistent transition timing
- Overreaction to difficult viewpoints if they are overrepresented in the probe set

---

## 5. Entanglement with Structural Updates

LoD switching occurs alongside:
- Densification
- Pruning
- Opacity resets
- SH degree increases
- Changes in `--probability_regularizer_weight`

### Implications
Multiple factors affect performance simultaneously.

### Risk
Misattributing improvements or regressions to LoD changes.

---

## 6. Coarse-to-Fine Can Hide Fine Failures

Low-resolution training stabilizes optimization but may delay fine detail learning.

### Risk
- Poor recovery of thin structures
- Slow adaptation after promotion
- Over-reliance on low-frequency explanations

---

# Bidirectional-Specific Risks (Experimental)

## 7. Oscillation Risk

Allowing fallback introduces feedback dynamics.

### Risk
- Ping-pong between scales
- Instability from noisy signals

---

## 8. Fragile "Best EMA" Reference

Using the best historical EMA as a reference point:

### Risk
- Overreaction after a lucky low value
- Unrealistic degradation detection

---

## 9. Loss of Fine Specialization

Fallback to coarse scale may:

- Regularize the model
- But also erase fine-detail gradients

### Risk
- Reduced sharpness
- Loss of learned high-frequency structure

---

## 10. Circular Recovery Logic

Recovery thresholds tied to fallback entry point.

### Risk
- Artificial "recovery" without real improvement
- Sensitivity to entry timing

---

# Design Guidelines

## Scheduler Design

- Promotion should be easier than fallback is strict
- Use robust baselines (average or median over early probes)
- Separate logic for:
  - scale-up (promotion)
  - scale-down (fallback)
- Avoid switching near structural updates
- Add hysteresis (require sustained conditions)

---

## Uncertainty Measurement

- Use a fixed set of canonical probe views
- Average multiple views per probe
- Prefer stable ensemble members over fresh uncontrolled resampling for controller decisions
- Use EMA + additional smoothing if needed
- Increase robustness for fallback triggers

---

## Logging & Diagnostics

Log at every LoD transition:

- active scale
- uncertainty probe loss (raw + EMA)
- uncertainty probe spread
- probability regularizer strength and value
- validation PSNR / L1
- gaussian count
- recent structural events

Current code also writes:

- `train_metrics.csv`
- `eval_metrics.csv`

Track:
- per-view uncertainty
- transition timing vs quality changes

Useful current fields include:

- `total_loss`
- `kl_scale_loss`
- `kl_xyz_loss`
- `kl_opacity_loss`
- `probability_regularizer`
- `probability_probe_loss`
- `probability_probe_loss_std`
- `probability_probe_loss_ema`

---

## Experimentation Strategy

1. Establish baselines:
   - fixed finest-scale training
   - fixed coarse-scale training
   - coarse-to-fine scheduler

2. Validate coarse-to-fine thoroughly before bidirectional

3. Run multiple seeds

4. Evaluate:
   - final quality
   - convergence speed
   - stability (no oscillations)

---

# Decision Framework

## Use Coarse-to-Fine (Primary)

- Stable and well-understood
- Helps early geometry formation
- Lower risk of instability

## Use Bidirectional (Conditional)

Only if:
- You observe consistent fine-scale instability
- Uncertainty spikes correlate with real quality drops

Fallback should be:
- rare
- conservative
- strongly validated

---

# Success Criteria

A valid scheduler should:

- Improve final finest-scale quality
- Reduce training instability
- Avoid oscillations
- Produce consistent results across seeds
- Show alignment between uncertainty and validation metrics

---

# Key Principle

**Uncertainty is only useful if it predicts real downstream quality.**

If it does not:
- It is noise
- And the scheduler will eventually optimize for the wrong signal
