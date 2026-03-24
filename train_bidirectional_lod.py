#
# Draft training entrypoint for a bidirectional probability-driven LoD policy.
#
# This file does not modify train.py. It is a stand-alone sketch of how the
# training loop could behave if it started at the finest scale, fell back to a
# coarser curriculum when finest-scale probability loss worsened, and then
# reintroduced detail while still validating progress from the finest scale.
#

import sys
import os
from argparse import ArgumentParser
from contextlib import nullcontext
from random import randint

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import forward_k_times, network_gui, render
from scene import GaussianModel, Scene
from train import (
    _append_csv_row,
    _compute_probability_probe_loss,
    _initialize_csv_logger,
    _select_probability_probe_indices,
    WANDB_FOUND,
    _select_fixed_wandb_eval_view,
    prepare_output_and_logger,
    training_report,
)
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    import wandb
except ImportError:  # pragma: no cover - train.py already tolerates missing wandb
    wandb = None


def _build_viewpoint_stacks(scene, resolution_scales):
    viewpoint_dict = {
        scale: scene.getTrainCameras(scale=scale)
        for scale in resolution_scales
    }
    viewpoint_indices = list(range(len(viewpoint_dict[resolution_scales[0]])))
    return viewpoint_dict, viewpoint_indices


def _refill_viewpoint_indices(num_viewpoints):
    return torch.randperm(num_viewpoints).tolist()


def _prepare_resolution_scales(resolution_scales):
    unique_scales = list(dict.fromkeys(resolution_scales))
    if not unique_scales:
        raise ValueError("resolution_scales must contain at least one scale.")
    return unique_scales


def _resolve_recovery_thresholds(recovery_thresholds, resolution_scales):
    expected_thresholds = max(len(resolution_scales) - 1, 0)
    if recovery_thresholds:
        if len(recovery_thresholds) != expected_thresholds:
            raise ValueError(
                "--probability_recovery_thresholds must provide exactly "
                f"{expected_thresholds} threshold ratios for scales {resolution_scales}."
            )
        return recovery_thresholds

    if expected_thresholds == 0:
        return []
    if expected_thresholds == 1:
        return [0.95]
    if expected_thresholds == 2:
        return [0.9, 0.75]

    raise ValueError(
        "No default probability_recovery_thresholds are defined for "
        f"{expected_thresholds} LoD transitions. Please provide them explicitly."
    )


def _init_bidirectional_lod_state(first_iter, resolution_scales):
    return {
        "resolution_scales": resolution_scales,
        "current_scale_idx": 0,
        "mode": "finest",
        "last_scale_change_iteration": first_iter,
        "ema_probability_loss": None,
        "previous_ema_probability_loss": None,
        "best_ema_probability_loss": None,
        "recovery_anchor_probability_loss": None,
        "recovery_trigger_probability_loss": None,
        "num_increase_events": 0,
    }


def _format_lod_status(lod_state):
    return (
        f"{lod_state['mode']}@"
        f"{lod_state['resolution_scales'][lod_state['current_scale_idx']]}"
    )


def _enter_recovery_mode(iteration, lod_state):
    if len(lod_state["resolution_scales"]) <= 1:
        return

    trigger_loss = lod_state["ema_probability_loss"]
    lod_state["mode"] = "recovering"
    lod_state["current_scale_idx"] = len(lod_state["resolution_scales"]) - 1
    lod_state["last_scale_change_iteration"] = iteration
    lod_state["recovery_anchor_probability_loss"] = trigger_loss
    lod_state["recovery_trigger_probability_loss"] = trigger_loss
    lod_state["num_increase_events"] = 0

    next_scale = lod_state["resolution_scales"][lod_state["current_scale_idx"]]
    print(
        f"\n[ITER {iteration}] Finest-scale probability loss degraded; "
        f"switching to recovery mode at scale {next_scale} "
        f"(EMA {trigger_loss:.6f})."
    )


def _promote_recovery_scale(iteration, lod_state):
    if lod_state["current_scale_idx"] == 0:
        return

    lod_state["current_scale_idx"] -= 1
    lod_state["last_scale_change_iteration"] = iteration
    next_scale = lod_state["resolution_scales"][lod_state["current_scale_idx"]]
    print(
        f"\n[ITER {iteration}] Finest-scale probability loss improved; "
        f"promoting recovery to scale {next_scale}."
    )


def _maybe_update_bidirectional_lod_scale(
    iteration,
    lod_state,
    probability_loss,
    probability_loss_ema_alpha,
    probability_recovery_thresholds,
    probability_lod_min_iterations,
    probability_lod_increase_ratio,
    probability_lod_increase_patience,
):
    probability_value = float(probability_loss.item())

    if lod_state["ema_probability_loss"] is None:
        lod_state["ema_probability_loss"] = probability_value
        lod_state["previous_ema_probability_loss"] = probability_value
        lod_state["best_ema_probability_loss"] = probability_value
        return

    previous_ema = lod_state["ema_probability_loss"]
    ema_probability_loss = (
        probability_loss_ema_alpha * probability_value
        + (1.0 - probability_loss_ema_alpha) * previous_ema
    )
    lod_state["previous_ema_probability_loss"] = previous_ema
    lod_state["ema_probability_loss"] = ema_probability_loss

    best_ema = lod_state["best_ema_probability_loss"]
    if best_ema is None or ema_probability_loss < best_ema:
        lod_state["best_ema_probability_loss"] = ema_probability_loss
        best_ema = ema_probability_loss

    if iteration - lod_state["last_scale_change_iteration"] < probability_lod_min_iterations:
        return

    if lod_state["mode"] == "finest":
        degradation_threshold = best_ema * probability_lod_increase_ratio
        if ema_probability_loss >= degradation_threshold:
            lod_state["num_increase_events"] += 1
        else:
            lod_state["num_increase_events"] = 0

        if lod_state["num_increase_events"] >= probability_lod_increase_patience:
            _enter_recovery_mode(iteration, lod_state)
        return

    if lod_state["mode"] != "recovering":
        return

    anchor = lod_state["recovery_anchor_probability_loss"]
    if anchor is None:
        lod_state["recovery_anchor_probability_loss"] = ema_probability_loss
        anchor = ema_probability_loss

    if lod_state["current_scale_idx"] == 0:
        recovery_target = min(anchor, best_ema)
        if ema_probability_loss <= recovery_target:
            lod_state["mode"] = "finest"
            lod_state["last_scale_change_iteration"] = iteration
            lod_state["recovery_anchor_probability_loss"] = None
            lod_state["recovery_trigger_probability_loss"] = None
            lod_state["num_increase_events"] = 0
            print(
                f"\n[ITER {iteration}] Recovery validated on finest scale "
                f"(EMA {ema_probability_loss:.6f} <= target {recovery_target:.6f})."
            )
        return

    threshold_idx = lod_state["current_scale_idx"] - 1
    threshold_ratio = probability_recovery_thresholds[threshold_idx]
    threshold = anchor * threshold_ratio
    if ema_probability_loss <= threshold:
        _promote_recovery_scale(iteration, lod_state)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    resolution_scales,
    probability_lod_interval,
    probability_lod_min_iterations,
    probability_loss_ema_alpha,
    probability_lod_probe_num_views,
    probability_lod_increase_ratio,
    probability_lod_increase_patience,
    probability_recovery_thresholds,
):
    opt.position_lr_max_steps = opt.iterations

    resolution_scales = _prepare_resolution_scales(resolution_scales)
    finest_scale = resolution_scales[0]
    probability_recovery_thresholds = _resolve_recovery_thresholds(
        probability_recovery_thresholds, resolution_scales
    )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    train_csv_fields = [
        "iteration",
        "photometric_loss",
        "total_loss",
        "kl_scale_loss",
        "kl_xyz_loss",
        "kl_opacity_loss",
        "probability_regularizer",
        "probability_probe_loss",
        "probability_probe_loss_std",
        "probability_probe_loss_ema",
        "probability_probe_loss_best_ema",
        "lod_scale",
        "lod_mode",
        "num_gaussians",
        "iter_time_ms",
    ]
    eval_csv_fields = [
        "iteration",
        "split",
        "eval_scale",
        "num_cameras",
        "l1",
        "psnr",
    ]
    train_metrics_csv = os.path.join(dataset.model_path, "train_metrics.csv")
    eval_metrics_csv = os.path.join(dataset.model_path, "eval_metrics.csv")
    _initialize_csv_logger(train_metrics_csv, train_csv_fields)
    _initialize_csv_logger(eval_metrics_csv, eval_csv_fields)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_dict, viewpoint_indices = _build_viewpoint_stacks(scene, resolution_scales)
    probability_probe_indices = _select_probability_probe_indices(
        len(viewpoint_dict[finest_scale]), probability_lod_probe_num_views
    )
    print(
        "Using fixed probability probe viewpoints "
        f"{probability_probe_indices} at finest scale {finest_scale}"
    )
    lod_state = _init_bidirectional_lod_state(first_iter, resolution_scales)
    fixed_wandb_eval_view = _select_fixed_wandb_eval_view(scene, finest_scale)
    ema_loss_for_log = 0.0

    from tqdm import tqdm

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (
                            torch.clamp(net_image, min=0, max=1.0) * 255
                        )
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        if not viewpoint_indices:
            viewpoint_indices = _refill_viewpoint_indices(len(viewpoint_dict[finest_scale]))
        viewpoint_idx = viewpoint_indices.pop(randint(0, len(viewpoint_indices) - 1))

        current_scale = lod_state["resolution_scales"][lod_state["current_scale_idx"]]
        viewpoint_cam = viewpoint_dict[current_scale][viewpoint_idx]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        model_id = torch.randint(0, gaussians.n_models, (1,)).item()
        gaussians.model_id = model_id

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss_kl_scal = gaussians.compute_kl_uniform_scal()
        loss_kl_xyz = gaussians.compute_kl_xyz()
        loss_kl_opacity = gaussians.compute_kl_opacity()
        probability_regularizer = (
            opt.probability_regularizer_weight
            * (loss_kl_scal + loss_kl_xyz + loss_kl_opacity)
        )
        loss += probability_regularizer

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            probability_loss = None
            probability_loss_std = None
            if probability_lod_interval > 0 and iteration % probability_lod_interval == 0:
                probability_loss, probability_loss_samples = _compute_probability_probe_loss(
                    viewpoint_dict,
                    finest_scale,
                    probability_probe_indices,
                    gaussians,
                    pipe,
                    background,
                )
                probability_loss_std = float(
                    probability_loss_samples.std(unbiased=False).item()
                )
                _maybe_update_bidirectional_lod_scale(
                    iteration,
                    lod_state,
                    probability_loss,
                    probability_loss_ema_alpha,
                    probability_recovery_thresholds,
                    probability_lod_min_iterations,
                    probability_lod_increase_ratio,
                    probability_lod_increase_patience,
                )

            if WANDB_FOUND and wandb is not None and wandb.run is not None:
                train_log = {
                    "train/photometric_loss": Ll1.item(),
                    "train/total_loss": loss.item(),
                    "train/kl_scale_loss": loss_kl_scal.item(),
                    "train/num_gaussians": gaussians.get_xyz.shape[0],
                    "train/lod_scale": current_scale,
                    "train/lod_mode": lod_state["mode"],
                }
                if probability_loss is not None:
                    train_log["train/probability_probe_loss"] = probability_loss.item()
                if probability_loss_std is not None:
                    train_log["train/probability_probe_loss_std"] = probability_loss_std
                if lod_state["ema_probability_loss"] is not None:
                    train_log["train/probability_probe_loss_ema"] = lod_state["ema_probability_loss"]
                if lod_state["best_ema_probability_loss"] is not None:
                    train_log["train/probability_probe_loss_best_ema"] = lod_state["best_ema_probability_loss"]
                wandb.log(train_log, step=iteration)

            _append_csv_row(
                train_metrics_csv,
                train_csv_fields,
                {
                    "iteration": iteration,
                    "photometric_loss": Ll1.item(),
                    "total_loss": loss.item(),
                    "kl_scale_loss": loss_kl_scal.item(),
                    "kl_xyz_loss": loss_kl_xyz.item(),
                    "kl_opacity_loss": loss_kl_opacity.item(),
                    "probability_regularizer": probability_regularizer.item(),
                    "probability_probe_loss": None if probability_loss is None else probability_loss.item(),
                    "probability_probe_loss_std": probability_loss_std,
                    "probability_probe_loss_ema": lod_state["ema_probability_loss"],
                    "probability_probe_loss_best_ema": lod_state["best_ema_probability_loss"],
                    "lod_scale": current_scale,
                    "lod_mode": lod_state["mode"],
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "iter_time_ms": iter_start.elapsed_time(iter_end),
                },
            )

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_metrics = {
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "LoD": _format_lod_status(lod_state),
                }
                if lod_state["ema_probability_loss"] is not None:
                    progress_metrics["ProbEMA"] = f"{lod_state['ema_probability_loss']:.7f}"
                progress_bar.set_postfix(progress_metrics)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            spawn_interval = dataset.spawn_interval

            training_report(
                dataset,
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                finest_scale,
                current_scale,
                loss_kl_scal,
                probability_regularizer,
                probability_loss,
                lod_state["ema_probability_loss"],
                fixed_wandb_eval_view,
                eval_metrics_csv,
                eval_csv_fields,
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % spawn_interval == 0:
                    gaussians.spawn(scene.cameras_extent)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


if __name__ == "__main__":
    import numpy as np

    parser = ArgumentParser(
        description="Bidirectional LoD training draft with finest-scale validation"
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=np.arange(1000, 20001, 1000).tolist())
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--resolution_scales", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--probability_lod_interval", type=int, default=50)
    parser.add_argument("--probability_lod_min_iterations", type=int, default=1000)
    parser.add_argument("--probability_loss_ema_alpha", type=float, default=0.1)
    parser.add_argument("--probability_lod_probe_num_views", type=int, default=4)
    parser.add_argument(
        "--probability_lod_increase_ratio",
        type=float,
        default=1.05,
        help="Enter recovery mode when finest-scale EMA rises above best EMA by this ratio.",
    )
    parser.add_argument(
        "--probability_lod_increase_patience",
        type=int,
        default=2,
        help="Number of consecutive degraded EMA checks required before recovery mode starts.",
    )
    parser.add_argument(
        "--probability_recovery_thresholds",
        nargs="*",
        type=float,
        default=[],
        help="Ratios against the recovery-entry EMA used to reintroduce finer scales.",
    )
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="variational-3dgs")
    parser.add_argument("--wandb_name", type=str, default="bidirectional-probability-lod")
    args = parser.parse_args(sys.argv[1:])

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not WANDB_FOUND and not args.disable_wandb:
        print("wandb not available: proceeding without Weights & Biases logging")

    wandb_context = nullcontext()
    if WANDB_FOUND and wandb is not None and not args.disable_wandb:
        wandb_context = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    with wandb_context:
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.resolution_scales,
            args.probability_lod_interval,
            args.probability_lod_min_iterations,
            args.probability_loss_ema_alpha,
            args.probability_lod_probe_num_views,
            args.probability_lod_increase_ratio,
            args.probability_lod_increase_patience,
            args.probability_recovery_thresholds,
        )

    print("\nTraining complete.")
