#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from random import randint
from argparse import ArgumentParser
from contextlib import nullcontext

from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams

import train as train_module


def _resolve_naive_lod_scale(iteration, lod_scales, naive_lod_stage_iterations):
    if naive_lod_stage_iterations <= 0:
        raise ValueError("--naive_lod_stage_iterations must be at least 1.")

    zero_based_step = max(iteration - 1, 0)
    stage_idx = min(zero_based_step // naive_lod_stage_iterations, len(lod_scales) - 1)
    return stage_idx, lod_scales[stage_idx]


def _maybe_update_naive_lod_scale(iteration, lod_state, naive_lod_stage_iterations):
    next_scale_idx, next_scale = _resolve_naive_lod_scale(
        iteration,
        lod_state["lod_scales"],
        naive_lod_stage_iterations,
    )
    if next_scale_idx != lod_state["current_scale_idx"]:
        previous_scale = lod_state["lod_scales"][lod_state["current_scale_idx"]]
        print(
            f"\n[ITER {iteration}] Promoting naive LoD training from resolution scale "
            f"{previous_scale} to {next_scale}"
        )
        lod_state["current_scale_idx"] = next_scale_idx
    return next_scale


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
    naive_lod_stage_iterations,
):
    opt.position_lr_max_steps = opt.iterations

    resolution_scales, lod_scales = train_module._prepare_resolution_scales(resolution_scales)
    finest_scale = resolution_scales[0]

    first_iter = 0
    tb_writer = train_module.prepare_output_and_logger(dataset)
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
        "lod_scale",
        "lod_stage_idx",
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
    train_module._initialize_csv_logger(train_metrics_csv, train_csv_fields)
    train_module._initialize_csv_logger(eval_metrics_csv, eval_csv_fields)

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

    viewpoint_dict, viewpoint_indices = train_module._build_viewpoint_stacks(
        scene, resolution_scales
    )
    initial_scale_idx, initial_scale = _resolve_naive_lod_scale(
        max(first_iter, 1),
        lod_scales,
        naive_lod_stage_iterations,
    )
    lod_state = {
        "lod_scales": lod_scales,
        "current_scale_idx": initial_scale_idx,
    }
    print(
        "Using naive LoD schedule with stage length "
        f"{naive_lod_stage_iterations} over scales {lod_scales}. "
        f"Starting at scale {initial_scale}."
    )

    fixed_wandb_eval_view = train_module._select_fixed_wandb_eval_view(scene, finest_scale)
    ema_loss_for_log = 0.0
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
            viewpoint_indices = train_module._refill_viewpoint_indices(
                len(viewpoint_dict[finest_scale])
            )
        viewpoint_idx = viewpoint_indices.pop(randint(0, len(viewpoint_indices) - 1))
        current_scale = _maybe_update_naive_lod_scale(
            iteration,
            lod_state,
            naive_lod_stage_iterations,
        )
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
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

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
            wandb = getattr(train_module, "wandb", None)
            if train_module.WANDB_FOUND and wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "train/photometric_loss": Ll1.item(),
                        "train/total_loss": loss.item(),
                        "train/kl_loss": probability_regularizer.item(),
                        "train/num_gaussians": gaussians.get_xyz.shape[0],
                        "train/lod_scale": current_scale,
                        "train/lod_stage_idx": lod_state["current_scale_idx"],
                    },
                    step=iteration,
                )

            train_module._append_csv_row(
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
                    "probability_probe_loss": None,
                    "probability_probe_loss_std": None,
                    "probability_probe_loss_ema": None,
                    "lod_scale": current_scale,
                    "lod_stage_idx": lod_state["current_scale_idx"],
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "iter_time_ms": iter_start.elapsed_time(iter_end),
                },
            )

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "LoD": current_scale,
                        "Stage": lod_state["current_scale_idx"],
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            spawn_interval = dataset.spawn_interval

            train_module.training_report(
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
                None,
                None,
                fixed_wandb_eval_view,
                eval_metrics_csv,
                eval_csv_fields,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                )

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

    parser = ArgumentParser(description="Naive LoD training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=np.arange(1000, 20001, 1000).tolist(),
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--resolution_scales", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--naive_lod_stage_iterations", type=int, default=5000)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="variational-3dgs")
    parser.add_argument("--wandb_name", type=str, default="naive-lod")
    args = parser.parse_args(sys.argv[1:])

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not train_module.WANDB_FOUND and not args.disable_wandb:
        print("wandb not available: proceeding without Weights & Biases logging")

    wandb = getattr(train_module, "wandb", None)
    wandb_context = nullcontext()
    if train_module.WANDB_FOUND and not args.disable_wandb and wandb is not None:
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
            args.naive_lod_stage_iterations,
        )

    print("\nTraining complete.")
