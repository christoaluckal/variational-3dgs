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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import csv
from lpipsPyTorch import lpips
from contextlib import nullcontext

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

if False: 
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
else: 
    TENSORBOARD_FOUND = False

def _build_viewpoint_stacks(scene, resolution_scales):
    viewpoint_dict = {
        scale: scene.getTrainCameras(scale=scale)
        for scale in resolution_scales
    }
    viewpoint_indices = list(range(len(viewpoint_dict[resolution_scales[0]])))
    return viewpoint_dict, viewpoint_indices


def _refill_viewpoint_indices(num_viewpoints):
    return torch.randperm(num_viewpoints).tolist()


def _select_probability_probe_indices(num_viewpoints, num_probe_views):
    if num_viewpoints <= 0:
        raise ValueError("At least one training viewpoint is required for probability probes.")
    if num_probe_views <= 0:
        raise ValueError("probability_lod_probe_num_views must be at least 1.")

    probe_count = min(num_viewpoints, num_probe_views)
    if probe_count == num_viewpoints:
        return list(range(num_viewpoints))

    return [
        ((2 * probe_idx + 1) * num_viewpoints) // (2 * probe_count)
        for probe_idx in range(probe_count)
    ]


def _prepare_resolution_scales(resolution_scales):
    unique_scales = list(dict.fromkeys(resolution_scales))
    if not unique_scales:
        raise ValueError("resolution_scales must contain at least one scale.")
    return unique_scales, list(reversed(unique_scales))


def _validate_probability_lod_thresholds(probability_lod_thresholds, lod_scales):
    expected_thresholds = max(len(lod_scales) - 1, 0)
    if len(probability_lod_thresholds) != expected_thresholds:
        raise ValueError(
            "--probability_lod_thresholds must provide exactly "
            f"{expected_thresholds} threshold ratios for LoD scales {lod_scales}."
        )


def _resolve_probability_lod_thresholds(probability_lod_thresholds, lod_scales):
    expected_thresholds = max(len(lod_scales) - 1, 0)
    if probability_lod_thresholds:
        _validate_probability_lod_thresholds(probability_lod_thresholds, lod_scales)
        return probability_lod_thresholds

    if expected_thresholds == 0:
        return []
    if expected_thresholds == 1:
        return [0.3]
    if expected_thresholds == 2:
        return [0.5, 0.3]

    raise ValueError(
        "No default probability_lod_thresholds are defined for "
        f"{expected_thresholds} LoD transitions. Please provide them explicitly."
    )


def _maybe_update_lod_scale(
    iteration,
    lod_state,
    probability_loss,
    probability_loss_ema_alpha,
    probability_lod_thresholds,
    probability_lod_min_iterations,
    probability_lod_baseline_warmup_probes,
):
    probability_value = float(probability_loss.item())
    if lod_state["ema_probability_loss"] is None:
        lod_state["ema_probability_loss"] = probability_value
        lod_state["baseline_probability_losses"].append(probability_value)
        if probability_lod_baseline_warmup_probes == 1:
            lod_state["baseline_probability_loss"] = probability_value
        return

    ema_probability_loss = lod_state["ema_probability_loss"]
    ema_probability_loss = (
        probability_loss_ema_alpha * probability_value
        + (1.0 - probability_loss_ema_alpha) * ema_probability_loss
    )
    lod_state["ema_probability_loss"] = ema_probability_loss

    if lod_state["baseline_probability_loss"] is None:
        lod_state["baseline_probability_losses"].append(probability_value)
        if len(lod_state["baseline_probability_losses"]) < probability_lod_baseline_warmup_probes:
            return

        baseline_tensor = torch.tensor(
            lod_state["baseline_probability_losses"], dtype=torch.float32
        )
        lod_state["baseline_probability_loss"] = float(torch.median(baseline_tensor).item())
        lod_state["ema_probability_loss"] = lod_state["baseline_probability_loss"]
        print(
            f"\n[ITER {iteration}] Established probability-loss baseline "
            f"from {len(lod_state['baseline_probability_losses'])} fixed-view probes: "
            f"{lod_state['baseline_probability_loss']:.6f}"
        )
        return

    if lod_state["current_scale_idx"] >= len(lod_state["lod_scales"]) - 1:
        return

    if iteration - lod_state["last_scale_change_iteration"] < probability_lod_min_iterations:
        return

    threshold_ratio = probability_lod_thresholds[lod_state["current_scale_idx"]]
    threshold = lod_state["baseline_probability_loss"] * threshold_ratio
    if ema_probability_loss <= threshold:
        lod_state["current_scale_idx"] += 1
        lod_state["last_scale_change_iteration"] = iteration
        next_scale = lod_state["lod_scales"][lod_state["current_scale_idx"]]
        print(
            f"\n[ITER {iteration}] Promoting LoD training to resolution scale {next_scale} "
            f"(EMA probability loss {ema_probability_loss:.6f} <= threshold {threshold:.6f})"
        )


def _to_wandb_image(image_tensor, caption):
    image = torch.clamp(image_tensor, 0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy()
    return wandb.Image(image, caption=caption)


def _uncertainty_to_viridis_image(uncertainty_tensor):
    if uncertainty_tensor.dim() == 3:
        if uncertainty_tensor.shape[0] == 1:
            uncertainty_map = uncertainty_tensor[0]
        else:
            uncertainty_map = uncertainty_tensor.mean(dim=0)
    elif uncertainty_tensor.dim() == 2:
        uncertainty_map = uncertainty_tensor
    else:
        raise ValueError(
            "uncertainty_tensor must have shape [C, H, W] or [H, W] for visualization."
        )

    unc_min = uncertainty_map.amin()
    unc_max = uncertainty_map.amax()
    if float((unc_max - unc_min).item()) < 1e-8:
        normalized = torch.zeros_like(uncertainty_map)
    else:
        normalized = (uncertainty_map - unc_min) / (unc_max - unc_min)

    viridis_stops = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    viridis_colors = torch.tensor(
        [
            [0.267004, 0.004874, 0.329415],
            [0.229739, 0.322361, 0.545706],
            [0.127568, 0.566949, 0.550556],
            [0.369214, 0.788888, 0.382914],
            [0.993248, 0.906157, 0.143936],
        ],
        dtype=normalized.dtype,
        device=normalized.device,
    )

    flat = normalized.reshape(-1)
    upper_idx = torch.bucketize(flat, viridis_stops, right=True)
    upper_idx = upper_idx.clamp(1, len(viridis_stops) - 1)
    lower_idx = upper_idx - 1

    lower_stops = viridis_stops[lower_idx]
    upper_stops = viridis_stops[upper_idx]
    interp = ((flat - lower_stops) / (upper_stops - lower_stops).clamp_min(1e-8)).unsqueeze(-1)
    mapped = (
        (1.0 - interp) * viridis_colors[lower_idx]
        + interp * viridis_colors[upper_idx]
    )

    return mapped.reshape(*normalized.shape, 3).permute(2, 0, 1)


def _initialize_csv_logger(csv_path, fieldnames):
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(csv_path, fieldnames, row):
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def _compute_probability_probe_loss(
    viewpoint_dict,
    finest_scale,
    probability_probe_indices,
    gaussians,
    pipe,
    background,
):
    probability_losses = []
    for viewpoint_idx in probability_probe_indices:
        viewpoint_cam = viewpoint_dict[finest_scale][viewpoint_idx]
        probability_render = forward_k_times(
            viewpoint_cam, gaussians, pipe, background
        )
        probability_gt = viewpoint_cam.original_image.cuda()
        probability_loss = nll_kernel_density(
            probability_render["comp_rgbs"].permute(1, 2, 3, 0),
            probability_render["comp_std"],
            probability_gt,
        )
        probability_losses.append(probability_loss)

    stacked_losses = torch.stack(probability_losses)
    return stacked_losses.mean(), stacked_losses


def _select_fixed_wandb_eval_view(scene, eval_scale):
    test_cameras = scene.getTestCameras(scale=eval_scale)
    if not test_cameras:
        return None
    # random_bytes = os.urandom(8)
    # view_idx = int.from_bytes(random_bytes, byteorder="big") % len(test_cameras)

    view_idx = len(test_cameras)//2

    return test_cameras[view_idx]


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
    probability_lod_thresholds,
    probability_lod_interval,
    probability_lod_min_iterations,
    probability_loss_ema_alpha,
    probability_lod_probe_num_views,
    probability_lod_baseline_warmup_probes,
):
    opt.position_lr_max_steps = opt.iterations

    resolution_scales, lod_scales = _prepare_resolution_scales(resolution_scales)
    finest_scale = resolution_scales[0]
    probability_lod_thresholds = _resolve_probability_lod_thresholds(
        probability_lod_thresholds, lod_scales
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
        "lod_scale",
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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_dict, viewpoint_indices = _build_viewpoint_stacks(scene, resolution_scales)
    probability_probe_indices = _select_probability_probe_indices(
        len(viewpoint_dict[finest_scale]), probability_lod_probe_num_views
    )
    print(
        "Using fixed probability probe viewpoints "
        f"{probability_probe_indices} at finest scale {finest_scale}"
    )
    lod_state = {
        "lod_scales": lod_scales,
        "current_scale_idx": 0,
        "last_scale_change_iteration": first_iter,
        "ema_probability_loss": None,
        "baseline_probability_loss": None,
        "baseline_probability_losses": [],
    }
    fixed_wandb_eval_view = _select_fixed_wandb_eval_view(scene, finest_scale)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):      
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if not viewpoint_indices:
            viewpoint_indices = _refill_viewpoint_indices(len(viewpoint_dict[finest_scale]))
        viewpoint_idx = viewpoint_indices.pop(randint(0, len(viewpoint_indices) - 1))
        current_scale = lod_state["lod_scales"][lod_state["current_scale_idx"]]
        viewpoint_cam = viewpoint_dict[current_scale][viewpoint_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        model_id = torch.randint(0, gaussians.n_models, (1,)).item()
        gaussians.model_id = model_id

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss_kl_scal = gaussians.compute_kl_uniform_scal()
        loss_kl_xyz = gaussians.compute_kl_xyz()
        loss_kl_opacity = gaussians.compute_kl_opacity()
        probability_regularizer = loss_kl_scal + loss_kl_xyz + loss_kl_opacity

        loss += probability_regularizer

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
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
                _maybe_update_lod_scale(
                    iteration,
                    lod_state,
                    probability_loss,
                    probability_loss_ema_alpha,
                    probability_lod_thresholds,
                    probability_lod_min_iterations,
                    probability_lod_baseline_warmup_probes,
                )
            if WANDB_FOUND and wandb.run is not None:
                train_log = {
                    "train/photometric_loss": Ll1.item(),
                    "train/total_loss": loss.item(),
                    "train/kl_scale_loss": loss_kl_scal.item(),
                    "train/num_gaussians": gaussians.get_xyz.shape[0],
                    "train/lod_scale": current_scale,
                }
                if probability_loss is not None:
                    train_log["train/probability_probe_loss"] = probability_loss.item()
                if probability_loss_std is not None:
                    train_log["train/probability_probe_loss_std"] = probability_loss_std
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
                    "lod_scale": current_scale,
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "iter_time_ms": iter_start.elapsed_time(iter_end),
                },
            )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_metrics = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "LoD": current_scale,
                }
                if lod_state["ema_probability_loss"] is not None:
                    progress_metrics["ProbEMA"] = f"{lod_state['ema_probability_loss']:.{7}f}"
                progress_bar.set_postfix(progress_metrics)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            spawn_interval = dataset.spawn_interval

            # Log and save
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
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % spawn_interval == 0:  # spawn interval should be a multiple of densification interval
                    gaussians.spawn(scene.cameras_extent)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

import torchvision

def training_report(
    dataset,
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene : Scene,
    renderFunc,
    renderArgs,
    eval_scale,
    training_scale,
    loss_kl_scal,
    probability_regularizer,
    probability_loss,
    ema_probability_loss,
    fixed_wandb_eval_view,
    eval_metrics_csv,
    eval_csv_fields,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train/lod_scale', training_scale, iteration)
        tb_writer.add_scalar('train_loss_patches/kl_scale_loss', loss_kl_scal.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/probability_regularizer', probability_regularizer.item(), iteration)
        if probability_loss is not None:
            tb_writer.add_scalar('train_loss_patches/probability_probe_loss', probability_loss.item(), iteration)
        if ema_probability_loss is not None:
            tb_writer.add_scalar('train_loss_patches/probability_probe_loss_ema', ema_probability_loss, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(scale=eval_scale)}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras(scale=eval_scale)[idx % len(scene.getTrainCameras(scale=eval_scale))] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                _append_csv_row(
                    eval_metrics_csv,
                    eval_csv_fields,
                    {
                        "iteration": iteration,
                        "split": config["name"],
                        "eval_scale": eval_scale,
                        "num_cameras": len(config["cameras"]),
                        "l1": l1_test.item(),
                        "psnr": psnr_test.item(),
                    },
                )
                if WANDB_FOUND and wandb.run is not None and config['name'] == 'test':
                    eval_log = {
                        "eval/L1": l1_test.item(),
                        "eval/PSNR": psnr_test.item(),
                    }
                    if fixed_wandb_eval_view is not None:
                        sample_gt = torch.clamp(fixed_wandb_eval_view.original_image.to("cuda"), 0.0, 1.0)
                        sample_out = forward_k_times(
                            fixed_wandb_eval_view, scene.gaussians, renderArgs[0], renderArgs[1]
                        )
                        sample_render = torch.clamp(sample_out["comp_rgb"], 0.0, 1.0)
                        sample_unc = _uncertainty_to_viridis_image(sample_out["comp_std"])
                        eval_log["eval/images/ground_truth"] = _to_wandb_image(
                            sample_gt, f"{fixed_wandb_eval_view.image_name} gt"
                        )
                        eval_log["eval/images/render_rgb"] = _to_wandb_image(
                            sample_render, f"{fixed_wandb_eval_view.image_name} render"
                        )
                        eval_log["eval/images/render_uncertainty"] = _to_wandb_image(
                            sample_unc, f"{fixed_wandb_eval_view.image_name} uncertainty"
                        )
                    wandb.log(eval_log, step=iteration)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        if iteration == testing_iterations[-1]: 
            render_set(dataset, scene, renderArgs[0], eval_scale)

from utils.image_utils import psnr, nll_kernel_density, ause_br
from gaussian_renderer import render, forward_k_times
from os import makedirs


def render_set(dataset, scene, pipeline, scale):
    gaussians, views = scene.gaussians, scene.getTestCameras(scale=scale)

    if len(views) == 0:
        print("\nNo test cameras available for render_set; skipping final evaluation render.")
        return

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    eval_depth = True if dataset.dataset_name == "LF" else False

    scene_name = scene.model_path.split("/")[-1]

    render_path = f"{scene.model_path}/test/ours_7000/renders"
    gts_path = f"{scene.model_path}/test/ours_7000/gt"
    unc_path = f"{scene.model_path}/test/ours_7000/unc"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(unc_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt = view.original_image[0:3, :, :]
        out = forward_k_times(view, gaussians, pipeline, background)
        mean = out['comp_rgb'].detach()
        rgbs = out['comp_rgbs'].detach()
        std = out['comp_std'].detach()
        depths = out['depths'].detach()

        mae = ((mean - gt)).abs()

        ause_mae, ause_err_mae, ause_err_by_var_mae = ause_br(std.reshape(-1), mae.reshape(-1), err_type='mae')
        mean_nll = nll_kernel_density(rgbs.permute(1,2,3,0), std, gt)

        psnr_all += psnr(mean, gt).mean().item()
        ssim_all += ssim(mean, gt).mean().item()
        lpips_all += lpips(mean, gt, net_type="vgg").mean().item()

        ause_mae_all += ause_mae.item()
        mean_nll_all += mean_nll.item()

        if eval_depth: 
            depths = depths * scene.depth_scale

            depth = depths.mean(dim=0)
            depth_std = depths.std(dim=0)
            depth_gt = view.depth

            depth_mae = ((depth - depth_gt)).abs()
            depth_ause_mae, depth_ause_err_mae, depth_ause_err_by_var_mae = ause_br(depth_std.reshape(-1), depth_mae.reshape(-1), err_type='mae')
            depth_ause_mae_all += depth_ause_mae

        torchvision.utils.save_image(mean, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(
            _uncertainty_to_viridis_image(std),
            os.path.join(unc_path, '{0:05d}'.format(idx) + ".png")
        )


    psnr_all /= len(views)
    ause_mae_all /= len(views)
    mean_nll_all /= len(views)
    ssim_all /= len(views)
    lpips_all /= len(views)

    depth_ause_mae_all /= len(views)

    csv_file = f"output/eval_results_{dataset.dataset_name}.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if eval_depth: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all} Depth AUSE {depth_ause_mae_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all])
        else: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all])

if __name__ == "__main__":
    # Set up command line argument parser
    import numpy as np
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1500, 2500, 3500, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=np.arange(1000,20001,1000).tolist())
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--resolution_scales", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--probability_lod_thresholds", nargs="*", type=float, default=[])
    parser.add_argument("--probability_lod_interval", type=int, default=50)
    parser.add_argument("--probability_lod_min_iterations", type=int, default=1000)
    parser.add_argument("--probability_loss_ema_alpha", type=float, default=0.1)
    parser.add_argument("--probability_lod_probe_num_views", type=int, default=4)
    parser.add_argument("--probability_lod_baseline_warmup_probes", type=int, default=5)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="variational-3dgs")
    parser.add_argument("--wandb_name", type=str, default="probability-lod")
    args = parser.parse_args(sys.argv[1:])

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not WANDB_FOUND and not args.disable_wandb:
        print("wandb not available: proceeding without Weights & Biases logging")

    wandb_context = nullcontext()
    if WANDB_FOUND and not args.disable_wandb:
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
            args.probability_lod_thresholds,
            args.probability_lod_interval,
            args.probability_lod_min_iterations,
            args.probability_loss_ema_alpha,
            args.probability_lod_probe_num_views,
            args.probability_lod_baseline_warmup_probes,
        )

    # All done
    print("\nTraining complete.")
