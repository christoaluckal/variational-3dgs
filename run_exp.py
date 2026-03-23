import subprocess
import sys


SOURCE_PATH = "../Splat-20250821T013834Z-1-001/Splat/scene"
WANDB_PROJECT = "variational-3dgs"
PROBABILITY_LOD_INTERVAL = 50
PROBABILITY_LOD_MIN_ITERATIONS = 1000
PROBABILITY_LOSS_EMA_ALPHA = 0.1
PROBABILITY_LOD_PROBE_NUM_VIEWS = 4
PROBABILITY_LOD_BASELINE_WARMUP_PROBES = 5


EXPERIMENTS = [
    {"label": "baseline", "start_scale": 2, "resolution_scales": [2]},
    {"label": "lod", "start_scale": 2, "resolution_scales": [2, 4, 8]},
    {"label": "baseline", "start_scale": 4, "resolution_scales": [4]},
    {"label": "lod", "start_scale": 4, "resolution_scales": [4, 8]},
    {"label": "baseline", "start_scale": 8, "resolution_scales": [8]},
    # {"label": "lod", "start_scale": 8, "resolution_scales": [8]},
]

NUM_MODELS = [2, 3, 6, 10]


def build_experiment_name(experiment, num_models):
    return f"new-home-{experiment['label']}-r{experiment['start_scale']}-m{num_models}"


def build_command(experiment, num_models):
    experiment_name = build_experiment_name(experiment, num_models)
    command = [
        sys.executable,
        "train.py",
        "-s",
        SOURCE_PATH,
        "-m",
        f"./output/{experiment_name}",
        "--resolution_scales",
        *[str(scale) for scale in experiment["resolution_scales"]],
        "--num_models",
        str(num_models),
        "--probability_lod_interval",
        str(PROBABILITY_LOD_INTERVAL),
        "--probability_lod_min_iterations",
        str(PROBABILITY_LOD_MIN_ITERATIONS),
        "--probability_loss_ema_alpha",
        str(PROBABILITY_LOSS_EMA_ALPHA),
        "--probability_lod_probe_num_views",
        str(PROBABILITY_LOD_PROBE_NUM_VIEWS),
        "--probability_lod_baseline_warmup_probes",
        str(PROBABILITY_LOD_BASELINE_WARMUP_PROBES),
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_name",
        experiment_name,
        "--eval",
    ]
    if "probability_lod_thresholds" in experiment:
        command.extend(
            [
                "--probability_lod_thresholds",
                *[str(threshold) for threshold in experiment["probability_lod_thresholds"]],
            ]
        )
    return command


def main():
    for experiment in EXPERIMENTS:
        for num_models in NUM_MODELS:
            command = build_command(experiment, num_models)
            print("Running:", " ".join(command))
            try:
                subprocess.run(command, check=True)
            except:
                pass


if __name__ == "__main__":
    main()
