import subprocess
import sys


SOURCE_PATH = "../Splat-20250821T013834Z-1-001/Splat/scene"
WANDB_PROJECT = "variational-3dgs"


EXPERIMENTS = [
    {"label": "baseline", "start_scale": 2, "resolution_scales": [2]},
    # {"label": "lod", "start_scale": 2, "resolution_scales": [2, 4, 8]},
    # {"label": "baseline", "start_scale": 4, "resolution_scales": [4]},
    # {"label": "lod", "start_scale": 4, "resolution_scales": [4, 8]},
    # {"label": "baseline", "start_scale": 8, "resolution_scales": [8]},
    # {"label": "lod", "start_scale": 8, "resolution_scales": [8]},
]

NUM_MODELS = [3, 6]


def build_experiment_name(experiment, num_models):
    return f"home-{experiment['label']}-r{experiment['start_scale']}-m{num_models}"


def build_command(experiment, num_models):
    experiment_name = build_experiment_name(experiment, num_models)
    return [
        sys.executable,
        "train_bidirectional_lod.py",
        "-s",
        SOURCE_PATH,
        "-m",
        f"./output/{experiment_name}",
        "--resolution_scales",
        *[str(scale) for scale in experiment["resolution_scales"]],
        "--num_models",
        str(num_models),
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_name",
        experiment_name,
        "--eval",
    ]


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
