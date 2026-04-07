import argparse
import os

import wandb
import yaml

from modules.training.train_patches_only import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Launch W&B sweeps for patch-only training.")
    parser.add_argument(
        "--sweep_config",
        type=str,
        default="modules/training/sweeps/bayes_default.yaml",
        help="Path to sweep YAML config.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of sweep runs for this agent.")

    parser.add_argument("--megadepth_root_path", type=str, default="/home/docker/torch/data/Megadepth")
    parser.add_argument("--synthetic_root_path", type=str, default="/home/docker/torch/data/coco_20k")
    parser.add_argument("--ckpt_save_path", type=str, required=True)
    parser.add_argument(
        "--training_type",
        type=str,
        default="xfeat_default",
        choices=["xfeat_default", "xfeat_synthetic", "xfeat_megadepth"],
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=160000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma_steplr", type=float, default=0.5)
    parser.add_argument(
        "--training_res",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(800, 608),
        help="Training resolution as width,height.",
    )
    parser.add_argument("--device_num", type=str, default="0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_ckpt_every", type=int, default=500)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--num_val_synth_imgs", type=int, default=100)
    parser.add_argument("--dual_softmax_temp", type=float, default=2.0)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="xfeat_polar_patch_descriptor_matching")
    parser.add_argument("--wandb_entity", type=str, default="dominik-w-ho")

    return parser.parse_args()


def load_sweep_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_runtime_config(base_args, sweep_cfg):
    cfg = dict(base_args)
    for key in ["lr", "gamma_steplr", "dual_softmax_temp", "patch_size"]:
        if key in sweep_cfg:
            cfg[key] = sweep_cfg[key]

    # Keep values in safe ranges to avoid failed runs.
    cfg["patch_size"] = int(max(8, cfg["patch_size"]))
    cfg["dual_softmax_temp"] = float(max(1e-4, cfg["dual_softmax_temp"]))
    return cfg


def run_single_training(base_args):
    run = wandb.init(project=base_args["wandb_project"], entity=base_args["wandb_entity"])
    try:
        runtime_cfg = merge_runtime_config(base_args, dict(wandb.config))

        os.environ["CUDA_VISIBLE_DEVICES"] = runtime_cfg["device_num"]

        trainer = Trainer(
            megadepth_root_path=runtime_cfg["megadepth_root_path"],
            synthetic_root_path=runtime_cfg["synthetic_root_path"],
            ckpt_save_path=runtime_cfg["ckpt_save_path"],
            model_name=runtime_cfg["training_type"],
            batch_size=runtime_cfg["batch_size"],
            n_steps=runtime_cfg["n_steps"],
            lr=runtime_cfg["lr"],
            gamma_steplr=runtime_cfg["gamma_steplr"],
            training_res=runtime_cfg["training_res"],
            device_num=runtime_cfg["device_num"],
            dry_run=runtime_cfg["dry_run"],
            save_ckpt_every=runtime_cfg["save_ckpt_every"],
            val_ratio=runtime_cfg["val_ratio"],
            val_every=runtime_cfg["val_every"],
            val_batches=runtime_cfg["val_batches"],
            val_seed=runtime_cfg["val_seed"],
            num_val_synth_imgs=runtime_cfg["num_val_synth_imgs"],
            dual_softmax_temp=runtime_cfg["dual_softmax_temp"],
            patch_size=runtime_cfg["patch_size"],
            wandb_project=runtime_cfg["wandb_project"],
            wandb_entity=runtime_cfg["wandb_entity"],
        )
        trainer.train()
    finally:
        if run is not None and wandb.run is not None:
            wandb.finish()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    base_args = vars(args)
    sweep_config = load_sweep_config(args.sweep_config)

    sweep_project = sweep_config.get("project", args.wandb_project)
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_project, entity=args.wandb_entity)

    wandb.agent(
        sweep_id,
        function=lambda: run_single_training(base_args),
        count=args.count,
        project=sweep_project,
        entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
