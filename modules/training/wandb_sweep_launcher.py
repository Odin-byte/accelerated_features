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
    parser.add_argument('--wandb_project', type=str, default='xfeat_polar_patch_descriptor_matching',
                        help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default='dominik-w-ho',
                        help='W&B entity/team name.')

    parser.add_argument('--megadepth_root_path', type=str, default='/home/docker/torch/dataMegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/home/docker/torch/data/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--max_num_keypoints_patch_gen', type=int, default=3000,
                        help='Maximum number of keypoints to allowed for patch generation per image. Default is 3000.')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Ratio of MegaDepth scenes to use for validation. Default is 0.05 (5%).')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Validate every N steps. Default is 500.')
    parser.add_argument('--val_batches', type=int, default=10,
                        help='Number of batches run during each validation phase. Default is 10.')
    parser.add_argument('--val_seed', type=int, default=42,
                        help='Random seed for validation. Default is 42.')
    parser.add_argument('--num_val_synth_imgs', type=int, default=100,
                        help='Number of synthetic images to use for validation. Default is 100.')
    parser.add_argument('--dual_softmax_temp', type=float, default=2.0,
                        help='Temperature parameter for dual softmax descriptor loss. Default is 2.0.')
    parser.add_argument('--patch_size', type=int, default=41,
                        help='Polar patch size used by the descriptor backbone. Default is 41.')
    parser.add_argument('--log_polar', action='store_true',
                        help='Whether to use log-polar coordinates. Default is False.')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Number of validation checks without improvement before stopping. 0 disables early stopping.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum validation loss decrease required to count as an improvement.')
    return parser.parse_args()


def load_sweep_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_runtime_config(base_args, sweep_cfg):
    cfg = dict(base_args)
    for key in ["lr", "gamma_steplr", "dual_softmax_temp", "patch_size", "log_polar"]:
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
            log_polar=runtime_cfg["log_polar"],
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
