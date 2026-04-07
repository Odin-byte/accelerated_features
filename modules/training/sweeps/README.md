# W&B Sweep Runbook

## 1. Install dependencies

```bash
pip install -r code/accelerated_features/requirements.txt
```

## 2. Login to W&B

```bash
wandb login
```

## 3. Launch a pilot sweep (5 runs)

```bash
cd code/accelerated_features
python -m modules.training.wandb_sweep_launcher \
  --sweep_config modules/training/sweeps/bayes_default.yaml \
  --count 5 \
  --ckpt_save_path /path/to/checkpoints \
  --megadepth_root_path /path/to/dataMegaDepth \
  --synthetic_root_path /path/to/coco_20k \
  --training_type xfeat_default \
  --n_steps 30000 \
  --val_every 500
```

## 4. Launch multiple agents (parallel workers)

Start the same command in multiple terminals. Each agent will pull a different run from the same sweep.

## Notes

- Sweep objective is `Validation/Total_Loss`.
- Tuned parameters in default configs: `lr`, `gamma_steplr`, `dual_softmax_temp`, `patch_size`.
- The trainer remains runnable as a normal script without sweeps.
