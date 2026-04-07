"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import glob
import os
import time
import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

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
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Polar patch size used by the descriptor backbone. Default is 32.')
    parser.add_argument('--wandb_project', type=str, default='xfeat_polar_patch_descriptor_matching',
                        help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default='dominik-w-ho',
                        help='W&B entity/team name.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import wandb

import numpy as np
import random

from modules.model_polar_patch_match_only import XFeatPolarPatchDescriptorGen
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *
from modules.interpolator import InterpolateSparse2d

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_default',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False, plot_test = False,
                       save_ckpt_every = 500, val_ratio = 0.05, val_every = 500, val_batches = 10, val_seed = 42,
                       num_val_synth_imgs = 100, dual_softmax_temp = 2.0, patch_size = 32,
                       wandb_project = 'xfeat_polar_patch_descriptor_matching', wandb_entity = 'dominik-w-ho'):

        self.batch_size = batch_size
        self.steps = n_steps
        self.dual_softmax_temp = dual_softmax_temp
        self.patch_size = patch_size
        self.best_val_total = float('inf')

        run_config = {
            'megadepth_root_path': megadepth_root_path,
            'synthetic_root_path': synthetic_root_path,
            'ckpt_save_path': ckpt_save_path,
            'training_type': model_name,
            'batch_size': batch_size,
            'n_steps': n_steps,
            'lr': lr,
            'gamma_steplr': gamma_steplr,
            'training_res': list(training_res),
            'device_num': device_num,
            'dry_run': dry_run,
            'save_ckpt_every': save_ckpt_every,
            'val_ratio': val_ratio,
            'val_every': val_every,
            'val_batches': val_batches,
            'val_seed': val_seed,
            'num_val_synth_imgs': num_val_synth_imgs,
            'dual_softmax_temp': dual_softmax_temp,
            'patch_size': patch_size,
        }

        if wandb.run is None:
            self.wandb_run = wandb.init(
                            project=wandb_project,
                            name=f"{model_name}_train_{time.strftime('%Y_%m_%d-%H_%M_%S')}",
                            entity=wandb_entity,
                            config=run_config,
            )
        else:
            self.wandb_run = wandb.run
            self.wandb_run.config.update(run_config, allow_val_change=True)

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatPolarPatchDescriptorGen(patch_size=self.patch_size).to(self.dev)

        #Setup optimizer 
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        # Validation setup
        self.val_ratio = val_ratio
        self.val_every = val_every
        self.val_batches = val_batches
        self.val_seed = val_seed
        self.num_val_synth_imgs = num_val_synth_imgs

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor_train = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
            
            self.augmentor_val = AugmentationPipe(
                                        img_dir=synthetic_root_path,
                                        device=self.dev,
                                        load_dataset=True,
                                        batch_size=int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution=training_res,
                                        warp_resolution=training_res,
                                        sides_crop=0.1,
                                        max_num_imgs=3000,
                                        num_test_imgs=num_val_synth_imgs,
                                        photometric=True,
                                        geometric=True,
                                        reload_step=10**12,
                                    )
        else:
            self.augmentor_train = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        self.dry_run = dry_run

        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = sorted(glob.glob(TRAIN_NPZ_ROOT + '/*.npz'))

            rng = random.Random(val_seed)
            rng.shuffle(npz_paths)

            n_val = max(1, int(len(npz_paths) * val_ratio))
            val_npz_paths = npz_paths[:n_val]
            train_npz_paths = npz_paths[n_val:]
            train_data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(train_npz_paths, desc="[MegaDepth] Loading TRAIN metadata")] )
            
            val_data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(val_npz_paths, desc="[MegaDepth] Loading VAL metadata")] )
        
            if self.dry_run:
                self.train_data_loader = DataLoader(train_data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=False)
                self.val_data_loader = DataLoader(val_data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=False)
            else:
                self.train_data_loader = DataLoader(train_data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=True)
                self.val_data_loader = DataLoader(val_data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=False)
                
            self.data_iter = iter(self.train_data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        if self.dry_run:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.plot_test = plot_test
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.model_name = model_name

    def compute_batch_loss(self, p1, p2, positives_c):
        loss_items = []

        for b in range(len(positives_c)):
            #Get positive correspondencies
            pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

            #Generate descriptors at kpts positions
            feats1 = self.net(p1[b].unsqueeze(0), pts1.unsqueeze(0)).squeeze(0)
            feats2 = self.net(p2[b].unsqueeze(0), pts2.unsqueeze(0)).squeeze(0)

            #Compute losses
            loss_ds, conf = dual_softmax_loss(feats1, feats2, temp=self.dual_softmax_temp)

            loss_items.append(loss_ds.unsqueeze(0))

        loss = torch.cat(loss_items, -1).mean()
        return loss
    
    def is_corrupted_batch(self, positives_c, min_matches=30):
        return any(len(p) < min_matches for p in positives_c)
    
    @torch.no_grad()
    def validate(self, current_step):
        print(f"Running validation at step {current_step}...")
        self.net.eval()

        metrics = {
            'md_loss': [],
            'syn_loss': [],
        }

        if hasattr(self, 'val_data_loader'):
            for batch_idx, d in enumerate(self.val_data_loader):
                if batch_idx >= self.val_batches:
                    break

                for k in d.keys():
                    if isinstance(d[k], torch.Tensor):
                        d[k] = d[k].to(self.dev)

                p1 = d['image0'].mean(1, keepdim=True)
                p2 = d['image1'].mean(1, keepdim=True)
                positives_md = megadepth_warper.spvs_coarse(d, 8)

                if self.is_corrupted_batch(positives_md):
                    continue

                loss = self.compute_batch_loss(p1, p2, positives_md)
                metrics['md_loss'].append(loss.item())

        if self.augmentor_train is not None:
            for _ in range(self.val_batches):
                p1s, p2s, H1, H2 = make_batch(self.augmentor_val, difficulty=0.10, train=False)
                h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                _, positives_syn = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor_val, h_coarse, w_coarse)

                p1s = p1s.mean(1, keepdim=True)
                p2s = p2s.mean(1, keepdim=True)

                if self.is_corrupted_batch(positives_syn):
                    continue

                loss = self.compute_batch_loss(p1s, p2s, positives_syn)
                metrics['syn_loss'].append(loss.item())

        md_loss = np.mean(metrics['md_loss']) if metrics['md_loss'] else float('nan')
        syn_loss = np.mean(metrics['syn_loss']) if metrics['syn_loss'] else float('nan')

        if self.model_name == 'xfeat_default':
            total = np.nanmean([md_loss, syn_loss])
        elif self.model_name == 'xfeat_megadepth':
            total = md_loss
        else:
            total = syn_loss

        if np.isnan(total):
            total = float('inf')

        self.wandb_run.log({
            'Validation/Total_Loss': total,
            'Validation/MegaDepth_Loss': md_loss,
            'Validation/Synthetic_Loss': syn_loss,
        }, step=current_step)

        if total < self.best_val_total:
            self.best_val_total = total
            best_path = self.ckpt_save_path + f'/{self.model_name}_best.pth'
            torch.save(self.net.state_dict(), best_path)

        self.net.train()

    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor_train is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor_train, difficulty)
        
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.train_data_loader)
                            d = next(self.data_iter)

                    if self.augmentor_train is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor_train, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor_train is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor_train, h_coarse, w_coarse)

                #Join megadepth & synthetic data
                with torch.inference_mode():
                    #RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor_train is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    #Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                #Check if batch is corrupted with too few correspondences
                is_corrupted = self.is_corrupted_batch(positives_c, min_matches=30)
                if is_corrupted:
                    print(f"Batch {i} is corrupted with too few correspondences")
                    self.wandb_run.log({'Training/Corrupted_Batch': 1}, step=i)
                if is_corrupted:
                    continue

                loss_items = []

                for b in range(len(positives_c)):
                    # print(len(positives_c))
                    # print(f"len(positives_c[{b}]): {len(positives_c[b])}")
                    #Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2] * 8, positives_c[b][:, 2:] * 8
                    # Print the min and max of pts1 and pts2 to check if they are within the image bounds
                    print(f"pts1 min: {pts1.min(dim=0).values}, max: {pts1.max(dim=0).values}")
                    print(f"pts2 min: {pts2.min(dim=0).values}, max: {pts2.max(dim=0).values}")

                    #Generate descriptors at kpts positions
                    feats1 = self.net(p1[b].unsqueeze(0), pts1.unsqueeze(0)).squeeze(0)
                    feats2 = self.net(p2[b].unsqueeze(0), pts2.unsqueeze(0)).squeeze(0)
                    # print(f"Shapes of feats1 and feats2: {feats1.shape}, {feats2.shape}")

                    #Compute losses
                    loss_ds, conf = dual_softmax_loss(feats1, feats2, temp=self.dual_softmax_temp)
                    # print(f"loss_ds: {loss_ds.item()}, conf shape: {conf.shape}")

                    loss_items.append(loss_ds.unsqueeze(0))

                loss = torch.cat(loss_items, -1).mean()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                pbar.set_description( 'Loss: {:.4f} '.format(loss.item()) )
                pbar.update(1)

                # Log metrics
                self.wandb_run.log({
                    'Training/Total_Loss': loss.item(),
                    'Training/LR': self.scheduler.get_last_lr()[0],
                }, step=i)

                if (i+1) % self.val_every == 0:
                    self.validate(i+1)

            self.wandb_run.finish()



def run_from_args(args):
    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        val_ratio=args.val_ratio,
        val_every=args.val_every,
        val_batches=args.val_batches,
        val_seed=args.val_seed,
        num_val_synth_imgs=args.num_val_synth_imgs,
        dual_softmax_temp=args.dual_softmax_temp,
        patch_size=args.patch_size,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    trainer.train()


if __name__ == '__main__':
    args = parse_arguments()
    run_from_args(args)
