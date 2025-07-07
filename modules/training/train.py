"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys

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
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='If True, preloads the model with weights and freezes everything but the heads')
    parser.add_argument('--eval_patience', type=int, default=False,
                        help='The number of evalulations allowed to happen without an improvement of the evaluation loss')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

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
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500, transfer_learning = False,
                       eval_images = 100, eval_patience = 20):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        self.transfer_learning = transfer_learning
        
        if transfer_learning:
            # If we want to adapt the model using transfer learning, preload the weights
            state_dict = torch.load("/home/docker/torch/code/accelerated_features/weights/xfeat.pt", map_location=self.dev)
            missing_keys, unexpected_keys = self.net.load_state_dict(state_dict)
            print(f"Sucessfully loaded pretrained weights into the XFeat class with {len(missing_keys)} missing keys.")

            # Freeze all layers
            for param in self.net.parameters():
                param.requires_grad = False

            for param in self.net.heatmap_head.parameters():
                param.requires_grad = True

            for param in self.net.keypoint_head.parameters():
                param.requires_grad = True

            for param in self.net.block_fusion.parameters():
                param.requires_grad = True

            for param in self.net.fine_matcher.parameters():
                param.requires_grad = True
            
            # Make the learning rate smaller
            lr = lr*0.1 # 10 percent of the original learning rate
            # synthetic_eval_root_path = synthetic_root_path + "eval"
            # print(synthetic_eval_root_path)
            # _, _, eval_imgs = next(os.walk(synthetic_eval_root_path))
            self.max_num_eval_batches = eval_images//batch_size
            self.last_eval_loss = None
            self.eval_patience = eval_patience
            self.patience_counter = 0
            self.best_net_state = None
            
        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 1500,
                                        num_test_imgs = 100,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 500
                                        )
            # self.eval_augmentor = AugmentationPipe(
            #                             img_dir = synthetic_eval_root_path,
            #                             device = self.dev, load_dataset= True,
            #                             batch_size = int(batch_size),
            #                             out_resolution = training_res, 
            #                             warp_resolution = training_res,
            #                             sides_crop = 0.1,
            #                             max_num_imgs = 100,
            #                             num_test_imgs = 5,
            #                             photometric = True,
            #                             geometric = True,
            #                             reload_step = 500,
            #                             eval=True
            #                             )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader = DataLoader(data, 
                                          batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name

    def eval(self, difficulty):
        """Calculate the used metrics on an unkown evaluation dataset.

        Args:
            difficulty (float): Sets the difficulty of the possible augmentations

        Returns:
            evaluation_metrics: Tuple of evaluation metrics in the following order
                                acc_coarse, acc_coarse_0, acc_coords, acc_pos, nb_coarse, loss, loss_coarse, loss_coord, loss_kp_pos, loss_l1
        """
        avg_acc_coarse_0 = []
        avg_acc_coarse = []
        avg_acc_coords = []
        avg_acc_pos = []

        avg_nb_coarse = []
        avg_loss = []
        avg_loss_coarse = []
        avg_loss_coord = []
        avg_loss_kp_pos = []
        avg_loss_l1 = []

        for batch in range(self.max_num_eval_batches):
                    p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty, train=False)
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)
                    p1s = p1s.mean(1, keepdim=True)
                    p2s = p2s.mean(1, keepdim=True)
                    p1 = p1s ; p2 = p2s
                    positives_c = positives_s_coarse
                    #Check if batch is corrupted with too few correspondences
                    is_corrupted = False
                    for p in positives_c:
                        if len(p) < 30:
                            is_corrupted = True

                    if is_corrupted:
                        continue

                    #Forward pass
                    feats1, kpts1, hmap1 = self.net(p1)
                    feats2, kpts2, hmap2 = self.net(p2)

                    loss_items = []

                    for b in range(len(positives_c)):
                        #Get positive correspondencies
                        pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                        #Grab features at corresponding idxs
                        m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                        m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                        #grab heatmaps at corresponding idxs
                        h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                        h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                        coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                        #Compute losses
                        loss_ds, conf = dual_softmax_loss(m1, m2)
                        loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                        loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                        loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                        loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                        acc_pos = (acc_pos1 + acc_pos2)/2

                        loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                        loss_items.append(loss_ds.unsqueeze(0))
                        loss_items.append(loss_coords.unsqueeze(0))
                        loss_items.append(loss_kp.unsqueeze(0))
                        loss_items.append(loss_kp_pos.unsqueeze(0))

                        if b == 0:
                            avg_acc_coarse_0.append(check_accuracy(m1, m2))

                    avg_acc_coarse.append(check_accuracy(m1, m2))
                    avg_acc_coords.append(acc_coords)
                    avg_acc_pos.append(acc_pos)
                    avg_nb_coarse.append(len(m1))
                    avg_loss.append(torch.cat(loss_items, -1).mean().detach().numpy())
                    avg_loss_coarse.append(loss_ds.item())
                    avg_loss_coord.append(loss_coords.item())
                    avg_loss_kp_pos.append(loss_kp_pos.item())
                    avg_loss_l1.append(loss_kp.item())

        return np.mean(avg_acc_coarse), np.mean(avg_acc_coarse_0), np.mean(avg_acc_coords), np.mean(avg_acc_pos), np.mean(avg_nb_coarse), np.mean(avg_loss), np.mean(avg_loss_coarse), np.mean(avg_loss_coord), np.mean(avg_loss_kp_pos), np.mean(avg_loss_l1)


    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
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
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                #Join megadepth & synthetic data
                with torch.inference_mode():
                    #RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
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
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                #Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    #Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    #Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    #grab heatmaps at corresponding idxs
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    #Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                if self.transfer_learning:
                    # Calculate evaluation
                    (
                        eval_acc_coarse,
                        eval_acc_coarse_0,
                        eval_acc_cords,
                        eval_acc_pos,
                        eval_nb_coarse,
                        eval_loss,
                        eval_loss_coarse,
                        eval_loss_coord,
                        eval_loss_kp_pos,
                        eval_loss_l1,
                    ) = self.eval(difficulty)

                    # Log metrics
                    self.writer.add_scalar('Loss/Eval/total', eval_loss.item(), i)
                    self.writer.add_scalar('Accuracy/Eval/coarse_synth', eval_acc_coarse_0, i)
                    self.writer.add_scalar('Accuracy/Eval/coarse_mdepth', eval_acc_coarse, i)
                    self.writer.add_scalar('Accuracy/Eval/fine_mdepth', eval_acc_cords, i)
                    self.writer.add_scalar('Accuracy/Eval/kp_position', eval_acc_pos, i)
                    self.writer.add_scalar('Loss/Eval/coarse', eval_loss_coarse, i)
                    self.writer.add_scalar('Loss/Eval/fine', eval_loss_coord, i)
                    self.writer.add_scalar('Loss/Eval/reliability', eval_loss_l1, i)
                    self.writer.add_scalar('Loss/Eval/keypoint_pos', eval_loss_kp_pos, i)
                    self.writer.add_scalar('Count/Eval/matches_coarse', eval_nb_coarse, i)

                    # Check if the weights perform better then the last based on the loss
                    if self.last_eval_loss != None and self.last_eval_loss > eval_loss:
                        self.patience_counter = 0
                        self.best_net_state = self.net.state_dict()
                    else:
                        self.patience_counter += 1
                    if self.patience_counter >= self.eval_patience:
                        torch.save(self.best_net_state, self.ckpt_save_path + f'/{self.model_name}_Best_Eval.pth')
                        print(f"Ended transferlearning training early after step {i}.")
                        break

                pbar.set_description( 'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                                                                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, loss_coarse, loss_coord, loss_l1, nb_coarse, loss_kp_pos, acc_pos) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)



if __name__ == '__main__':

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
        transfer_learning=args.transfer_learning,
        eval_images=100,
        eval_patience=100
    )

    #The most fun part
    trainer.train()
