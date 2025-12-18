"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time

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
    parser.add_argument('--batch_size', type=int, default=3,
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
import cv2
import random

from modules.two_staged_model import *
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
                       training_res = (800, 608), device_num="0", dry_run = False, plot_test = True,
                       save_ckpt_every = 500):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatPolarPatchesModel().to(self.dev)

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
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        self.dry_run = dry_run

        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )
        
            if self.dry_run:
                self.data_loader = DataLoader(data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=False)
            else:
                self.data_loader = DataLoader(data, 
                                              batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                              shuffle=True)
                
            self.data_iter = iter(self.data_loader)

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
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name

    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap
    
    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

		#Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
    
    def create_test_plot(self, transform_data, current_step, min_cossim = 0.82):
        top_k = 4096
        tensor_image0 = transform_data["image0"][0]
        tensor_image1 = transform_data["image1"][0]

        # Check if there is a batch dim
        if tensor_image0.dim() < 4:
            tensor_image0 = tensor_image0.unsqueeze(0)

        if tensor_image1.dim() < 4:
            tensor_image1 = tensor_image1.unsqueeze(0)

        B = tensor_image0.shape[0]
        H_orig = tensor_image1.shape[-2]
        W_orig = tensor_image1.shape[-1]
        #Forward pass
        feature_vectors1, keypoints_raw1, keypoints1, reliability_vectors1 = self.net(tensor_image0)
        feature_vectors2, keypoints_raw2, keypoints2, reliability_vectors2 = self.net(tensor_image1)

        feature_vector_list = [feature_vectors1, feature_vectors2]
        keypoints_list = [keypoints1, keypoints2]
        reliability_vector_list = [reliability_vectors1, reliability_vectors2]

        keypoints_imgs = []
        descriptors_imgs = []
        scores_imgs = []

        for feature_vectors, keypoints, reliability_vectors in zip(feature_vector_list, keypoints_list, reliability_vector_list):
            # Zero out the scores a position (0,0) -- invalid keypoints
            reliability_vectors[torch.all(keypoints == 0, dim=-1)] = -1

            #Select top-k features
            idxs = torch.argsort(-reliability_vectors, dim=1)
            # Sort the keypoints based on the sorted scores
            sorted_keypoints = torch.gather(keypoints, 1, idxs.expand(-1, -1, 2))
            sorted_feature_vectors = torch.gather(feature_vectors, 1, idxs.expand(-1, -1, 64))
            mkpts = sorted_keypoints[:, :top_k, :]
            feature_vectors = sorted_feature_vectors[:, :top_k, :]
            scores = torch.gather(reliability_vectors.squeeze(-1), 1, idxs.squeeze(-1))[:, :top_k]
            valid = scores > 0

            #L2-Normalize
            feats = F.normalize(feature_vectors, dim=-1)

            valid = scores > 0

            for b in range(B):
                keypoints_imgs.append(mkpts[b][valid[b]])
                descriptors_imgs.append(feats[b][valid[b]])
                scores_imgs.append(scores[b][valid[b]])
            
        # Match the keypoints betweeen the two images

        feats1 = descriptors_imgs[0]
        feats2 = descriptors_imgs[1]

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
		
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        matched_pts_in_img0, matched_pts_in_img1 = keypoints_imgs[0][idx0].cpu().numpy(), keypoints_imgs[1][idx1].cpu().numpy()

        # Sanity check for the warp function

        # Get scale factors (should be [sx, sy] or scalar)
        if 'scale0' in transform_data and 'scale1' in transform_data:
            scale0 = transform_data['scale0'][0]
            scale1 = transform_data['scale1'][0]
        else:
            scale0 = torch.tensor([1.0, 1.0], device=tensor_image0.device)
            scale1 = torch.tensor([1.0, 1.0], device=tensor_image1.device)

        # print(f"keypoints_imgs[0] shape: {keypoints_imgs[0].shape}")
        # print(f"keypoints_imgs[0][idx0] shape: {keypoints_imgs[0][idx0].shape}")

        # print(f"MEGA_Test_plot - kpts1[{b}].shape: {keypoints_imgs[0].shape}, kpts1[{b}] range: {keypoints_imgs[0].min():.1f}-{keypoints_imgs[0].max():.1f}")


        # Scale to match intrinsics space
        matched_pts_in_img0_warp = keypoints_imgs[0][idx0].cpu() * scale0  # now in same space as K0

        # Warp using scaled point
        valid_mask, wrpd_points = megadepth_warper.warp_kpts(
            matched_pts_in_img0_warp.unsqueeze(0),
            transform_data['depth0'][0].unsqueeze(0),
            transform_data['depth1'][0].unsqueeze(0),
            transform_data['T_0to1'][0].unsqueeze(0),
            transform_data['K0'][0].unsqueeze(0),
            transform_data['K1'][0].unsqueeze(0)
        )

        # Plot the warped points shape and range for debugging
        # print(f"Test plot warped- wrpd_points.shape: {wrpd_points.shape}, wrpd_points range: {wrpd_points.min():.1f}-{wrpd_points.max():.1f}")

        # Scale points back into resized net image coordinates
        wrpd_points = wrpd_points / scale1

        # Print range of warped points after scaling
        # print(f"Test plot warped after scaling- wrpd_points range: {wrpd_points[0].min():.1f}-{wrpd_points[0].max():.1f}")
        # print(f"Matched pts in img1 shape: {matched_pts_in_img1.shape}, range: {matched_pts_in_img1.min():.1f}-{matched_pts_in_img1.max():.1f}")
        error_list = [
            np.linalg.norm(wrpd_points[0][i].cpu().numpy() - matched_pts_in_img1[i]) < 10
            for i in range(len(matched_pts_in_img1))
        ]        # Calculate the mean error between the matched points and the warped points
        mean_error = np.mean([np.linalg.norm(wrpd_points[0][i].cpu().numpy() - matched_pts_in_img1[i]) for i in range(len(matched_pts_in_img1))])
        
        # Export the mean pixel error and the percentage of geometrically correctly matched points to tensorboard
        self.writer.add_scalar("Accuracy/Mean pixel error", mean_error, global_step=current_step)
        self.writer.add_scalar("Accuracy/Percentage of correct matches", np.mean(error_list), global_step=current_step)

        # Create a single image with both images next to each other with the lines between the matched points
        concat_img = np.concat([tensor_image0.numpy(), tensor_image1.numpy()], axis=3).squeeze(0)
        img = np.transpose(concat_img, (1, 2, 0))  # Now (H, W, 3)
        img = img[..., ::-1]

        # Get image widths
        w0 = tensor_image0.shape[-1]  # width of first image

        # Shift second image's x coords
        pts2_shifted = matched_pts_in_img1.copy()
        pts2_shifted[:, 0] += w0  # shift x by width of first image

        # OpenCV expects BGR
        img_np_bgr = img[..., ::-1].copy()

        # Draw matches
        for (x1, y1), (x2, y2), is_valid in zip(matched_pts_in_img0.astype(int), pts2_shifted.astype(int), error_list):
            color = (0, 255, 255) if is_valid else (0, 0, 255)
            cv2.line(img_np_bgr, (x1, y1), (x2, y2), color, 1)
            cv2.circle(img_np_bgr, (x1, y1), 2, color, -1)
            cv2.circle(img_np_bgr, (x2, y2), 2, color, -1)

        # Convert BGR back to RGB for TensorBoard
        img_np_rgb = img_np_bgr[..., ::-1].copy()

        # Convert back to Torch (N, C, H, W)
        img_tb = torch.from_numpy(img_np_rgb).permute(2, 0, 1) # uint8

        self.writer.add_image("Training data", img_tb, global_step=current_step)

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
                        print(f"First point in p1s: {p1s[0, :, 0, 0]}")

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                if self.plot_test:
                    if i % 10 == 0:
                        self.create_test_plot(d, current_step=i)
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
                        print(f"Batch {i} is corrupted with too few correspondences: {len(p)}")

                if is_corrupted:
                    continue

                #Forward pass
                feats1, keypoints_raw1, kpts1, hmap1 = self.net(p1)
                feats2, keypoints_raw2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    num_synthetic = len(positives_s_coarse) if self.augmentor is not None else 0
                    
                    if b < num_synthetic:
                        # SYNTHETIC: Use H2[b] for this specific batch item
                        # print(f"Shape of kpts1[{b}]: {kpts1[b].shape}")
                        # print(f"First point in kpts1[{b}]: {kpts1[b][0]}")
                        # print(f"SYNTH - p1[{b}].shape: {p1[b].shape}, kpts1[{b}].shape: {kpts1[b].shape}, kpts1[{b}] range: {kpts1[b].min():.1f}-{kpts1[b].max():.1f}")


                        pts2_warped = warp_synthetic_points(
                            kpts1[b],
                            p2[b],
                            H1,
                            H2,
                            self.augmentor,
                            b
                        )
                        # Remove out of bounds points
                        pts2_warped = pts2_warped[
                            (pts2_warped[:, 0] >= 0) & (pts2_warped[:, 0] < p2.shape[3]) &
                            (pts2_warped[:, 1] >= 0) & (pts2_warped[:, 1] < p2.shape[2])
                        ]

                        # Number of keypoints found in img2
                        # print(f"kpts2[{b}] range: {kpts2[b].min():.1f}-{kpts2[b].max():.1f}, shape: {kpts2[b].shape}")

                        # # Print the current two images and plot the warped points vs original points for debugging
                        # import matplotlib.pyplot as plt
                        # plt.figure(figsize=(10, 5))
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(p1[b].permute(1, 2, 0).cpu())
                        # plt.scatter(kpts1[b][:, 0].cpu(), kpts1[b][:, 1].cpu(), c='r', s=5)
                        # plt.title("Image 1 with keypoints")
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(p2[b].permute(1, 2, 0).cpu())
                        # plt.scatter(pts2_warped[:, 0].cpu(), pts2_warped[:, 1].cpu(), c='g', s=5)
                        # plt.scatter(kpts2[b][:, 0].cpu(), kpts2[b][:, 1].cpu(), c='r', s=5)
                        # plt.title("Image 2 with warped and original keypoints")
                        # plt.show()

                        # print(f"Shape of warped points: {pts2_warped.shape} and shape of kpts2[{b}]: {kpts2[b].shape}")
                        
                        idx_positive_matches_img1, idx_positive_matches_img2 = get_matching_idx(
                            pts2_warped, kpts2[b], threshold=5.0
                        )
                    else:
                        # MegaDepth...
                        b_adjusted = b - num_synthetic
                        # print(f"MEGA - p1[{b}].shape: {p1[b].shape}, kpts1[{b}].shape: {kpts1[b].shape}, kpts1[{b}] range: {kpts1[b].min():.1f}-{kpts1[b].max():.1f}")
                        # Get scale factors (should be [sx, sy] or scalar)
                        if 'scale0' in d and 'scale1' in d:
                            scale0 = d['scale0'][b_adjusted]
                            scale1 = d['scale1'][b_adjusted]
                            # print(f"Found scale0: {scale0} and scale1: {scale1}")
                        else:
                            scale0 = torch.tensor([1.0, 1.0], device=p1[b].device)
                            scale1 = torch.tensor([1.0, 1.0], device=p2[b].device)

                        # print(f"Before warp keypoints_imgs[0] shape: {kpts1[b].shape}")

                        # Scale to match intrinsics space
                        kpts1_scaled = kpts1[b].cpu() * scale0  # Convert coarse → full-res

                        # Warp using scaled point
                        valid_mask, wrpd_points = megadepth_warper.warp_kpts(
                            kpts1_scaled.unsqueeze(0),
                            d['depth0'][b_adjusted].unsqueeze(0),
                            d['depth1'][b_adjusted].unsqueeze(0),
                            d['T_0to1'][b_adjusted].unsqueeze(0),
                            d['K0'][b_adjusted].unsqueeze(0),
                            d['K1'][b_adjusted].unsqueeze(0)
                        )
                        # print(f"MEGA warped- wrpd_points.shape: {wrpd_points.shape}, wrpd_points range: {wrpd_points.min():.1f}-{wrpd_points.max():.1f}")


                        # Scale points back into resized net image coordinates
                        wrpd_points = wrpd_points / scale1
                        wrpd_points = wrpd_points[valid_mask]
                        # print(f"MEGA warped after scaling- wrpd_points range: {wrpd_points.min():.1f}-{wrpd_points.max():.1f}")
                        # print(f"Range of kpts2[{b}]: {kpts2[b].min():.1f}-{kpts2[b].max():.1f}")
                        # # Show the two shapes with the warped points for debugging
                        # import matplotlib.pyplot as plt
                        # plt.figure(figsize=(10, 5))
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(p1[b].permute(1, 2, 0).cpu())
                        # plt.scatter(kpts1[b][:, 0].cpu(), kpts1[b][:, 1].cpu(), c='r', s=5)
                        # plt.title("Megadepth Image 1 with keypoints")
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(p2[b].permute(1, 2, 0).cpu())
                        # plt.scatter(wrpd_points.squeeze(0)[:, 0].cpu(), wrpd_points.squeeze(0)[:, 1].cpu(), c='g', s=5)
                        # plt.scatter(kpts2[b][:, 0].cpu(), kpts2[b][:, 1].cpu(), c='b', s=5)
                        # plt.title("Megadepth Image 2 with warped and original keypoints")
                        # plt.show()


                        # Filter for points that are within a given threshold of the warped points
                        # to determine positive correspondences
                        idx_positive_matches_img1, idx_positive_matches_img2 = get_matching_idx(
                            wrpd_points.squeeze(0), kpts2[b], threshold=5.0)


                    if len(idx_positive_matches_img1) >= 2 or len(idx_positive_matches_img2) >= 2:
                        # Enough matches to compute losses that require correspondences                       

                        #Matched points in both images
                        pts1 = kpts1[b][idx_positive_matches_img1]
                        pts2 = kpts2[b][idx_positive_matches_img2]

                        #Grab features at corresponding idxs
                        m1 = feats1[b][idx_positive_matches_img1]
                        m2 = feats2[b][idx_positive_matches_img2]
                        # print(f"Shapes in Keypoint loss: m1: {m1.shape}, m2: {m2.shape}")

                        #grab heatmaps at corresponding idxs
                        h1 = hmap1[b][idx_positive_matches_img1].squeeze(-1)
                        h2 = hmap2[b][idx_positive_matches_img2].squeeze(-1)

                        # M1 and M2 are not entire maps anymore, but only the features at found keypoints
                        coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                        #Compute losses
                        loss_ds, conf = circle_loss(m1, m2, margin=0.5, scale=64)
                        loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)
    
                        # print(f"Shapes in Keypoint loss: h1: {h1.shape}, h2: {h2.shape}, conf: {conf.shape}")
                        loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)


                    else:
                        # Not enough matches to compute losses that require correspondences
                        m1, m2, h1, h2 = None, None, None, None
                        loss_ds = torch.tensor(0.0, device=self.dev)
                        loss_coords = torch.tensor(0.0, device=self.dev)
                        acc_coords = 0.0
                        conf = None
                        loss_kp = torch.tensor(0.0, device=self.dev)
    
                    loss_kp_pos1, acc_pos1 = alike_distill_loss(keypoints_raw1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(keypoints_raw2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2



                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0 and m1 is not None and m2 is not None:
                        acc_coarse_0 = check_accuracy(m1, m2)
                    else:
                        acc_coarse_0 = 0.0

                if m1 is not None and m2 is not None:
                    acc_coarse = check_accuracy(m1, m2)
                else:
                    acc_coarse = 0.0

                nb_coarse = len(m1) if m1 is not None else 0
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
        save_ckpt_every=args.save_ckpt_every
    )

    #The most fun part
    trainer.train()
