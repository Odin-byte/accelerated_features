"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import torch
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import pdb
import cv2
from modules.training.utils import map_cyclic_polar_feature_coords_to_img_coords, map_img_coords_to_cyclic_polar_feature_coords
from modules.training.utils import plot_corrs

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long().clip(0, 2000-1)

    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Draw cross marks on the image for each keypoint
    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth0.numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in kpts0_long[b, nonzero_mask[b], :].numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    # h, w = depth1.shape[1:3]
    # covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0

    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0


@torch.no_grad()
def spvs_coarse(data, scale_h = 10.666, scale_w = 4.0, scale_lin_conv = 8):
    """
        Supervise corresp with dense depth & camera poses
    """

    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    #scale = 8
    # scale0 and scale1 are the scales for h and w for each image in the batch
    # these image specific scales need to be multiplied by the scale of the network
    # to get the final scale for the image before the cyclic convolution
    scale0 = torch.tensor([scale_h, scale_w], device=device) * data['scale0'][:, None] if 'scale0' in data else torch.tensor([scale_h, scale_w], device=device)[None, None, :]
    scale1 = torch.tensor([scale_h, scale_w], device=device) * data['scale1'][:, None] if 'scale1' in data else torch.tensor([scale_h, scale_w], device=device)[None, None, :]
    
    h0, h1 = H0 / scale_h, H1 / scale_h
    w0, w1 = W0 / scale_w, W1 / scale_w

    h1 = int(h1)
    w1 = int(w1)

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    # [N, hw, 2]
    # Move to grid points from feature space to image space
    # Scale the grid points by the linear scale of the standard convolution layers
    grid_pt1_i = 8 * grid_pt1_c
    # Apply the non linear mapping
    grid_pt1_i = map_cyclic_polar_feature_coords_to_img_coords(grid_pt1_i, H0, W0, kernel_size=3, stride=1, padding=1)
    # warp kpts bi-directionally and check reproj error
    nonzero_m1, w_pt1_i  =  warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0']) 
    nonzero_m2, w_pt1_og =  warp_kpts(   w_pt1_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1']) 


    dist = torch.linalg.norm( grid_pt1_i - w_pt1_og, dim=-1)
    mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

    #_, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    batched_corrs = [ torch.cat([w_pt1_i[i, mask_mutual[i]] / data['scale0'][i],
                       grid_pt1_i[i, mask_mutual[i]] / data['scale1'][i]],dim=-1) for i in range(len(mask_mutual))]


    #Remove repeated correspondences - this is important for network convergence
    corrs = []
    for pts in batched_corrs:
        lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1
        lut_mat21 = torch.clone(lut_mat12)
        # src_pts_cyclic = map_img_coords_to_cyclic_polar_feature_coords(
        #             src_pts, rh, rw, kernel_size=3, stride=1, padding=1
        #         )
        # Move the points from image space to feature space
        src_pts = map_img_coords_to_cyclic_polar_feature_coords(
            pts[:, :2], H0, W0, kernel_size=3, stride=1, padding=1
        )
        tgt_pts = map_img_coords_to_cyclic_polar_feature_coords(
            pts[:, 2:], H1, W1, kernel_size=3, stride=1, padding=1
        )
        src_pts = pts[:, :2] / scale_lin_conv
        tgt_pts = pts[:, 2:] / scale_lin_conv

        try:
            # Compute integer indices
            y_idx = src_pts[:, 1].long()
            x_idx = src_pts[:, 0].long()
            # Create a mask for valid indices
            valid_mask = (
                (y_idx >= 0) & (y_idx < lut_mat12.shape[0]) &
                (x_idx >= 0) & (x_idx < lut_mat12.shape[1])
            )
            # Filter only valid points
            y_idx = y_idx[valid_mask]
            x_idx = x_idx[valid_mask]
            src_pts_valid = src_pts[valid_mask]
            tgt_pts_valid = tgt_pts[valid_mask]
            # Assign only valid points to the lookup table
            lut_mat12[y_idx, x_idx] = torch.cat([src_pts_valid, tgt_pts_valid], dim=1)
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            # Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            y_idx = tgt_pts[:, 1].long()
            x_idx = tgt_pts[:, 0].long()
            valid_mask = (
                (y_idx >= 0) & (y_idx < lut_mat21.shape[0]) &
                (x_idx >= 0) & (x_idx < lut_mat21.shape[1])
            )
            y_idx = y_idx[valid_mask]
            x_idx = x_idx[valid_mask]
            src_pts_valid = src_pts[valid_mask]
            tgt_pts_valid = tgt_pts[valid_mask]
            lut_mat21[y_idx, x_idx] = torch.cat([src_pts_valid, tgt_pts_valid], dim=1)
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

            corrs.append(points)
            print(f"Number of correspondences found: {len(points)}")
        except:
            pdb.set_trace()
            print('..')

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

    return corrs

@torch.no_grad()
def get_correspondences(pts2, data, idx):
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape

    pts2 = pts2[None, ...]

    scale0 = data['scale0'][idx, None][None, ...] if 'scale0' in data else 1
    scale1 = data['scale1'][idx, None][None, ...] if 'scale1' in data else 1

    pts2 = scale1 * pts2 * 8

    # warp kpts bi-directionally and check reproj error
    nonzero_m1, pts1  = warp_kpts(pts2, data['depth1'][idx][None, ...], data['depth0'][idx][None, ...], data['T_1to0'][idx][None, ...], 
                                                                        data['K1'][idx][None, ...], data['K0'][idx][None, ...]) 

    corrs = torch.cat([pts1[0, :] / data['scale0'][idx],
                       pts2[0, :] / data['scale1'][idx]],dim=-1)

    #plot_corrs(data['image0'][idx], data['image1'][idx], corrs[:, :2], corrs[:, 2:])

    return corrs

