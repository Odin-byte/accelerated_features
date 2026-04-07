"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cyclic_polar_patches import PolarPatchGenerator
import time

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)

class XFeatPolarPatchDescriptorGen(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self, patch_size=32):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1)
		self.patch_size = patch_size


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )
		
		# Main backbone

		# CyclicPolarCoordinateConvLayer

		self.polar_patch_generator = PolarPatchGenerator(
			in_channels=1,
			out_channels=1,
			patch_size=self.patch_size,
			kernel_size=3,
		)

		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),
										BasicLayer( 4,  8, stride=2),
										BasicLayer( 8,  8, stride=1),
										BasicLayer( 8, 24, stride=2),
									)

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1),
										BasicLayer(24, 24, stride=1),
									 )

		self.block3 = nn.Sequential(
										BasicLayer(24, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, 1, padding=0),
									 )
		self.block4 = nn.Sequential(
										BasicLayer(64, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 )

		self.block5 = nn.Sequential(
										BasicLayer( 64, 128, stride=2),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128,  64, 1, padding=0),
									 )

		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0)
									 )


		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)


		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########

		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)

	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x, points):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
				points -> torch.Tensor(B, N, 2) keypoint locations
			return:
				feats     ->  torch.Tensor(B, N, 64) dense local features

		"""
		# If points is a list convert it to tensor
		if isinstance(points, list):
			points = torch.stack(points, dim=0)


		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		#main backbone
		polar_patches = self.polar_patch_generator(x, points)
		batches_list = []

		for batch_patch in polar_patches:
			# print(patch.shape)
			x1 = self.block1(batch_patch)
			x2 = self.block2(x1 + self.skip1(batch_patch))
			x3 = self.block3(x2)
			x4 = self.block4(x3)
			x5 = self.block5(x4)
			#pyramid fusion
			x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
			x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')

			fused = self.block_fusion(x3 + x4 + x5)
			descriptor = fused.mean(dim=(-2, -1))  # (N_patches, 64)
			batches_list.append(descriptor)
		
		# Append the descriptors for all patches into a single tensor
		feats = torch.stack(batches_list, dim=0)  # (B, N_patches, 64)

		# print(deep_feats.shape)

		# heatmap_input = feats
		# heatmap = self.heatmap_head(heatmap_input)  # Reliability map
		# keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

		return feats #, keypoints, heatmap
	

if __name__ == "__main__":
		# Test the model
		model = XFeatPolarPatchDescriptorGen()
		x = torch.randn(10, 1, 800, 608)  # Example input tensor
		test_points = torch.randn(10, 100, 2)  # Example keypoint locations (B, N, 2)
		feats = model(x, test_points)

		print("Input shape:", x.shape)
		print("Output shapes:")
		# print("keypoints shape:", keypoints.shape)
		print("feats shape:", feats.shape)
		# print("heatmap shape:", heatmap.shape)

		# Calculate ratio between input and feature map size for x an y
		# ratio_x = x.shape[2] / feats.shape[2]
		# ratio_y = x.shape[3] / feats.shape[3]
		# print("Ratio x:", ratio_x)
		# print("Ratio y:", ratio_y)
