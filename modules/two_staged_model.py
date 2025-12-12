"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cyclic_polar_patches import PolarPatchGenerator
import time

def get_kpts_heatmap(kpts, softmax_temp = 1.0):
	scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
	B, _, H, W = scores.shape
	heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
	heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
	return heatmap

def NMS(x, threshold = 0.05, kernel_size = 5):
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

class XFeatPolarPatchesModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1)


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )
		
		# Backbone for Descriptor and Reliability Vectors
		# This backbone is used in the second stage for the generated keypoint positions / patches
		# The input for this backbone are polar patches extracted around keypoint positions

		self.polar_patch_generator = PolarPatchGenerator(
												in_channels = 1,
												out_channels = 1,
												patch_size = 65, # Patch size of X by X pixels
												kernel_size=3,
												stride=1
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


		self.reliability_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid(),
										nn.AdaptiveAvgPool2d((1,1)),
									)

		# Untouched Keypoint Head (works with the unfolded input (cartesian))
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


	def forward(self, x, wanted_kpts=1000):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		# Generate keypoints
		keypoints_raw = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

		keypoint_pos = get_kpts_heatmap(keypoints_raw)  # Keypoint heatmap
		keypoint_pos = NMS(keypoint_pos, threshold=0.05, kernel_size=5)  # Keypoint positions

		#TODO: Allow a maximum of 1000 keypoints per image, sample them over the entire image
		num_keyponints = keypoint_pos.shape[1]
		sample_step = num_keyponints // wanted_kpts
		if sample_step == 0:
			sample_step = 1
		keypoint_pos = keypoint_pos[:, ::sample_step, :]  # Sample to wanted_kpts
		# If the keypoints are less than wanted_kpts, pad with zeros
		if keypoint_pos.shape[1] < wanted_kpts:
			pad_size = wanted_kpts - keypoint_pos.shape[1]
			padding = torch.zeros((keypoint_pos.shape[0], pad_size, 2), dtype=torch.long, device=keypoint_pos.device)
			keypoint_pos = torch.cat([keypoint_pos, padding], dim=1)
		if keypoint_pos.shape[1] > wanted_kpts:
			keypoint_pos = keypoint_pos[:, :wanted_kpts, :]
		# print(f"keypoint_pos.shape: {keypoint_pos.shape}")
		# print(f"Range of keypoint positions: x [{keypoint_pos[...,0].min().item()}, {keypoint_pos[...,0].max().item()}], y [{keypoint_pos[...,1].min().item()}, {keypoint_pos[...,1].max().item()}]")

		# keypoint_pos is of shape (B, N, 2) with (x, y) positions, as every batch is padded to the same size N
		# some positions might be (0,0) if less than N keypoints were found

		# Generate feature and reliability vectors
		polar_patches = self.polar_patch_generator.forward(x, keypoint_pos)
		# polar_patches is of shape (B, N, C, patch_size, patch_size)
		# where C is the number of channels of the input x (after normalization)
		# B is the batch size
		# N is the number of keypoints (padded to the same size for all batch items)

		# Pass polar patches through the backbone but save the number of batches and keypoints
		B, N, C, PS, PS = polar_patches.shape

		# print("Number of keypoints:", N)
		# print(f"Reshaped polar patches shape: {polar_patches.reshape(-1, 1, PS, PS).shape}")

		x1 = self.block1(polar_patches.reshape(-1, 1, PS, PS)) #Reshape to (B*N, C, H, W)
		# print(f"Shape after block1: {x1.shape}")
		skip_out = self.skip1(polar_patches.reshape(-1, 1, PS, PS))
		# Interpolate skip connection output to match x1 size
		skip_out = F.interpolate(skip_out, (x1.shape[-2], x1.shape[-1]), mode='bilinear', align_corners=True)
		x2 = self.block2(x1 + skip_out)
		x3 = self.block3(x2)
		x4 = self.block4(x3)
		x5 = self.block5(x4)

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		feats = self.block_fusion( x3 + x4 + x5 )
		feats = F.adaptive_avg_pool2d(feats, (1,1))  # Global average pooling

		#Reshape back to (B, N, C) with C = 64
		feats_final = feats.reshape(B, N, -1)

		#heads
		reliability_vectors = self.reliability_head(feats) # Reliability vectors
		reliability_vectors_final = reliability_vectors.reshape(B, N, -1)
		# Reshape feats back to (B, 64, H/8, W/8)

		return feats_final, keypoints_raw, keypoint_pos, reliability_vectors_final
	

if __name__ == "__main__":
		# Test the model
		model = XFeatPolarPatchesModel()
		x = torch.randn(10, 1, 800, 608)  # Example input tensor
		B = x.shape[0]
		feature_vectors, keypoints, reliability_vectors = model(x)

		print("Feature vectors shape:", feature_vectors.shape)  # Expected: (B, N, 64)
		print("Keypoints shape:", keypoints.shape)              # Expected: (B, N, 2)
		print("Reliability vectors shape:", reliability_vectors.shape)  # Expected: (B, N, 1)

		### Check sorting and selection of top-k keypoints based on reliability scores
		### as its done in the inference function
		top_k = feature_vectors.shape[1]  # Number of keypoints per batch item

		# Zero out the scores a position (0,0) -- invalid keypoints
		reliability_vectors[torch.all(keypoints == 0, dim=-1)] = -1

		# Print out the highest reliability scores for the first batch item with its idx
		print("Top reliability scores for first batch item:")
		print(torch.topk(reliability_vectors[0].squeeze(-1), k=10))

		#Select top-k features
		idxs = torch.argsort(-reliability_vectors, dim=1)
		# Sort the keypoints based on the sorted scores without splitting them into x and y
		sorted_keypoints = torch.gather(keypoints, 1, idxs.expand(-1, -1, 2))
		sorted_feature_vectors = torch.gather(feature_vectors, 1, idxs.expand(-1, -1, 64))
		mkpts = sorted_keypoints[:, :top_k, :]
		feature_vectors = sorted_feature_vectors[:, :top_k, :]
		print(mkpts.shape)
		scores = torch.gather(reliability_vectors.squeeze(-1), 1, idxs.squeeze(-1))[:, :top_k]
		valid = scores > 0



		# Sanity check: verify sorting is correct
		print("\n=== Sanity Check ===")
		print(f"Top 5 scores after sorting: {scores[0, :5]}")
		print(f"Should be descending and > 0 for valid keypoints")

		# Check that scores match the sorted reliability vectors
		original_scores_at_sorted_indices = reliability_vectors[0].squeeze(-1)[idxs[0].squeeze(-1)[:5]]
		print(f"Original reliability at sorted indices: {original_scores_at_sorted_indices}")
		print(f"Match scores[0, :5]? {torch.allclose(scores[0, :5], original_scores_at_sorted_indices)}")

		# Check that keypoints are properly reordered
		print(f"\nFirst 3 sorted keypoint positions: {mkpts[0, :3]}")
		print(f"Corresponding original positions: {keypoints[0][idxs[0, :3].squeeze(-1)]}")
		print(f"Match? {torch.allclose(mkpts[0, :3], keypoints[0][idxs[0, :3].squeeze(-1)])}")

		# Check that invalid keypoints (0,0) have score -1 and appear at the end
		invalid_count = (scores[0] == -1).sum().item()
		print(f"\nNumber of invalid keypoints in top_k: {invalid_count}")
		if invalid_count > 0:
			print(f"Last {min(3, invalid_count)} scores (should be -1): {scores[0, -3:]}")


		#L2-Normalize
		print(f"Feature vectors shape before normalization: {feature_vectors.shape}")
		feats = F.normalize(feature_vectors, dim=-1)
		print(f"Feature vectors shape after normalization: {feats.shape}")
		print(f"Shape of valid mask: {valid.shape}")
		print(f"Shape of descriptors: {feats.shape}")
		output =  [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]
		
		# Verify valid mask filters correctly or all batches
		for b in range(B):
			print(f"\nBatch {b}:")
			print(f"Number of valid keypoints: {output[b]['keypoints'].shape[0]}")
			if output[b]['keypoints'].shape[0] > 0:
				print(f"First 3 valid keypoints: {output[b]['keypoints'][:3]}")
				print(f"First 3 valid scores: {output[b]['scores'][:3]}")
				print(f"First 3 valid descriptors: {output[b]['descriptors'][:3]}")
			else:
				print("No valid keypoints found.")