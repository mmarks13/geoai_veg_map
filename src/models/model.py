# point_transformer_upsampler.py

import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph, PointTransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys
from src.utils.chamfer_distance import chamfer_distance

# from preprocess import check_tensor

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


from torch_geometric.nn import TransformerConv  # Using TransformerConv instead of GCNConv
from torch_geometric.nn import PointTransformerConv

###############################################
# 1. FeatureExtractor
###############################################

class FeatureExtractor(nn.Module):
    """
    Takes positions, attributes, and a precomputed edge_index and returns features.
    Uses PointTransformerConv layers to capture local geometric detail.
    """
    def __init__(self, feature_dim=64, attr_dim=3):
        super().__init__()
        # First PointTransformerConv layer: input dimension is 3+attr_dim (position + attributes)
        self.pt_conv1 = PointTransformerConv(in_channels=attr_dim, out_channels=64)
        # Second layer
        self.pt_conv2 = PointTransformerConv(in_channels=64, out_channels=feature_dim)

    def forward(self, pos, attr, edge_index):
        """
        Args:
            pos: Tensor of shape [N, 3] containing point coordinates.
            attr: Tensor of shape [N, attr_dim] containing point attributes.
            edge_index: Tensor of shape [2, E] containing the edge indices.
        Returns:
            x_feat: Tensor of shape [N, feature_dim] representing per-point features.
        """
        
        # Use PointTransformerConv with the combined features
        # Note: we still use pos for positional encoding within the transformer
        x_feat = self.pt_conv1(attr, pos, edge_index)
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
        return x_feat
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch.nn.functional import scaled_dot_product_attention  # PyTorch 2.0+ feature


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointTransformerConv
from torch.nn.functional import scaled_dot_product_attention


class FeatureExtractor(nn.Module):
    """
    Takes positions, attributes, and a precomputed edge_index and returns features.
    Uses PointTransformerConv layers to capture local geometric detail and Flash Attention.
    """
    def __init__(self, feature_dim=64, attr_dim=3):
        super().__init__()
        # First PointTransformerConv layer with half the output dimension
        self.pt_conv1 = PointTransformerConv(in_channels=attr_dim, out_channels=feature_dim//2)
        # Second layer with original feature dimension
        self.pt_conv2 = PointTransformerConv(in_channels=feature_dim//2, out_channels=feature_dim)
        
        # Flash Attention implementation
        self.num_heads = 4  # Can be adjusted
        self.head_dim = feature_dim // self.num_heads
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, pos, attr, edge_index):
        """
        Args:
            pos: Tensor of shape [N, 3] containing point coordinates.
            attr: Tensor of shape [N, attr_dim] containing point attributes.
            edge_index: Tensor of shape [2, E] containing the edge indices.
        Returns:
            x_feat: Tensor of shape [N, feature_dim] representing per-point features.
        """
        # Use PointTransformerConv layers
        x_feat = self.pt_conv1(attr, pos, edge_index)
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
        
        # Apply Flash Attention
        batch_size = 1  # Assuming single batch
        seq_len = x_feat.size(0)  # Number of points
        
        # Add batch dimension
        x_batched = x_feat.unsqueeze(0)  # [1, N, feature_dim]
        
        # Project to queries, keys, values
        q = self.q_proj(x_batched).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_batched).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_batched).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply scaled dot product attention (Flash Attention)
        attn_output = scaled_dot_product_attention(q, k, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, x_feat.size(1))
        x_feat = self.out_proj(attn_output).squeeze(0)  # [N, feature_dim]
        
        return x_feat


class Feature_Expansion(nn.Module):
    def __init__(self, feat_dim=64, up_ratio=2, fnl_attn_hds=2):
        """
        Feature expansion using PointTransformerConv and Flash Attention.
        
        This module:
          (a) Uses a PointTransformerConv to expand features.
          (b) Performs periodic shuffle to upsample features.
          (c) Refines the features using a flash attention-based self-attention module.
        
        Args:
            feat_dim (int): Input feature dimension.
            up_ratio (int): Upsampling ratio (r). For example, 2 doubles the number of points.
            fnl_attn_hds (int): Number of attention heads for the final self-attention refinement layer.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.up_ratio = up_ratio

        # PointTransformerConv for feature expansion
        self.point_transformer = PointTransformerConv(
            in_channels=feat_dim,
            out_channels=up_ratio * feat_dim
        )
        
        # Ensure feat_dim is divisible by number of heads
        if feat_dim % fnl_attn_hds != 0:
            adjusted_dim = feat_dim + (fnl_attn_hds - (feat_dim % fnl_attn_hds))
            self.dim_adjust = nn.Linear(feat_dim, adjusted_dim)
            self.adjusted_dim = adjusted_dim
        else:
            self.dim_adjust = nn.Identity()
            self.adjusted_dim = feat_dim
        
        # Flash Attention implementation
        self.num_heads = fnl_attn_hds
        self.head_dim = self.adjusted_dim // fnl_attn_hds
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.k_proj = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.v_proj = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.out_proj = nn.Linear(self.adjusted_dim, feat_dim)  # Project back to original dimension
    
    def forward(self, x_feat, edge_index, original_pos):
        """
        Args:
            x_feat: [N, feat_dim] node features.
            edge_index: [2, E] graph connectivity.
            original_pos: [N, 3] original coordinates.
        Returns:
            x_out: [r*N, feat_dim] expanded and refined features.
        """
        N, C = x_feat.shape
        r = self.up_ratio
        
        # 1) Feature expansion with PointTransformerConv
        x_expanded = self.point_transformer(x_feat, original_pos, edge_index)  # [N, r*feat_dim]
        x_expanded = F.relu(x_expanded)
        
        # 2) Reshape to [N, r, feat_dim]
        x_reshaped = x_expanded.view(N, r, C)
        
        # 3) Periodic shuffle: reshape to [r*N, feat_dim]
        x_shuffled = x_reshaped.permute(1, 0, 2).reshape(r * N, C)
        
        # 4) Adjust dimensions if needed
        x_adj = self.dim_adjust(x_shuffled)  # [r*N, adjusted_dim]
        
        # 5) Apply Flash Attention
        x_batched = x_adj.unsqueeze(0)  # [1, r*N, adjusted_dim]
        
        # Project to queries, keys, values
        q = self.q_proj(x_batched).view(1, r*N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_batched).view(1, r*N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_batched).view(1, r*N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply scaled dot product attention (Flash Attention)
        attn_output = scaled_dot_product_attention(q, k, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, r*N, self.adjusted_dim)
        x_final = self.out_proj(attn_output).squeeze(0)  # [r*N, feat_dim]
        
        return x_final


###############################################
# 3. PointUpsampler
###############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class PointUpsampler(nn.Module):
    def __init__(self, 
                 feat_dim=64, 
                 up_ratio=2, 
                 pos_mlp_hidden=32, 
                 up_attn_hds=2, 
                 up_concat=False, 
                 up_beta=False, 
                 up_dropout=0, 
                 fnl_attn_hds=2,
                 attr_dim=3):  # Add attr_dim parameter
        """
        Point cloud upsampler using:
          (a) FeatureExtractor -> normalized output
          (b) Feature_Expansion -> normalized output
          (c) MLP decoder -> final 3D coordinates.
        """
        super().__init__()

        # ------------------------
        # 1) Feature Extractor
        # ------------------------
        self.feature_extractor = FeatureExtractor(feature_dim=feat_dim, attr_dim=attr_dim)
        
        # Add LayerNorm right after the feature extractor output
        self.norm_after_extractor = nn.LayerNorm(feat_dim)

        # ------------------------
        # 2) Node Shuffle
        # ------------------------
        self.node_shuffle = Feature_Expansion(
            feat_dim=feat_dim,
            up_ratio=up_ratio,
            pos_mlp_hidden=pos_mlp_hidden,
            up_attn_hds=up_attn_hds,
            up_concat=up_concat,
            up_beta=up_beta,
            up_dropout=up_dropout,
            fnl_attn_hds=fnl_attn_hds
        )
        
        # Add LayerNorm right after the NodeShuffle output
        # The shape after NodeShuffle is [r*N_dep, self.node_shuffle.adjusted_dim]
        self.norm_after_nodeshuffle = nn.LayerNorm(self.node_shuffle.adjusted_dim)
        
        # ------------------------
        # 3) Decoder
        # ------------------------
        total_dim = self.node_shuffle.adjusted_dim  # (feat_dim + pos_mlp_hidden), possibly adjusted
        self.point_decoder = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, dep_points, edge_index, dep_attr=None):
        # Default to zero attributes if None is provided
        if dep_attr is None:
            dep_attr = torch.zeros(dep_points.size(0), 3, device=dep_points.device)
            
        # ====== 1) Feature Extraction ======
        x_feat = self.feature_extractor(dep_points, dep_attr, edge_index)  # [N_dep, feat_dim]
        # check_tensor(x_feat, "FeatureExtractor output (x_feat)")
        
        # # Normalize + optional activation
        # x_feat = self.norm_after_extractor(x_feat)  # [N_dep, feat_dim]
        # x_feat = F.relu(x_feat)
        # check_tensor(x_feat, "FeatureExtractor output AFTER LayerNorm + ReLU (x_feat)")

        # ====== 2) NodeShuffle Expansion ======
        x_up = self.node_shuffle(x_feat, edge_index, dep_points)  # [r*N_dep, adjusted_dim]
        # check_tensor(x_up, "NodeShuffle output (x_up)")
        
        # # Normalize + optional activation
        # x_up = self.norm_after_nodeshuffle(x_up)  # [r*N_dep, adjusted_dim]
        # x_up = F.relu(x_up)
        # check_tensor(x_up, "NodeShuffle output AFTER LayerNorm + ReLU (x_up)")
        
        # ====== 3) Decode to 3D Coordinates ======
        pred_points = self.point_decoder(x_up)  # [r*N_dep, 3]
        # check_tensor(pred_points, "PointDecoder output (pred_points)")

        return pred_points



