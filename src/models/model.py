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
        self.pt_conv1 = PointTransformerConv(in_channels=3+attr_dim, out_channels=64)
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
        # Concatenate position and attributes
        x_combined = torch.cat([pos, attr], dim=1)  # shape: [N, 3 + attr_dim]
        
        # Use PointTransformerConv with the combined features
        # Note: we still use pos for positional encoding within the transformer
        x_feat = self.pt_conv1(x_combined, pos, edge_index)
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
        return x_feat
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class Feature_Expansion(nn.Module):
    def __init__(self, feat_dim=64, up_ratio=2, pos_mlp_hidden=32, 
                 up_attn_hds=2, up_concat=False, up_beta=False, up_dropout=0, 
                 fnl_attn_hds=2):
        """
        Relative positional encoding with a self-attention refinement.
    
        This module:
          (a) Uses a TransformerConv (instead of GCNConv) to expand features.
          (b) Performs periodic shuffle to upsample features.
          (c) Computes relative positional features based on original coordinates.
          (d) Concatenates the relative positional features to the upsampled features.
          (e) Refines the concatenated features using a self-attention module.
    
        Args:
            feat_dim (int): Input feature dimension.
            up_ratio (int): Upsampling ratio (r). For example, 2 doubles the number of points.
            pos_mlp_hidden (int): Hidden size for the MLP that processes relative positions.
            up_attn_hds (int): Number of attention heads for the TransformerConv layer.
            up_concat (bool): If True, concatenates attention head outputs in TransformerConv. If False, averages them.
            up_beta (bool): If True, uses an additional learnable weighting factor in TransformerConv.
            up_dropout (float): Dropout rate applied to TransformerConv outputs.
            fnl_attn_hds (int): Number of attention heads for the final self-attention refinement layer.
        
        Returns:
            x_out (torch.Tensor): [r*N, total_dim] refined features after relative positional encoding and self-attention.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.up_ratio = up_ratio

        # TransformerConv: if up_concat=True, output shape will be [N, up_attn_hds * (up_ratio*feat_dim)]
        self.transformer_conv = TransformerConv(in_channels=feat_dim,
                                                out_channels=up_ratio * feat_dim,
                                                heads=up_attn_hds,
                                                concat=up_concat,
                                                beta=up_beta,
                                                dropout=up_dropout)
        
        # 1) Insert LayerNorm here. If up_concat=True with multiple heads,
        #    your output dimension is up_attn_hds * (up_ratio * feat_dim).
        #    Otherwise, it's just (up_ratio * feat_dim).
        hidden_dim = up_attn_hds * (up_ratio * feat_dim) if up_concat and up_attn_hds > 1 \
                     else (up_ratio * feat_dim)
        self.norm_after_transformer = nn.LayerNorm(hidden_dim)

        
        # If we use concat with multiple attention heads, add a learned projection to map concatenated outputs back down.
        if up_concat and up_attn_hds > 1:
            # Input dimension: up_attn_hds * (up_ratio*feat_dim); output: up_ratio*feat_dim.
            self.projection = nn.Sequential(
                nn.Linear(up_attn_hds * up_ratio * feat_dim, up_ratio * feat_dim),
                nn.ReLU(),
                nn.Linear(up_ratio * feat_dim, up_ratio * feat_dim)
            )
        else:
            self.projection = nn.Identity()
        
        # An MLP to process relative positions (3D -> pos_mlp_hidden)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden, pos_mlp_hidden)
        )

        # When using PyTorch’s MultiheadAttention, the embed_dim (here, the sum of the feature dimensions) must be exactly divisible by the number of attention heads.
        # After concatenation, the feature dimension becomes: feat_dim + pos_mlp_hidden.
        self.total_dim = feat_dim + pos_mlp_hidden

        # Compute remainder and adjust total_dim if necessary.
        remainder = self.total_dim % fnl_attn_hds

        if remainder != 0:
            # Increase dimension to the next multiple.
            adjusted_dim = self.total_dim + (fnl_attn_hds - remainder)
            self.dim_adjust = nn.Linear(self.total_dim, adjusted_dim)
            self.adjusted_dim = adjusted_dim
        else:
            self.dim_adjust = nn.Identity()
            self.adjusted_dim = self.total_dim
        
        self.self_attn = nn.MultiheadAttention(embed_dim=self.adjusted_dim,
                                                num_heads=fnl_attn_hds,
                                                batch_first=True)


    
    def forward(self, x_feat, edge_index, original_pos):
        """
        Args:
            x_feat: [N, feat_dim] node features.
            edge_index: [2, E] graph connectivity.
            original_pos: [N, 3] original coordinates corresponding to x_feat.
        Returns:
            x_out: [r*N, total_dim] refined features after relative positional encoding and self-attention.
        """
        N, C = x_feat.shape
        r = self.up_ratio

        # 1) Feature Expansion using TransformerConv (replacing GCNConv)
        x_expanded = self.transformer_conv(x_feat, edge_index)  # shape: [N, ?]
        # Apply LayerNorm to stabilize activations
        x_expanded = self.norm_after_transformer(x_expanded)
        x_expanded = F.relu(x_expanded)
        
        # check_tensor(x_expanded, "TransformerConv output (x_expanded)")
        
        # 2) If using concatenated heads, apply the learned projection.
        if self.transformer_conv.concat:
            # x_expanded has shape [N, up_attn_hds * r * feat_dim].
            # Apply learned projection: maps [N, up_attn_hds * r * feat_dim] -> [N, r * feat_dim]
            x_projected = self.projection(x_expanded)  # [N, r * feat_dim]
            # Reshape to [N, r, feat_dim]
            x_projected = x_projected.view(N, r, C)
        else:
            # When concat=False, output is already [N, r * feat_dim]
            x_projected = x_expanded.view(N, r, C)
        # check_tensor(x_projected, "Projected features (x_projected)")
        
        # 3) Periodic shuffle: reshape from [N, r, feat_dim] to [r*N, feat_dim]
        x_shuffled = x_projected.permute(1, 0, 2).reshape(r * N, C)
        # check_tensor(x_shuffled, "Shuffled features (x_shuffled)")

        # 4) Compute relative positions: subtract the global centroid from original positions.
        global_centroid = original_pos.mean(dim=0, keepdim=True)  # [1, 3]
        rel_pos = original_pos - global_centroid  # [N, 3]
        # Tile these relative positions r times: [r*N, 3]
        rel_pos_tiled = rel_pos.unsqueeze(0).expand(r, N, 3).reshape(r * N, 3)
        # check_tensor(rel_pos, "Relative positions (rel_pos)")
        # check_tensor(rel_pos_tiled, "Tiled relative positions (rel_pos_tiled)")


        # 5) Process relative positions via MLP to obtain positional features.
        rel_pos_features = self.pos_mlp(rel_pos_tiled)  # [r*N, pos_mlp_hidden]
        # check_tensor(rel_pos_features, "Processed relative positional features (rel_pos_features)")

        # 6) Concatenate the processed relative positional features with the upsampled features.
        x_concat = torch.cat([x_shuffled, rel_pos_features], dim=1)  # [r*N, feat_dim + pos_mlp_hidden]
        # check_tensor(x_concat, "Concatenated features (x_concat)")
        
        
        # 7) Apply a simple self-attention mechanism.
        # apply the adjustment before feeding into into self-attention
        x_adj = self.dim_adjust(x_concat)  # Now has shape [r*N, adjusted_dim]
        x_attn_in = x_adj.unsqueeze(0)  # [1, r*N, adjusted_dim]
        # MultiheadAttention expects input shape [batch, seq_len, embed_dim]. We use batch_first=True.
        x_attn_out, _ = self.self_attn(x_attn_in, x_attn_in, x_attn_in)
        x_attn_out = x_attn_out.squeeze(0)
        # check_tensor(x_attn_out, "Self-attention output (x_attn_out)")

        
        return x_attn_out


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



