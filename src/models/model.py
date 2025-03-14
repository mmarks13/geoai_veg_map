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
# 1. FeatureExtractor (as before)
###############################################

class FeatureExtractor(nn.Module):
    """
    Takes positions and a precomputed edge_index and returns features.
    Uses PointTransformerConv layers to capture local geometric detail.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        # First PointTransformerConv layer: input dimension is 3 (point coordinates)
        self.pt_conv1 = PointTransformerConv(in_channels=3, out_channels=64)
        # Second layer: input dimension 64, output dimension is feature_dim.
        self.pt_conv2 = PointTransformerConv(in_channels=64, out_channels=feature_dim)

    def forward(self, pos, edge_index):
        """
        Args:
            pos: Tensor of shape [N, 3] containing point coordinates.
            edge_index: Tensor of shape [2, E] containing the edge indices.
        Returns:
            x_feat: Tensor of shape [N, feature_dim] representing per-point features.
        """
        # Use the positions as the initial feature vector.
        x_feat = self.pt_conv1(pos, pos, edge_index)
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
        return x_feat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class NodeShuffleFeatureExpansion_Relative_Attn(nn.Module):
    def __init__(self, feat_dim=64, up_ratio=2, pos_mlp_hidden=32, 
                 up_attn_hds=2, up_concat=False, up_beta=False, up_dropout=0, 
                 fnl_attn_hds=2):
        """
        Relative positional encoding with a self-attention refinement.
    
        This module:
          (a) Uses a TransformerCon                                                      v (instead of GCNConv) to expand features.
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
# 3. NodeShufflePointUpsampler_Relative_Attn
###############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class NodeShufflePointUpsampler_Relative_Attn(nn.Module):
    def __init__(self, 
                 feat_dim=64, 
                 up_ratio=2, 
                 pos_mlp_hidden=32, 
                 up_attn_hds=2, 
                 up_concat=False, 
                 up_beta=False, 
                 up_dropout=0, 
                 fnl_attn_hds=2):
        """
        Point cloud upsampler using:
          (a) FeatureExtractor -> normalized output
          (b) NodeShuffleFeatureExpansion_Relative_Attn -> normalized output
          (c) MLP decoder -> final 3D coordinates.
        """
        super().__init__()

        # ------------------------
        # 1) Feature Extractor
        # ------------------------
        self.feature_extractor = FeatureExtractor(feature_dim=feat_dim)
        
        # Add LayerNorm right after the feature extractor output
        self.norm_after_extractor = nn.LayerNorm(feat_dim)

        # ------------------------
        # 2) Node Shuffle
        # ------------------------
        self.node_shuffle = NodeShuffleFeatureExpansion_Relative_Attn(
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

    def forward(self, dep_points, edge_index):
        # ====== 1) Feature Extraction ======
        x_feat = self.feature_extractor(dep_points, edge_index)  # [N_dep, feat_dim]
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





# def chamfer_distance(pc1, pc2):
#     """
#     Computes the Chamfer distance between two point clouds.
#     pc1: [N1, 3]
#     pc2: [N2, 3]
#     """
#     dist = torch.cdist(pc1, pc2)  # [N1, N2]
#     min_dist_pc1, _ = dist.min(dim=1)  # [N1]
#     min_dist_pc2, _ = dist.min(dim=0)  # [N2]
#     return min_dist_pc1.mean() + min_dist_pc2.mean()

# import torch




# from pytorch3d.loss import chamfer_distance as cdist

# def chamfer_distance_pytorch3d(pc1, pc2):
#     """
#     Computes the Chamfer distance using PyTorch3D, converting inputs to float32 
#     temporarily for loss computation.
#     """
#     # Save the current device.
#     device = pc1.device
    
#     # Convert to float32 without detaching (so gradients are preserved).
#     pc1_f32 = pc1.to(torch.float32)
#     pc2_f32 = pc2.to(torch.float32)
    
#     # Check for NaNs or INF in the inputs to chamfer_distance
#     # check_tensor(pc1_f32, "Chamfer input pc1_f32")
#     # check_tensor(pc2_f32, "Chamfer input pc2_f32")
#     # Add batch dimension.
#     pc1_f32 = pc1_f32.unsqueeze(0)
#     pc2_f32 = pc2_f32.unsqueeze(0)
    
#     loss, _ = cdist(pc1_f32, pc2_f32)
#     return loss.to(device)



import torch
import torch.nn.functional as F

def sliced_wasserstein_distance(pc1, pc2, num_projections=50, p=2):
    """
    Approximates the Wasserstein distance using Sliced Wasserstein Distance (SWD).
    
    Args:
        pc1 (torch.Tensor): [N, d] point cloud.
        pc2 (torch.Tensor): [M, d] point cloud.
        num_projections (int): Number of random directions (projections) to use.
        p (int): The exponent in the L^p norm (typically 1 or 2).
        
    Returns:
        torch.Tensor: A scalar tensor representing the SWD.
    """
    # Ensure point clouds are of shape [N, d] and [M, d]
    d = pc1.shape[1]
    device = pc1.device

    # Generate random projection directions on the unit sphere.
    projections = torch.randn(num_projections, d, device=device)
    projections = F.normalize(projections, p=2, dim=1)  # shape: [num_projections, d]

    # Project both point clouds onto these directions.
    proj_pc1 = pc1 @ projections.t()  # shape: [N, num_projections]
    proj_pc2 = pc2 @ projections.t()  # shape: [M, num_projections]

    # Sort the projections along the point dimension.
    proj_pc1_sorted, _ = torch.sort(proj_pc1, dim=0)  # shape: [N, num_projections]
    proj_pc2_sorted, _ = torch.sort(proj_pc2, dim=0)  # shape: [M, num_projections]

    # To handle different numbers of points, interpolate both sorted projections to a common grid.
    # Here, we choose a fixed number of bins (e.g., max(N, M)).
    num_bins = max(pc1.shape[0], pc2.shape[0])
    
    # A helper: interpolate a sorted projection to num_bins
    def interpolate_sorted(sorted_proj, original_length, num_bins):
        # sorted_proj: [original_length, num_projections]
        # Unsqueeze and permute to shape [1, num_projections, original_length]
        sorted_proj = sorted_proj.unsqueeze(0).permute(0, 2, 1)
        sorted_proj_interp = F.interpolate(sorted_proj, size=num_bins, mode='linear', align_corners=True)
        # Permute back to shape [num_bins, num_projections]
        return sorted_proj_interp.squeeze(0).permute(1, 0)
    
    proj_pc1_interp = interpolate_sorted(proj_pc1_sorted, pc1.shape[0], num_bins)  # [num_bins, num_projections]
    proj_pc2_interp = interpolate_sorted(proj_pc2_sorted, pc2.shape[0], num_bins)  # [num_bins, num_projections]

    # Compute the p-th power of the difference, average over bins and then over projections.
    swd = torch.abs(proj_pc1_interp - proj_pc2_interp).pow(p).mean(dim=0).pow(1.0/p)
    return swd.mean()

# Example usage:
# pc1 and pc2 are torch tensors on the GPU.
# distance = sliced_wasserstein_distance(pc1, pc2, num_projections=50, p=2)


# Update the run_inference_and_visualize_2plots function to use the new data structure
def run_inference_and_visualize_2plots(
    trained_model, 
    model_data, 
    index1=0, 
    index2=1, 
    device='cuda', 
    width=14, 
    height=3,
    hide_labels=False
):
    """
    Runs inference on two different samples (index1 & index2),
    visualizes both in a single row of six plots with an empty separator,
    and displays Chamfer Distance (CD) and Hausdorff Distance (HD).
    """

    def process_sample(index):
        """Extracts data, runs inference, and computes distance metrics."""
        sample = model_data[index]
        
        # Extract normalized points and edge index from the precomputed data
        dep_points_norm = sample['dep_points_norm']  # [N_dep, 3]
        uav_points_norm = sample['uav_points_norm']  # [N_uav, 3]
        
        # Use the precomputed edge index from dep_edge_index (set by precompute_knn_inplace)
        # or fall back to a specific k value from knn_edge_indices
        if 'dep_edge_index' in sample:
            edge_index = sample['dep_edge_index']
        elif 'knn_edge_indices' in sample and 30 in sample['knn_edge_indices']:  # Default to k=30
            edge_index = sample['knn_edge_indices'][30]
        else:
            # Last resort fallback
            print(f"Warning: No precomputed edges found for sample {index}, computing KNN")
            edge_index = knn_graph(dep_points_norm, k=30, loop=False)
            edge_index = to_undirected(edge_index, num_nodes=dep_points_norm.size(0))

        # Move to device
        dep_points_norm = dep_points_norm.to(device)
        uav_points_norm = uav_points_norm.to(device)
        edge_index = edge_index.to(device)

        # Run inference with normalized points
        trained_model.eval()
        with torch.no_grad():
            pred_points_norm = trained_model(dep_points_norm, edge_index)

        # Compute distances
        orig_chamfer_dist = chamfer_distance(dep_points_norm, uav_points_norm)
        upsmpl_chamfer_dist = chamfer_distance(pred_points_norm, uav_points_norm)

        orig_hausdorff_dist = hausdorff_distance(dep_points_norm, uav_points_norm)
        upsmpl_hausdorff_dist = hausdorff_distance(pred_points_norm, uav_points_norm)
        
        # Return CPU data for plotting
        return (
            dep_points_norm.cpu(),
            uav_points_norm.cpu(),
            pred_points_norm.cpu(),
            orig_chamfer_dist,
            upsmpl_chamfer_dist,
            orig_hausdorff_dist,
            upsmpl_hausdorff_dist
        )

    # Process both samples
    dep1, uav1, pred1, chamfer1_orig, chamfer1_upsmpl, hausdorff1_orig, hausdorff1_upsmpl = process_sample(index1)
    dep2, uav2, pred2, chamfer2_orig, chamfer2_upsmpl, hausdorff2_orig, hausdorff2_upsmpl = process_sample(index2)
    
    # The rest of the function remains unchanged...
    # Create a single row of subplots with an empty separator
    fig, axes = plt.subplots(1, 7, figsize=(width, height), subplot_kw={'projection': '3d'})

    def configure_axes(ax, title):
        """Helper function to configure axis formatting."""
        if hide_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title('')  # Remove title
        else:
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6, pad=0)
            ax.ticklabel_format(style='plain', axis='both')
            ax.xaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_visible(False)
            ax.zaxis.get_offset_text().set_visible(False)

    def add_distance_labels(ax, chamfer_dist, hausdorff_dist):
        """Add Chamfer Distance (CD) and Hausdorff Distance (HD) labels to the plot."""
        ax.text2D(0, 0.1, f"CD: {chamfer_dist:.4f}", transform=ax.transAxes, fontsize=7, color='black')
        ax.text2D(0, 0, f"HD: {hausdorff_dist:.4f}", transform=ax.transAxes, fontsize=7, color='black')

    # First three plots => Sample index1
    axes[0].scatter(dep1[:, 0], dep1[:, 1], dep1[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[0], f"3DEP ({index1})")
    add_distance_labels(axes[0], chamfer1_orig, hausdorff1_orig)

    axes[1].scatter(uav1[:, 0], uav1[:, 1], uav1[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[1], f"UAV ({index1})")

    axes[2].scatter(pred1[:, 0], pred1[:, 1], pred1[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[2], f"Upsampled ({index1})")
    add_distance_labels(axes[2], chamfer1_upsmpl, hausdorff1_upsmpl)

    # Empty plot for separation
    axes[3].axis("off")

    # Next three plots => Sample index2
    axes[4].scatter(dep2[:, 0], dep2[:, 1], dep2[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[4], f"3DEP ({index2})")
    add_distance_labels(axes[4], chamfer2_orig, hausdorff2_orig)

    axes[5].scatter(uav2[:, 0], uav2[:, 1], uav2[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[5], f"UAV ({index2})")

    axes[6].scatter(pred2[:, 0], pred2[:, 1], pred2[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[6], f"Upsampled ({index2})")
    add_distance_labels(axes[6], chamfer2_upsmpl, hausdorff2_upsmpl)

    # Adjust layout to remove excess whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)

    plt.show()
    


###############################################
# 4. Example Usage
###############################################

if __name__ == '__main__':
    # Dummy input data for testing:
    N_dep = 1024       # Number of input points
    feat_dim = 64
    up_ratio = 2
    pos_mlp_hidden = 32
    attn_heads = 4

    # Create dummy point cloud and edge_index (for example, from a kNN graph)
    dep_points = torch.rand(N_dep, 3).cuda()  # random points on GPU

    # For demonstration, create a dummy edge_index connecting sequential points.
    # In practice, use a proper kNN or radius graph.
    edge_index = torch.tensor([list(range(N_dep-1)), list(range(1, N_dep))], dtype=torch.long).cuda()

    # Instantiate the upsampler model.
    model = NodeShufflePointUpsampler_Relative_Attn(
        feat_dim=feat_dim,
        up_ratio=up_ratio,
        pos_mlp_hidden=pos_mlp_hidden,
        attn_heads=attn_heads
    ).cuda()

    # Forward pass.
    pred_points = model(dep_points, edge_index)
    print(f"Output shape: {pred_points.shape}")  # Expected shape: [up_ratio * N_dep, 3]
