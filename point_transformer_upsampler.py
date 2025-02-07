# point_cloud_upsampler.py

import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph, PointTransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data  # <-- Add this line
import matplotlib.pyplot as plt


def precompute_knn_inplace(filtered_data, k=10):
    """
    Loops through filtered_data and precomputes the KNN graph for each
    sample's 'dep_points'. Stores the result in sample['dep_edge_index'].

    filtered_data: list of dicts, each like:
      {
          'dep_points': shape [3, N_dep],
          'uav_points': shape [3, N_uav]
      }
    k: Number of neighbors for KNN graph.
    """
    for sample in filtered_data:
        dep_points = sample['dep_points'].contiguous()   # [N_dep, 3]

        # Build the KNN graph once, offline
        edge_index = knn_graph(dep_points, k=k, loop=False)
        # Make sure edges are undirected
        edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))

        # Store for later use
        sample['dep_edge_index'] = edge_index


def normalize_pair(dep_points, uav_points):
    """
    Given two point clouds dep_points & uav_points, each [N, 3],
    we compute a common center & scale so that both clouds are in
    the same normalized coordinate system.

    Returns:
      dep_points_norm: [N_dep, 3] normalized
      uav_points_norm: [N_uav, 3] normalized
      center:          [1, 3]     the mean
      scale:           scalar     bounding radius
    """
    # 1) Combine them to find a shared centroid and scale
    combined = torch.cat([dep_points, uav_points], dim=0)  # [N_dep + N_uav, 3]

    center = combined.mean(dim=0, keepdim=True)  # shape [1, 3]
    combined_centered = combined - center        # shape [N_dep+N_uav, 3]

    # 2) Scale by max distance from origin (L2 norm)
    #    e.g. so the entire combined cloud fits inside [-1, 1].
    scale = combined_centered.norm(dim=1).max()  # single scalar
    scale = scale.clamp_min(1e-9)                # avoid div by zero

    # 3) Apply to both
    dep_points_norm = (dep_points - center) / scale
    uav_points_norm = (uav_points - center) / scale

    return dep_points_norm, uav_points_norm, center, scale


class PointCloudUpsampleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        dep_points = sample['dep_points']       # [N_dep, 3]
        uav_points = sample['uav_points']       # [N_uav, 3]
        edge_index = sample['dep_edge_index']   # [2, E]
        bbox = sample['bbox']
        
        dtm_1m = sample['dtm_1m'].squeeze(0)
        dsm_1m = sample['dsm_1m'].squeeze(0)

        dtm_50cm = sample['dtm_50cm'].squeeze(0)
        dsm_50cm = sample['dsm_50cm'].squeeze(0)
        
        wts = sample['ch_50cm'].squeeze(0)   # shape [N_uav]

        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points, uav_points)
        
        # Wrap the scalar values in lists so they are iterable.
        return (dep_points_norm, uav_points_norm, edge_index, center, scale, 
                wts, dtm_1m, dsm_1m,dtm_50cm,dsm_50cm,bbox)



def variable_size_collate(batch):
    dep_list, uav_list, edge_list = [], [], []
    center_list, scale_list = [], []
    wts_list = [] 
    dtm_1m_list = [] 
    dsm_1m_list = [] 
    dtm_50cm_list = [] 
    dsm_50cm_list = [] 
    bbox_list = []
    
    for (dep_pts, uav_pts, e_idx, center, scale, wts, dtm_1m, dsm_1m,dtm_50cm,dsm_50cm, bbox) in batch:
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        center_list.append(center)
        scale_list.append(scale)
        wts_list.append(wts)
        dtm_1m_list.append(dtm_1m)
        dsm_1m_list.append(dsm_1m)
        dtm_50cm_list.append(dtm_50cm)
        dsm_50cm_list.append(dsm_50cm)
        bbox_list.append(bbox)
    return dep_list, uav_list, edge_list, center_list, scale_list, wts_list, dtm_1m_list, dsm_1m_list,dtm_50cm_list,dsm_50cm_list,bbox_list



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
        pos: Tensor of shape [N, 3] containing point coordinates.
        edge_index: Tensor of shape [2, E] containing the edge indices.
        Returns:
          x_feat: Tensor of shape [N, feature_dim] representing per-point features.
        """
        # Use the positions as the initial feature vector.
        x_feat = self.pt_conv1(pos, pos, edge_index)
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
        return x_feat


class FeatureExpansion(nn.Module):
    def __init__(self, num_out_points=2048):
        super().__init__()
        self.num_out_points = num_out_points

    def forward(self, x_feat):
        """
        x_feat: [N_dep, feature_dim]
        Return repeated or truncated to [N_out, feature_dim].
        """
        N_in, feat_dim = x_feat.shape
        if self.num_out_points <= N_in:
            # Just slice if we already have enough points
            x_expanded = x_feat[:self.num_out_points]
        else:
            repeats = (self.num_out_points // N_in) + 1
            x_tiled = x_feat.repeat((repeats, 1))
            x_expanded = x_tiled[:self.num_out_points]
        return x_expanded


class PointSetGenerator(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x_feat):
        return self.mlp(x_feat)

class TransformerPointUpsampler(nn.Module):
    def __init__(self, feature_dim=64, num_out_points=2048):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.feature_expander = FeatureExpansion(num_out_points=num_out_points)
        self.point_generator = PointSetGenerator(feature_dim=feature_dim)

    def forward(self, dep_points, edge_index):
        """
        dep_points: [N_dep, 3]
        edge_index: [2, E]   (precomputed adjacency)
        returns: upsampled [N_out, 3]
        """
        # 1) Extract features using the PointTransformerConv-based feature extractor
        x_feat = self.feature_extractor(dep_points, edge_index)
        # 2) Expand features
        x_expanded = self.feature_expander(x_feat)
        # 3) Generate final points
        pred_points = self.point_generator(x_expanded)
        return pred_points


def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer distance between two point clouds.
    pc1: [N1, 3]
    pc2: [N2, 3]
    """
    dist = torch.cdist(pc1, pc2)  # [N1, N2]
    min_dist_pc1, _ = dist.min(dim=1)  # [N1]
    min_dist_pc2, _ = dist.min(dim=0)  # [N2]
    return min_dist_pc1.mean() + min_dist_pc2.mean()



def run_inference_and_visualize_2plots(
    trained_model, 
    filtered_data, 
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
        """Extracts, normalizes, runs inference, and computes distance metrics."""
        sample = filtered_data[index]
        dep_points_raw = sample['dep_points']  # [N_dep, 3]
        uav_points_raw = sample['uav_points']  # [N_uav, 3]
        edge_index = sample['dep_edge_index']

        # 1) Normalize
        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points_raw, uav_points_raw)

        # 2) Move to device
        dep_points_norm = dep_points_norm.to(device)
        uav_points_norm = uav_points_norm.to(device)
        edge_index      = edge_index.to(device)

        # 3) Build a PyG Data object for inference.
        data = Data(pos=dep_points_norm, edge_index=edge_index)


        # 4) Run inference in normalized space.
        trained_model.eval()
        with torch.no_grad():
            # Now pass the single Data object.
            pred_points_norm = trained_model(data.pos, data.edge_index)


        # 5) Compute distances (using your existing chamfer_distance and hausdorff_distance functions).
        orig_chamfer_dist = chamfer_distance(dep_points_norm, uav_points_norm)
        upsmpl_chamfer_dist = chamfer_distance(pred_points_norm, uav_points_norm)

        orig_hausdorff_dist = hausdorff_distance(dep_points_norm, uav_points_norm)
        upsmpl_hausdorff_dist = hausdorff_distance(pred_points_norm, uav_points_norm)
        # 5) Return CPU data for plotting
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
    
    # # Remove batch dimension from predicted points if it exists:
    # if pred1.ndim == 3 and pred1.shape[0] == 1:
    #     pred1 = pred1.squeeze(0)
    # if pred2.ndim == 3 and pred2.shape[0] == 1:
    #     pred2 = pred2.squeeze(0)
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
    
def hausdorff_distance(pc1, pc2):
    """Computes the bidirectional Hausdorff distance between two point clouds."""
    dist = torch.cdist(pc1, pc2)  # [N1, N2]
    min_dist_pc1, _ = dist.min(dim=1)  # [N1]
    min_dist_pc2, _ = dist.min(dim=0)  # [N2]
    return max(min_dist_pc1.max().item(), min_dist_pc2.max().item())


def weighted_chamfer_distance(
    pc1, pc2, weights, 
    clamp_min=None, clamp_max=None,
    scale_min=None, scale_max=None
):
    """
    Computes a weighted Chamfer Distance between two point clouds,
    using the same weights for both pc1 and pc2, with optional clamping
    and scaling of 'weights'.
    
    Args:
        pc1: Tensor of shape [N1, 3]
        pc2: Tensor of shape [N2, 3]
        weights: Tensor of shape [N2], the per-point weights for pc2.
        
        clamp_min (float or None): If not None, clamp weights to >= clamp_min
        clamp_max (float or None): If not None, clamp weights to <= clamp_max
        scale_min (float or None): If not None, final weight range lower bound
        scale_max (float or None): If not None, final weight range upper bound
        
    Returns:
        A scalar tensor representing the weighted Chamfer distance.
    """
    # 0) Optionally clamp
    if clamp_min is not None or clamp_max is not None:
        weights = torch.clamp(weights, min=clamp_min, max=clamp_max)
    
    # 1) Optionally rescale to a new range [scale_min, scale_max]
    if (scale_min is not None) and (scale_max is not None):
        # Current min/max of the weights
        w_min = weights.min()
        w_max = weights.max()
        
        # Avoid division by zero if all weights are the same
        denom = (w_max - w_min).clamp_min(1e-9)
        
        # Map [w_min..w_max] -> [0..1]
        w_norm = (weights - w_min) / denom
        
        # Then map [0..1] -> [scale_min..scale_max]
        weights = w_norm * (scale_max - scale_min) + scale_min
    
    # 2) Compute pairwise distances between points in pc1 and pc2.
    dist = torch.cdist(pc1, pc2)  # shape: [N1, N2]
    
    # 3) For each point in pc1, find the minimum distance to pc2 (+ indices).
    min_dist_pc1, nn_indices_pc1 = dist.min(dim=1)  # shape: [N1]
    
    # 4) For each point in pc2, find the minimum distance to pc1.
    min_dist_pc2, _ = dist.min(dim=0)  # shape: [N2]
    
    # 5) Use the weight of the corresponding nearest neighbor in pc2 for each point in pc1.
    weights_pc1 = weights[nn_indices_pc1]  # shape: [N1]
    
    # 6) Weighted loss for pc1
    #    sum(...) / weights_pc1.sum() => average over pc1 points, weighted by pc2's weights
    weighted_loss_pc1 = (min_dist_pc1 * weights_pc1).sum() / weights_pc1.sum()
    
    # 7) Weighted loss for pc2
    weighted_loss_pc2 = (min_dist_pc2 * weights).sum() / weights.sum()
    
    return weighted_loss_pc1 + weighted_loss_pc2


if __name__ == "__main__":
    # Example usage when running this script directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePointUpsampler(feature_dim=96, num_out_points=4000).to(device)
    print("SimplePointUpsampler model instantiated.")
