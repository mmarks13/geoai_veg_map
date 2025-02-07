import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph, DynamicEdgeConv, EdgeConv
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_max
import matplotlib.pyplot as plt
from torch_geometric.data import Data  # <-- Add this line


###############################################################################
# 1. Utility Functions
###############################################################################

def precompute_knn_inplace(filtered_data, k=10):
    """
    Precomputes a k-NN graph for each sample’s 'dep_points' and stores the edge index
    in sample['dep_edge_index'].
    
    Each sample is a dict with:
      'dep_points': [3, N_dep]
      'uav_points': [3, N_uav]
    """
    for sample in filtered_data:
        dep_points = sample['dep_points']  # [N_dep, 3]
        # Build the k-NN graph based on Euclidean distance (no self-loops)
        edge_index = knn_graph(dep_points, k=k, loop=False)
        # Convert to undirected graph
        edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))
        sample['dep_edge_index'] = edge_index

def normalize_pair(dep_points, uav_points):
    """
    Normalizes two point clouds (each of shape [N, 3]) to a common coordinate system.
    
    Returns:
      dep_points_norm: [N_dep, 3] normalized
      uav_points_norm: [N_uav, 3] normalized
      center:          [1, 3] the mean of combined points
      scale:           scalar bounding radius
    """
    combined = torch.cat([dep_points, uav_points], dim=0)
    center = combined.mean(dim=0, keepdim=True)
    combined_centered = combined - center
    scale = combined_centered.norm(dim=1).max().clamp_min(1e-9)
    dep_points_norm = (dep_points - center) / scale
    uav_points_norm = (uav_points - center) / scale
    return dep_points_norm, uav_points_norm, center, scale

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer distance between two point clouds:
      pc1: [N1, 3], pc2: [N2, 3]
    """
    dist = torch.cdist(pc1, pc2)
    min_dist_pc1, _ = dist.min(dim=1)
    min_dist_pc2, _ = dist.min(dim=0)
    return min_dist_pc1.mean() + min_dist_pc2.mean()

###############################################################################
# 2. Dataset and Collation
###############################################################################

class PointCloudUpsampleDataset(Dataset):
    """
    Each sample is a dict with keys:
      'dep_points': [3, N_dep]
      'uav_points': [3, N_uav]
      'dep_edge_index': [2, E] (precomputed)
    """
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        dep_points = sample['dep_points'].contiguous()  # [N_dep, 3]
        uav_points = sample['uav_points'].contiguous()  # [N_uav, 3]
        edge_index = sample['dep_edge_index']               # [2, E]
        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points, uav_points)
        return dep_points_norm, uav_points_norm, edge_index, center, scale

def variable_size_collate(batch):
    dep_list, uav_list, edge_list, center_list, scale_list = [], [], [], [], []
    for (dep_pts, uav_pts, e_idx, center, scale) in batch:
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        center_list.append(center)
        scale_list.append(scale)
    return dep_list, uav_list, edge_list, center_list, scale_list

###############################################################################
# 3. Network Architecture
###############################################################################
# 3.1 Feature Extraction using DynamicEdgeConv (Simple, Out-of-the-Box)
#
# We define a feature extractor using two DynamicEdgeConv layers.
# Each DynamicEdgeConv layer receives an MLP (defined via nn.Sequential) that takes
# concatenated features of a node and the difference (neighbor - node) as input.
###############################################################################

class FeatureExtractorModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=64, k_first=10, k_second=10):
        """
        Feature extractor using two DynamicEdgeConv layers.
        
        - First layer uses k_first nearest neighbors computed on the input (positions).
        - Second layer uses k_second nearest neighbors computed on the learned feature space.
        """
        super().__init__()
        # First DynamicEdgeConv: input is the point positions.
        self.edgeconv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            k=k_first,
            aggr='max'
        )
        # Second DynamicEdgeConv: input is the features from the first layer.
        self.edgeconv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            ),
            k=k_second,
            aggr='max'
        )
    def forward(self, x, batch=None):
        # x: [N, in_channels] (typically, the normalized positions)
        x1 = self.edgeconv1(x, batch)  # [N, hidden_channels]
        x2 = self.edgeconv2(x1, batch) # [N, out_channels]
        return x2



class HybridFeatureExtractor(nn.Module):
    """
    A hybrid feature extractor that applies:
    1) EdgeConv with precomputed adjacency (static k-NN)
    2) DynamicEdgeConv with dynamic neighbors in feature space
    """
    def __init__(self, in_channels=3, hidden_channels=64,
                 out_channels=64, k_dynamic=10, aggr='max'):
        """
        Args:
            in_channels   (int):   Input feature dimension (e.g., 3 for x,y,z).
            hidden_channels(int):  Dimension of intermediate features.
            out_channels  (int):   Final output feature dimension.
            k_dynamic     (int):   Number of neighbors for DynamicEdgeConv pass.
            aggr          (str):   Aggregation scheme ('max', 'mean', 'add').
        """
        super().__init__()

        # First layer: EdgeConv requires a user-provided edge_index
        # We'll define a small MLP for it:
        self.edgeconv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ),
            aggr=aggr  # e.g., 'max'
        )

        # Second layer: DynamicEdgeConv recomputes adjacency in feature space
        self.edgeconv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            ),
            k=k_dynamic,
            aggr=aggr  # e.g., 'max'
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x          (Tensor): [N, in_channels] point-wise features (positions).
            edge_index (Tensor): [2, E] adjacency for the first layer (static).
            batch      (Tensor, optional): batch assignment if you have multiple
                                           examples in a single pass.
        Returns:
            Tensor of shape [N, out_channels]: extracted features.
        """

        # 1) First pass: Static EdgeConv using precomputed edge_index
        x1 = self.edgeconv1(x, edge_index)  # [N, hidden_channels]

        # 2) Second pass: DynamicEdgeConv (k-dynamic) in feature space
        #    Here we ignore the precomputed adjacency. The layer re-finds neighbors.
        x2 = self.edgeconv2(x1, batch)      # [N, out_channels]

        return x2


# 3.2 Context-Aware Feature Expansion
#
# This module expands the features channel-wise, then reshuffles them (periodic shuffle)
# to upsample the number of points. It also adds a simple 2D grid encoding and refines
# the features with a self-attention unit.
###############################################################################

class ContextAwareExpansion(nn.Module):
    def __init__(self, in_channels, expansion_factor=4, out_channels=None):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.linear_expand = nn.Linear(in_channels, expansion_factor * in_channels)
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        # Simple linear projection to encode 2D grid coordinates.
        self.grid_proj = nn.Linear(2, in_channels)
        self.out_proj = nn.Linear(in_channels, out_channels)
    def periodic_shuffle(self, x):
        # x: [B, N, r * C] --> reshape to [B, r * N, C]
        B, N, RC = x.size()
        r = self.expansion_factor
        C = RC // r
        x = x.view(B, N, r, C)           # [B, N, r, C]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, r, N, C]
        x = x.view(B, r * N, C)          # [B, r*N, C]
        return x
    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.size()
        x_exp = self.linear_expand(x)    # [B, N, r * C]
        x_shuffled = self.periodic_shuffle(x_exp)  # [B, r*N, C]
        # Generate a simple 2D grid for positional encoding.
        rN = x_shuffled.size(1)
        grid_size = int(rN ** 0.5)
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, grid_size),
                                          torch.linspace(0, 1, grid_size),
                                          indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [grid_size^2, 2]
        if grid.size(0) < rN:
            pad = grid[:(rN - grid.size(0))]
            grid = torch.cat([grid, pad], dim=0)
        grid = grid.unsqueeze(0).to(x.device)  # [1, rN, 2]
        grid_encoded = self.grid_proj(grid)      # [1, rN, C]
        grid_encoded = grid_encoded.expand(B, -1, -1)  # [B, rN, C]
        x_with_grid = x_shuffled + grid_encoded
        # Apply self-attention:
        attn_output, _ = self.attention(x_with_grid, x_with_grid, x_with_grid)
        out = self.out_proj(attn_output)
        return out  # [B, r*N, out_channels]


class SimpleExpansion(nn.Module):
    """
    A simplified version of ContextAwareExpansion that:
      1) Expands features via a single linear layer.
      2) Performs a periodic shuffle to increase the number of points.
      3) Applies a final linear layer to get the desired out_channels.
      4) Omits attention and 2D grid encoding for simplicity.
    """
    def __init__(self, in_channels, expansion_factor=4, out_channels=None):
        super().__init__()
        self.expansion_factor = expansion_factor

        # Expand channels from in_channels to expansion_factor * in_channels
        self.linear_expand = nn.Linear(in_channels, expansion_factor * in_channels)

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        # Final projection to out_channels
        self.out_proj = nn.Linear(in_channels, out_channels)

    def periodic_shuffle(self, x):
        """
        x: [B, N, r*C] => reshaped to => [B, r*N, C]

        B: Batch size
        N: Number of input points
        r: expansion_factor
        C: channel dimension after the final shape
        """
        B, N, RC = x.size()
        r = self.expansion_factor
        C = RC // r

        # Reshape to [B, N, r, C], then permute to [B, r, N, C]
        x = x.view(B, N, r, C)
        x = x.permute(0, 2, 1, 3).contiguous()
        # Flatten the first two dims => [B, r*N, C]
        x = x.view(B, r * N, C)
        return x

    def forward(self, x):
        """
        x: [B, N, C]
        Returns: [B, r*N, out_channels]
        """
        B, N, C = x.size()

        # 1) Expand channel dimension => [B, N, r*C]
        x_exp = self.linear_expand(x)

        # 2) Periodic shuffle => [B, r*N, C]
        x_shuffled = self.periodic_shuffle(x_exp)

        # 3) Map shuffled features to desired out_channels => [B, r*N, out_channels]
        out = self.out_proj(x_shuffled)

        return out

# 3.3 Regression: MLP to map features to 3D coordinates.
###############################################################################

class RegressionModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3)
        )
    def forward(self, x):
        return self.mlp(x)  # [num_out_points, 3]

# 3.4 Final Network: Combining all components.
###############################################################################

class PointCloudUpsamplerNet(nn.Module):
    def __init__(self, feature_dim=64, expansion_factor=4, num_out_points=2048, k_dynamic=10):
        super().__init__()
        
        # Use the hybrid feature extractor:
        self.feature_extractor = HybridFeatureExtractor(
            in_channels=3,
            hidden_channels=64,
            out_channels=feature_dim,
            k_dynamic=k_dynamic,
            aggr='max'
        )
        # self.expansion = ContextAwareExpansion(
        #     in_channels=feature_dim,
        #     expansion_factor=expansion_factor,
        #     out_channels=feature_dim
        # )
        self.expansion = SimpleExpansion(
            in_channels=feature_dim,
            expansion_factor=expansion_factor,
            out_channels=feature_dim
        )
        self.regressor = RegressionModule(in_channels=feature_dim)
        self.num_out_points = num_out_points

    def forward(self, dep_points, edge_index, batch=None):
        """
        dep_points: [N, 3] (x,y,z or normalized coords)
        edge_index: [2, E] precomputed adjacency
        batch:      optional, indicates separate subclouds
        """
        # (1) Hybrid Feature Extraction
        features = self.feature_extractor(dep_points, edge_index, batch)  # [N, feature_dim]

        # (2) Expand/Shuffle
        features = features.unsqueeze(0)  # [1, N, feature_dim]
        expanded_features = self.expansion(features)  # [1, r*N, feature_dim]
        B, L, C = expanded_features.size()
        if L > self.num_out_points:
            expanded_features = expanded_features[:, :self.num_out_points, :]
        elif L < self.num_out_points:
            pad = expanded_features[:, -1:, :].repeat(1, self.num_out_points - L, 1)
            expanded_features = torch.cat([expanded_features, pad], dim=1)
        expanded_features = expanded_features.squeeze(0)  # [num_out_points, feature_dim]

        # (3) Regress final coords
        pred_points = self.regressor(expanded_features)  # [num_out_points, 3]
        return pred_points


from torch_geometric.data import Data  # <-- Add this line


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

def compute_distances(pc1, pc2):
    """
    Computes pairwise distances between two point clouds and extracts
    the minimum distances for each point.

    Args:
        pc1: Tensor of shape [N1, 3]
        pc2: Tensor of shape [N2, 3]

    Returns:
        min_dist_pc1: Tensor of shape [N1], minimum distances from pc1 to pc2
        min_dist_pc2: Tensor of shape [N2], minimum distances from pc2 to pc1
    """
    dist = torch.cdist(pc1, pc2)  # shape: [N1, N2]
    min_dist_pc1, _ = dist.min(dim=1)  # [N1]
    min_dist_pc2, _ = dist.min(dim=0)  # [N2]
    return min_dist_pc1, min_dist_pc2

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer Distance between two point clouds.

    Args:
        pc1: Tensor of shape [N1, 3]
        pc2: Tensor of shape [N2, 3]

    Returns:
        A scalar tensor representing the Chamfer distance.
    """
    min_dist_pc1, min_dist_pc2 = compute_distances(pc1, pc2)
    return min_dist_pc1.mean() + min_dist_pc2.mean()

def hausdorff_distance(pc1, pc2):
    """
    Computes a differentiable approximation of the bidirectional Hausdorff distance
    between two point clouds.

    Args:
        pc1: Tensor of shape [N1, 3]
        pc2: Tensor of shape [N2, 3]

    Returns:
        A scalar tensor representing the Hausdorff distance.
    """
    min_dist_pc1, min_dist_pc2 = compute_distances(pc1, pc2)
    return torch.max(min_dist_pc1.max(), min_dist_pc2.max())

def softmax_max(x, alpha=10):
    """Approximates max(x) using softmax weighting."""
    return torch.sum(x * torch.softmax(alpha * x, dim=0))

def hausdorff_distance_soft(pc1, pc2, alpha=10):
    min_dist_pc1, min_dist_pc2 = compute_distances(pc1, pc2)
    return torch.max(softmax_max(min_dist_pc1, alpha), softmax_max(min_dist_pc2, alpha))

def combined_loss(pc1, pc2):
    min_dist_pc1, min_dist_pc2 = compute_distances(pc1, pc2)
    chamfer = min_dist_pc1.mean() + min_dist_pc2.mean()
    hausdorff = hausdorff_distance_soft(pc1, pc2, alpha=10)

    return 0.9 * chamfer + 0.1 * hausdorff

###############################################################################
# 4. Example Usage (Script Mode)
###############################################################################

if __name__ == "__main__":
    # Create synthetic data for demonstration.
    filtered_data = []
    for _ in range(5):
        N_dep = random.randint(50, 150)
        N_uav = random.randint(200, 500)
        dep_pts = torch.rand(3, N_dep) * 10  # [3, N_dep]
        uav_pts = torch.rand(3, N_uav) * 10  # [3, N_uav]
        sample = {'dep_points': dep_pts, 'uav_points': uav_pts}
        filtered_data.append(sample)
    # Precompute k-NN graphs based on positions for each sample.
    precompute_knn_inplace(filtered_data, k=10)
    
    # Create dataset and dataloader.
    dataset = PointCloudUpsampleDataset(filtered_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=variable_size_collate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the network.
    model = PointCloudUpsamplerNet(feature_dim=64, expansion_factor=4, num_out_points=256,
                                  k_first=10, k_second=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 2
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for dep_list, uav_list, edge_list, center_list, scale_list in dataloader:
            optimizer.zero_grad()
            loss_batch = 0.0
            for dep_points, uav_points, e_idx, center, scale in zip(dep_list, uav_list, edge_list, center_list, scale_list):
                dep_points = dep_points.to(device)
                uav_points = uav_points.to(device)
                # For DynamicEdgeConv-based feature extraction, we do not need to pass edge_index here.
                pred_points = model(dep_points, e_idx)  # [num_out_points, 3]
                loss = chamfer_distance(pred_points, uav_points)
                loss_batch += loss
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Chamfer Loss: {avg_loss:.6f}")
