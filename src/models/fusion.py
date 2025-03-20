import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import math
import torch
import torch.nn as nn
from torch_geometric.nn import PointTransformerConv


# Note on tensor dimensions:
# N: Number of points in point cloud
# P: Number of patches (typically 16 for a 4x4 grid)
# D_p: Point feature dimension
# D_patch: Patch embedding dimension
# E: Number of edges in the graph

class SpatialFusion(nn.Module):
    """
    An improved spatial fusion module that properly utilizes PointTransformerConv
    for spatially-aware feature transformation.
    """
    def __init__(
        self,
        point_dim,              # Dimension of point features
        patch_dim=32,           # Dimension of patch embeddings
        use_naip=False,         # Whether to use NAIP features
        use_uavsar=False,       # Whether to use UAVSAR features
        num_patches=16,         # Number of patch embeddings per modality
        temperature=0.1,        # Temperature parameter for distance weighting
        max_dist_ratio=0.5      # Maximum distance ratio for patch influence
    ):
        super().__init__()
        self.point_dim = point_dim
        self.patch_dim = patch_dim
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
        self.num_patches = num_patches
        self.temperature = temperature
        self.max_dist_ratio = max_dist_ratio
        
        # If neither modality is used, this module becomes a pass-through
        if not (use_naip or use_uavsar):
            self.identity = True
            return
        else:
            self.identity = False
        
        # Calculate output dimension after concatenation
        concat_dim = point_dim
        if use_naip:
            concat_dim += patch_dim
        if use_uavsar:
            concat_dim += patch_dim
        
        # Intermediate dimension reduction before PointTransformerConv (optional)
        self.pre_projection = nn.Linear(concat_dim, concat_dim)
        
        # Use PointTransformerConv for spatially-aware projection
        self.pt_conv_projection = PointTransformerConv(
            in_channels=concat_dim, 
            out_channels=point_dim,
            pos_nn=nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, point_dim)
            ),
            attn_nn=nn.Sequential(
                nn.Linear(point_dim, 64),
                nn.ReLU(),
                nn.Linear(64, point_dim)
            )
        )
    
    def get_patch_positions(self, img_bbox, patches_per_side, center, scale):
        """
        Compute normalized positions of patches within an image bbox
        
        Args:
            img_bbox: [minx, miny, maxx, maxy] of the image
            patches_per_side: Number of patches per side (e.g., 4 for 4x4 grid)
            center: [1, 3] Center used for point cloud normalization
            scale: Scale used for point cloud normalization
            
        Returns:
            normalized_positions: [num_patches, 2] tensor with normalized x,y coordinates
        """
        device = img_bbox.device if isinstance(img_bbox, torch.Tensor) else torch.device('cpu')
        
        # Convert to tensor if needed
        if not isinstance(img_bbox, torch.Tensor):
            img_bbox = torch.tensor(img_bbox, device=device, dtype=torch.float32)  # [4]
        
        minx, miny, maxx, maxy = img_bbox
        
        # Calculate patch size
        patch_size_x = (maxx - minx) / patches_per_side
        patch_size_y = (maxy - miny) / patches_per_side
        
        # Create grid of patch centers
        x_centers = torch.linspace(
            minx + patch_size_x/2, 
            maxx - patch_size_x/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        y_centers = torch.linspace(
            miny + patch_size_y/2, 
            maxy - patch_size_y/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        # Create all combinations
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')  # Both [patches_per_side, patches_per_side]
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [patches_per_side^2, 2] = [num_patches, 2]
        
        # Normalize positions using the same center and scale as the point cloud
        center_xy = center[:, :2].to(device)  # [1, 2]
        normalized_positions = (positions - center_xy) / scale  # [num_patches, 2]
        
        return normalized_positions
    
    def process_patch_embeddings(self, point_positions, patch_embeddings, patch_bbox, center, scale):
        """
        Process patch embeddings for fusion with point features
        
        Args:
            point_positions: Normalized point positions [N, 2]
            patch_embeddings: Patch embeddings [P, D_patch] where P = num_patches
            patch_bbox: Bounding box of the imagery [minx, miny, maxx, maxy]
            center: Center used for point cloud normalization [1, 3]
            scale: Scale used for point cloud normalization
            
        Returns:
            patch_contribution: Weighted patch features for each point [N, D_patch]
        """
        device = point_positions.device
        
        # Move patch embeddings to the correct device
        patch_embeddings = patch_embeddings.to(device)  # [P, D_patch]
        
        # Calculate normalized patch positions
        patches_per_side = int(math.sqrt(self.num_patches))
        patch_positions = self.get_patch_positions(
            patch_bbox, 
            patches_per_side,
            center,
            scale
        ).to(device)  # [P, 2]
        
        # Calculate distances between points and patches
        squared_diffs = (
            point_positions.unsqueeze(1) -  # [N, 1, 2]
            patch_positions.unsqueeze(0)    # [1, P, 2]
        ).pow(2)  # [N, P, 2]
        
        distances = torch.sqrt(squared_diffs.sum(dim=-1))  # [N, P]
        
        # Apply hard distance cutoff
        distance_mask = (distances <= self.max_dist_ratio)  # [N, P]
        
        # Apply soft distance weighting
        weights = torch.exp(-distances / self.temperature)  # [N, P]
        weights = weights * distance_mask.float()  # [N, P]
        
        # Normalize weights
        weight_sums = weights.sum(dim=1, keepdim=True)  # [N, 1]
        valid_points = (weight_sums > 1e-6).float()  # [N, 1]
        weights = weights / (weight_sums + 1e-10)  # [N, P]
        
        # Weight patch features for each point
        patch_contribution = torch.matmul(weights, patch_embeddings)  # [N, P] @ [P, D_patch] = [N, D_patch]
        
        # Zero out contribution for points with no valid patches
        patch_contribution = patch_contribution * valid_points  # [N, D_patch]
        
        return patch_contribution
    
    def forward(self, point_features, edge_index, point_positions, naip_embeddings=None, uavsar_embeddings=None,
                main_bbox=None, naip_bbox=None, uavsar_bbox=None, center=None, scale=None):
        """
        Fuse point features with patch embeddings based on spatial proximity,
        using PointTransformerConv for spatially-aware feature transformation
        
        Args:
            point_features: Point features [N, D_p]
            edge_index: Edge indices for graph connectivity [2, E]
            point_positions: Point positions in 3D space [N, 3]
            naip_embeddings: NAIP patch embeddings [P, D_patch] or None
            uavsar_embeddings: UAVSAR patch embeddings [P, D_patch] or None
            main_bbox: Bounding box of the point cloud [xmin, ymin, xmax, ymax]
            naip_bbox: Bounding box of NAIP imagery [minx, miny, maxx, maxy]
            uavsar_bbox: Bounding box of UAVSAR imagery [minx, miny, maxx, maxy]
            center: Center used for point cloud normalization [1, 3]
            scale: Scale used for point cloud normalization
            
        Returns:
            fused_features: Point features enhanced with patch information [N, D_p]
        """
        # Identity case - no imagery modalities used
        if self.identity:
            return point_features  # [N, D_p]
        
        # If we don't have position information, we can't do spatial fusion
        if point_positions is None or center is None or scale is None:
            return point_features  # [N, D_p]
        
        # Get device
        device = point_features.device
        point_positions = point_positions.to(device)  # [N, 3]
        
        # Process modalities
        to_concat = [point_features]  # List containing [N, D_p]
        
        # Process NAIP modality if available
        if self.use_naip and naip_embeddings is not None and naip_bbox is not None:
            naip_contribution = self.process_patch_embeddings(
                point_positions[:, :2],  # Using only x,y coordinates for spatial weighting
                naip_embeddings, 
                naip_bbox, 
                center, 
                scale
            )  # [N, D_patch]
            to_concat.append(naip_contribution)
        
        # Process UAVSAR modality if available
        if self.use_uavsar and uavsar_embeddings is not None and uavsar_bbox is not None:
            uavsar_contribution = self.process_patch_embeddings(
                point_positions[:, :2],  # Using only x,y coordinates for spatial weighting
                uavsar_embeddings, 
                uavsar_bbox, 
                center, 
                scale
            )  # [N, D_patch]
            to_concat.append(uavsar_contribution)
        
        # Concatenate features
        concatenated = torch.cat(to_concat, dim=1)  # [N, D_p + D_patch] or [N, D_p + 2*D_patch] if both modalities
        
        # Apply optional preprocessing before PointTransformerConv
        concatenated = self.pre_projection(concatenated)  # [N, concat_dim]
        
        # Apply PointTransformerConv with spatial awareness
        fused_features = self.pt_conv_projection(
            x=concatenated,           # [N, concat_dim]
            pos=point_positions,      # [N, 3]
            edge_index=edge_index     # [2, E]
        )  # [N, D_p]
        
        return fused_features