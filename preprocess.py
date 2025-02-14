import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data 
import matplotlib.pyplot as plt
import torch.nn.functional as F



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
    
def normalize_pair_with_bbox(dep_points, uav_points, bbox, normalization_type='mean_std', grid_size=32):
    """
    Given two point clouds (dep_points and uav_points) of shape [N, 3] and a 2D bounding box,
    compute grid indices for each point based on the original (x, y) coordinates and then
    normalize the point clouds using one of two strategies:
      - 'mean_std': normalization based on the combined point clouds' mean and max L2 distance.
      - 'bbox': normalization based on the provided bounding box for x,y and computed min/max for z.
    
    Parameters:
      dep_points (Tensor): [N_dep, 3] tensor of dependent point cloud coordinates.
      uav_points (Tensor): [N_uav, 3] tensor of UAV point cloud coordinates.
      bbox (tuple): (xmin, ymin, xmax, ymax) representing the global 2D bounding box.
      normalization_type (str): 'mean_std' or 'bbox' (default: 'mean_std').
      grid_size (int): Number of patches per side for the grid (default: 32).
      
    Returns:
      dep_points_norm (Tensor): [N_dep, 3] normalized dependent points.
      uav_points_norm (Tensor): [N_uav, 3] normalized UAV points.
      center (Tensor): [1, 3] center used for normalization.
      scale (Tensor): Scalar scale used for normalization.
      dep_grid_indices (Tensor): [N_dep] grid indices (int) for each dependent point.
      uav_grid_indices (Tensor): [N_uav] grid indices (int) for each UAV point.
    """
    # Unpack bounding box (assumed to be global for both point clouds)
    xmin, ymin, xmax, ymax = bbox

    # Compute grid indices for each point based on original x,y coordinates.
    # This uses the global bounding box regardless of normalization.

    #NOTE: I need to confirm that this will match what comes out of clay.
    def compute_grid_indices(points):
        # points: [N, 3] - we use only the first two coordinates (x, y)
        x = points[:, 0]
        y = points[:, 1]
        # Normalize to [0, 1] using bbox
        norm_x = (x - xmin) / (xmax - xmin)
        norm_y = (y - ymin) / (ymax - ymin)
        # Clamp in case points are slightly outside due to numerical issues
        norm_x = norm_x.clamp(0, 1)
        norm_y = norm_y.clamp(0, 1)
        # Convert to grid cell indices
        ix = (norm_x * grid_size).long().clamp(max=grid_size - 1)
        iy = (norm_y * grid_size).long().clamp(max=grid_size - 1)
        # Compute 1D index (row-major order)
        grid_indices = iy * grid_size + ix
        return grid_indices

    dep_grid_indices = compute_grid_indices(dep_points)
    uav_grid_indices = compute_grid_indices(uav_points)

    # Combine both point clouds to compute normalization parameters if needed.
    combined = torch.cat([dep_points, uav_points], dim=0)  # [N_dep+N_uav, 3]
    
    if normalization_type == 'mean_std':
        # Use the mean and the maximum L2 norm of the centered points.
        center = combined.mean(dim=0, keepdim=True)  # shape [1, 3]
        combined_centered = combined - center
        scale = combined_centered.norm(dim=1).max().clamp_min(1e-9)  # scalar
    elif normalization_type == 'bbox':
        # For x and y, use the provided bbox; for z, compute from the data.
        # Compute z min and max from the combined cloud:
        z_min = combined[:, 2].min()
        z_max = combined[:, 2].max()
        # Compute center: for x,y use the middle of the bbox; for z, use the average.
        center_xy = torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=combined.dtype, device=combined.device)
        center_z = combined[:, 2].mean().unsqueeze(0)
        center = torch.cat([center_xy, center_z], dim=0).unsqueeze(0)  # shape [1, 3]
        # Scale: use the max range among x, y (from bbox) and z (from data)
        scale_xy = max(xmax - xmin, ymax - ymin)
        scale_z = (z_max - z_min).item()  # convert to Python float
        scale = torch.tensor(max(scale_xy, scale_z), dtype=combined.dtype, device=combined.device).clamp_min(1e-9)
    else:
        raise ValueError("normalization_type must be either 'mean_std' or 'bbox'.")

    # Normalize the point clouds using the chosen parameters
    dep_points_norm = (dep_points - center) / scale
    uav_points_norm = (uav_points - center) / scale

    return dep_points_norm, uav_points_norm, center, scale, dep_grid_indices, uav_grid_indices

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"WARNING: {name} contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"WARNING: {name} contains INF values!")


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
        # bbox = sample['bbox']
        
        # dtm_1m = sample['dtm_1m'].squeeze(0)
        # dsm_1m = sample['dsm_1m'].squeeze(0)

        # dtm_50cm = sample['dtm_50cm'].squeeze(0)
        # dsm_50cm = sample['dsm_50cm'].squeeze(0)
        
        # wts = sample['ch_50cm'].squeeze(0)   # shape [N_uav]

        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points, uav_points)
        
        # Debug: Check that the normalized points are finite
        check_tensor(dep_points_norm, f"dep_points_norm (index {idx})")
        check_tensor(uav_points_norm, f"uav_points_norm (index {idx})")
        
        # Wrap the scalar values in lists so they are iterable.
        # return (dep_points_norm, uav_points_norm, edge_index, center, scale, 
        #         wts, dtm_1m, dsm_1m,dtm_50cm,dsm_50cm,bbox)
        return (dep_points_norm, uav_points_norm, edge_index)

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
        # bbox = sample['bbox'] #in same CRS as dep_points and uav_points

        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points, uav_points)
        
        return (dep_points_norm, uav_points_norm, edge_index)


def variable_size_collate(batch):
    dep_list, uav_list, edge_list = [], [], []
    for (dep_pts, uav_pts, e_idx) in batch:
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        
    return dep_list, uav_list, edge_list


def variable_size_collate(batch):
    dep_list, uav_list, edge_list = [], [], []
    # center_list, scale_list = [], []
    # wts_list = [] 
    # dtm_1m_list = [] 
    # dsm_1m_list = [] 
    # dtm_50cm_list = [] 
    # dsm_50cm_list = [] 
    # bbox_list = []
    for (dep_pts, uav_pts, e_idx) in batch:
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        
    return dep_list, uav_list, edge_list
    
    
    # for (dep_pts, uav_pts, e_idx, center, scale, wts, dtm_1m, dsm_1m,dtm_50cm,dsm_50cm, bbox) in batch:
    #     dep_list.append(dep_pts)
    #     uav_list.append(uav_pts)
    #     edge_list.append(e_idx)
    #     center_list.append(center)
    #     scale_list.append(scale)
    #     wts_list.append(wts)
    #     dtm_1m_list.append(dtm_1m)
    #     dsm_1m_list.append(dsm_1m)
    #     dtm_50cm_list.append(dtm_50cm)
    #     dsm_50cm_list.append(dsm_50cm)
    #     bbox_list.append(bbox)
    # return dep_list, uav_list, edge_list, center_list, scale_list, wts_list, dtm_1m_list, dsm_1m_list,dtm_50cm_list,dsm_50cm_list,bbox_list

