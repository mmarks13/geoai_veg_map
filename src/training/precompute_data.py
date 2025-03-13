import torch
from torch.utils.data import Dataset
from datetime import datetime
from typing import Dict, Any, List
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

##########################################
# Utility Functions for Date Parsing
##########################################

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string using multiple methods.
    
    First, attempts to use datetime.fromisoformat (which supports ISO 8601 strings
    with timezone offsets, e.g., "2023-10-25T00:00:00+00:00").
    If that fails, it will try common strptime formats.
    
    Returns:
      A datetime object.
    """
    try:
        # This handles ISO 8601 with timezone offsets.
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    # Fallback: Try common formats.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} not recognized.")



def compute_relative_dates(dates: List[str], reference_date: datetime) -> torch.Tensor:
    """
    Compute relative dates (in days) for a list of date strings relative to the reference_date.
    Both the reference_date and each date in the list are converted to date objects (ignoring time).
    
    Returns:
      A tensor of shape [n_images, 1] with the relative day differences.
    """
    ref_date_only = reference_date.date()
    rel_dates = []
    for date_str in dates:
        d = parse_date(date_str)
        d_date_only = d.date()
        delta_days = (d_date_only - ref_date_only).days
        rel_dates.append(float(delta_days))
    return torch.tensor(rel_dates, dtype=torch.float32).unsqueeze(1)

##########################################
# Imagery Preprocessing Functions
##########################################

def preprocess_naip_imagery(tile: Dict[str, Any], reference_date: datetime) -> Dict[str, Any]:
    """
    Preprocess NAIP imagery from the flattened data structure.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'naip_imgs': Tensor of shape [n_images, 4, h, w] (4 spectral bands)
         - 'naip_dates': List of date strings
         - 'naip_ids': List of image IDs
         - 'naip_img_bbox': NAIP imagery bounding box [minx, miny, maxx, maxy]
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      
    Returns:
      A dictionary with:
         - 'images': The NAIP imagery tensor
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The NAIP imagery bounding box
    """
    # Get NAIP imagery tensor
    images = tile['naip_imgs']  # Tensor: [n_images, 4, h, w]
    
    # Get dates and compute relative dates
    dates = tile['naip_dates']
    relative_dates = compute_relative_dates(dates, reference_date)
    
    return {
        'images': images,                  # Image tensor: [n_images, 4, h, w]
        'ids': tile['naip_ids'],           # List of image IDs
        'dates': dates,
        'relative_dates': relative_dates,  # Tensor: [n_images, 1]
        'img_bbox': tile['naip_img_bbox'], # Bounding box
        'bands': tile['naip_bands']        # Band information
    }

def preprocess_uavsar_imagery(tile: Dict[str, Any], reference_date: datetime) -> Dict[str, Any]:
    """
    Preprocess UAVSAR imagery from the flattened data structure.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'uavsar_imgs': Tensor of shape [n_images, n_bands, h, w]
         - 'uavsar_dates': List of date strings
         - 'uavsar_ids': List of image IDs
         - 'uavsar_img_bbox': UAVSAR imagery bounding box
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      
    Returns:
      A dictionary with:
         - 'images': The UAVSAR imagery tensor
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The UAVSAR imagery bounding box
    """
    # Get UAVSAR imagery tensor
    images = tile['uavsar_imgs']  # Tensor: [n_images, n_bands, h, w]
    
    # Get dates and compute relative dates
    dates = tile['uavsar_dates']
    relative_dates = compute_relative_dates(dates, reference_date)
    
    return {
        'images': images,                     # Image tensor: [n_images, n_bands, h, w]
        'ids': tile['uavsar_ids'],            # List of image IDs
        'dates': dates,   
        'relative_dates': relative_dates,     # Tensor: [n_images, 1]
        'img_bbox': tile['uavsar_img_bbox'],  # Bounding box
        'bands': tile['uavsar_bands']         # Band information
    }


##########################################
# Point Cloud Normalization Function
##########################################

def normalize_point_clouds_with_bbox(dep_points: torch.Tensor,
                                     uav_points: torch.Tensor,
                                     bbox: tuple,
                                     normalization_type: str = 'bbox',
                                     grid_size: int = 20):
    """
    Normalizes 3DEP (dep_points) and UAV point clouds to a common spatial coordinate
    system defined by a 2D bounding box, and computes grid indices for each point.
    
    Inputs:
      dep_points: [N_dep, 3] tensor of 3DEP point coordinates.
      uav_points: [N_uav, 3] tensor of UAV point coordinates.
      bbox: Tuple (xmin, ymin, xmax, ymax) defining the spatial extent.
      normalization_type: 'mean_std' or 'bbox'. 'bbox' normalizes x,y using bbox and z using data stats.
      grid_size: Number of patches per side (e.g., 32 -> 32x32 grid).
      
    Returns:
      dep_points_norm: [N_dep, 3] normalized 3DEP points.
      uav_points_norm: [N_uav, 3] normalized UAV points.
      center: [1, 3] tensor representing the normalization center.
      scale: Scalar tensor used for normalization.
      dep_grid_indices: [N_dep] tensor of grid indices (int) for each 3DEP point.
      uav_grid_indices: [N_uav] tensor of grid indices for each UAV point.
      grid_coords: [grid_size, grid_size, 2] tensor of normalized grid coordinates for positional encoding.
    """
    xmin, ymin, xmax, ymax = bbox

    def compute_grid_indices(points: torch.Tensor) -> torch.Tensor:
        # Use only x,y coordinates: points shape [N, 3]
        x = points[:, 0]
        y = points[:, 1]
        # Normalize x and y to the range [0, 1] using the bounding box.
        norm_x = (x - xmin) / (xmax - xmin)
        norm_y = (y - ymin) / (ymax - ymin)
        norm_x = norm_x.clamp(0, 1)
        norm_y = norm_y.clamp(0, 1)
        # Convert normalized coordinates to grid indices.
        ix = (norm_x * grid_size).long().clamp(max=grid_size - 1)
        iy = (norm_y * grid_size).long().clamp(max=grid_size - 1)
        # Compute a 1D grid index in row-major order.
        grid_indices = iy * grid_size + ix
        return grid_indices

    # Compute grid indices for both point clouds.
    dep_grid_indices = compute_grid_indices(dep_points)  # [N_dep]
    uav_grid_indices = compute_grid_indices(uav_points)  # [N_uav]

    # Combine both point clouds to compute normalization parameters.
    combined = torch.cat([dep_points, uav_points], dim=0)  # Shape: [N_dep + N_uav, 3]

    if normalization_type == 'mean_std':
        center = combined.mean(dim=0, keepdim=True)  # [1, 3]
        combined_centered = combined - center
        scale = combined_centered.norm(dim=1).max().clamp_min(1e-9)
    elif normalization_type == 'bbox':
        # For x and y, use the center of the bounding box; for z, use the mean of z.
        center_xy = torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2],
                                 dtype=combined.dtype, device=combined.device)
        center_z = combined[:, 2].mean().unsqueeze(0)
        center = torch.cat([center_xy, center_z], dim=0).unsqueeze(0)  # [1, 3]
        scale_xy = max(xmax - xmin, ymax - ymin)
        scale_z = (combined[:, 2].max() - combined[:, 2].min()).item()
        scale_value = max(scale_xy, scale_z)

        # Create scale tensor properly depending on its type
        if isinstance(scale_value, torch.Tensor):
            scale = scale_value.clone().detach().clamp_min(1e-9)
        else:
            # It's a scalar, so we can use torch.tensor() safely
            scale = torch.tensor(scale_value, dtype=combined.dtype, device=combined.device).clamp_min(1e-9)
    else:
        raise ValueError("normalization_type must be either 'mean_std' or 'bbox'.")

    # Normalize point clouds.
    dep_points_norm = (dep_points - center) / scale   # [N_dep, 3]
    uav_points_norm = (uav_points - center) / scale   # [N_uav, 3]

    # Create a grid of normalized coordinates for positional encoding.
    # Grid covers [0, 1] x [0, 1] domain.
    x_coords = torch.linspace(0, 1, grid_size)
    y_coords = torch.linspace(0, 1, grid_size)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape: [grid_size, grid_size, 2]

    return (dep_points_norm, uav_points_norm, center, scale,
            dep_grid_indices, uav_grid_indices, grid_coords)

##########################################
# Precomputation Function
##########################################

def precompute_dataset(data_list, normalization_type: str = 'bbox', grid_size: int = 20):
    """
    Precompute all necessary features for each tile in the dataset.
    This function is updated to work with the flattened data structure.
    
    For each sample (tile), this function performs:
      - Downsampling of UAV points using the first available downsample mask.
      - Normalization of 3DEP (dep_points) and downsampled UAV points to a common coordinate system,
        using the provided 2D bounding box.
      - Computation of grid indices for each point and a grid of normalized coordinates (for positional encoding).
      - Parsing of the UAV acquisition date and preprocessing of NAIP and UAVSAR imagery,
        including temporal sorting and computing relative acquisition dates.
      - Computation of KNN edge indices for 3DEP points for each k in the list.
    
    Returns:
      precomputed_data_list: List of dictionaries with keys:
         - 'dep_points_norm': [N_dep, 3] normalized 3DEP points.
         - 'uav_points_norm': [N_uav, 3] downsampled & normalized UAV points.
         - 'center': [1, 3] normalization center.
         - 'scale': Scalar normalization factor.
         - 'dep_grid_indices': [N_dep] grid indices for 3DEP points.
         - 'uav_grid_indices': [N_uav] grid indices for UAV points.
         - 'grid_coords': [grid_size, grid_size, 2] normalized grid coordinates.
         - 'knn_edge_indices': Dict mapping k to KNN edge index tensors of shape [2, E] for 3DEP points.
         - 'naip': Preprocessed NAIP imagery data.
         - 'uavsar': Preprocessed UAVSAR imagery data.
         - 'tile_id': Optional tile identifier.
    """
    precomputed_data_list = []
    # Define list of k values for KNN computation.
    # k_values = [10, 15]
    k_values = [10, 15, 20, 30, 40, 50, 60]
    
    for sample in data_list:
        # --- Point Cloud Preprocessing ---
        dep_points = sample['dep_points']       # [N_dep, 3] (3DEP points)
        dep_pnt_attr = sample['dep_pnt_attr']   # [N_dep, 3] (3DEP point attributes)
        uav_points = sample['uav_points']       # [N_uav, 3] (UAV points)
        uav_pnt_attr = sample['uav_pnt_attr']   # [N_uav, 3] (UAV point attributes)
        bbox = sample['bbox']                   # (xmin, ymin, xmax, ymax)
        
        # Downsample UAV points using the provided downsample mask
        if 'uav_downsample_mask' in sample and sample['uav_downsample_mask'] is not None:
            mask = sample['uav_downsample_mask']
            if mask.dtype != torch.bool:
                mask = mask.bool()
            uav_points = uav_points[mask]  # [N_uav_down, 3]
            if uav_pnt_attr is not None:
                uav_pnt_attr = uav_pnt_attr[mask]  # [N_uav_down, 3]
        # Alternatively, use the first mask from the list if available
        elif 'uav_downsample_masks' in sample and len(sample['uav_downsample_masks']) > 0:
            mask = sample['uav_downsample_masks'][0]
            if mask.dtype != torch.bool:
                mask = mask.bool()
            uav_points = uav_points[mask]  # [N_uav_down, 3]
            if uav_pnt_attr is not None:
                uav_pnt_attr = uav_pnt_attr[mask]  # [N_uav_down, 3]
        
        # Normalize point clouds and compute grid indices.
        (dep_points_norm, uav_points_norm, center, scale,
         dep_grid_indices, uav_grid_indices, grid_coords) = normalize_point_clouds_with_bbox(
            dep_points, uav_points, bbox, normalization_type=normalization_type, grid_size=grid_size
        )
        
        # --- KNN Edge Indices Computation ---
        # Compute KNN graphs for the normalized 3DEP points for each k in k_values.
        knn_edge_indices = {}
        for k in k_values:
            # print(f"Computing KNN graph for k={k}...")
            # Compute knn_graph: returns edge_index of shape [2, E]
            edge_index_k = knn_graph(dep_points_norm, k=k, loop=False)
            # Make graph undirected.
            edge_index_k = to_undirected(edge_index_k, num_nodes=dep_points_norm.size(0))
            knn_edge_indices[k] = edge_index_k
        # print("KNN graph computation complete.")
        
        # --- Imagery Preprocessing ---
        # Get reference date from UAV metadata
        ref_date_str = sample['uav_meta']['datetime']
        ref_date = parse_date(ref_date_str)
        
        # Process NAIP and UAVSAR imagery if available
        naip_preprocessed = None
        uavsar_preprocessed = None
        
        if sample.get('has_naip', False) and 'naip_imgs' in sample:
            naip_preprocessed = preprocess_naip_imagery(sample, ref_date)
            
        if sample.get('has_uavsar', False) and 'uavsar_imgs' in sample:
            uavsar_preprocessed = preprocess_uavsar_imagery(sample, ref_date)
        
        # Create unified precomputed sample.
        precomputed_sample = {
            'dep_points_norm': dep_points_norm,        # [N_dep, 3]
            'uav_points_norm': uav_points_norm,        # [N_uav_down, 3]
            'dep_points_attr': dep_pnt_attr,           # [N_dep, 3]
            'uav_points_attr': uav_pnt_attr if 'uav_pnt_attr' in sample else None,  # [N_uav_down, 3]
            'center': center,                          # [1, 3]
            'scale': scale,                            # Scalar
            'dep_grid_indices': dep_grid_indices,      # [N_dep]
            'uav_grid_indices': uav_grid_indices,      # [N_uav_down]
            'grid_coords': grid_coords,                # [grid_size, grid_size, 2]
            'knn_edge_indices': knn_edge_indices,      # Dict: k -> [2, E] edge indices for 3DEP
            'naip': naip_preprocessed,                 # Preprocessed NAIP imagery data
            'uavsar': uavsar_preprocessed,             # Preprocessed UAVSAR imagery data
            'tile_id': sample.get('tile_id', None)     # Optional identifier
        }
        
        precomputed_data_list.append(precomputed_sample)
    
    return precomputed_data_list

##########################################
# Main Script to Load, Precompute, and Save Data
##########################################

import warnings

def main():
    # File paths for the input datasets.
    training_file = 'data/processed/model_data/test/training_tiles.pt'
    validation_file = 'data/processed/model_data/test/validation_tiles.pt'
    test_file = 'data/processed/model_data/test/test_tiles.pt'
    
    # Load the PyTorch file containing the tiles
    print("Loading the PyTorch file...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                                message="You are using `torch.load` with `weights_only=False`")
        # Load datasets (each is expected to be a list of tile dictionaries).
        training_tiles = torch.load(training_file)
        validation_tiles = torch.load(validation_file)
        test_tiles = torch.load(test_file)
    
    print("Precomputing training dataset...")
    precomputed_training = precompute_dataset(training_tiles, normalization_type='bbox', grid_size=20)
    print("Precomputing validation dataset...")
    precomputed_validation = precompute_dataset(validation_tiles, normalization_type='bbox', grid_size=20)
    print("Precomputing test dataset...")
    precomputed_test = precompute_dataset(test_tiles, normalization_type='bbox', grid_size=20)
    
    # Save the precomputed datasets.
    torch.save(precomputed_training, 'data/processed/model_data/precomputed_training_tiles.pt')
    torch.save(precomputed_validation, 'data/processed/model_data/precomputed_validation_tiles.pt')
    torch.save(precomputed_test, 'data/processed/model_data/precomputed_test_tiles.pt')
    
    print("Precomputation complete. Precomputed files saved.")

if __name__ == '__main__':
    main()