import torch
from torch.utils.data import Dataset
from datetime import datetime
from typing import Dict, Any, List
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import numpy as np

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
# Band Statistics Computation
##########################################

def compute_global_band_statistics(data_list, imagery_type='naip'):
    """
    Compute global mean and standard deviation for each band across the entire dataset.
    Handles NA, NaN, and Inf values by ignoring them in the computation.
    
    Inputs:
        data_list: List of sample dictionaries.
        imagery_type: Either 'naip' or 'uavsar'.
        
    Returns:
        means: Tensor of shape [n_bands] with mean values for each band.
        stds: Tensor of shape [n_bands] with standard deviation values for each band.
        invalid_counts: Dict with counts of invalid values per band.
    """
    # Check if imagery exists in the dataset
    img_key = f'{imagery_type}_imgs'
    if not any(img_key in sample for sample in data_list):
        print(f"No {imagery_type} imagery found in the dataset.")
        return None, None, None
    
    # First find number of bands from first valid sample
    n_bands = None
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            n_bands = sample[img_key].shape[1]
            break
    
    if n_bands is None:
        print(f"Could not determine number of bands for {imagery_type}.")
        return None, None, None
    
    # Initialize statistics
    sum_pixels = torch.zeros(n_bands, dtype=torch.float64)
    count_valid_pixels = torch.zeros(n_bands, dtype=torch.int64)
    invalid_counts = {
        'nan': torch.zeros(n_bands, dtype=torch.int64),
        'inf': torch.zeros(n_bands, dtype=torch.int64)
    }
    
    # First pass: compute mean and count invalid values
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            imgs = sample[img_key]  # Shape: [n_images, n_bands, height, width]
            
            # Process each band separately to handle invalid values
            for band_idx in range(n_bands):
                band_data = imgs[:, band_idx, :, :]  # Shape: [n_images, height, width]
                
                # Count NaN values
                nan_mask = torch.isnan(band_data)
                invalid_counts['nan'][band_idx] += nan_mask.sum().item()
                
                # Count Inf values
                inf_mask = torch.isinf(band_data)
                invalid_counts['inf'][band_idx] += inf_mask.sum().item()
                
                # Create combined mask of valid values
                valid_mask = ~(nan_mask | inf_mask)
                
                # Sum valid values and count them
                sum_pixels[band_idx] += band_data[valid_mask].sum().to(torch.float64)
                count_valid_pixels[band_idx] += valid_mask.sum().item()
    
    # Print invalid value statistics
    total_pixels = sum(count_valid_pixels).item()
    for band_idx in range(n_bands):
        total_invalid = invalid_counts['nan'][band_idx] + invalid_counts['inf'][band_idx]
        print(f"{imagery_type} Band {band_idx}:{count_valid_pixels[band_idx]} valid pixels, {total_invalid} invalid values ({invalid_counts['nan'][band_idx]} NaN, {invalid_counts['inf'][band_idx]} Inf)")
    
    # Check if we have enough valid pixels
    if torch.any(count_valid_pixels == 0):
        print(f"Some {imagery_type} bands have no valid pixels!")
        zero_bands = torch.where(count_valid_pixels == 0)[0].tolist()
        print(f"Bands with no valid pixels: {zero_bands}")
        return None, None, invalid_counts
    
    # Calculate means
    means = sum_pixels / count_valid_pixels
    
    # Second pass: compute standard deviation
    sum_squared_diff = torch.zeros_like(means)
    
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            imgs = sample[img_key]
            
            # Process each band separately
            for band_idx in range(n_bands):
                band_data = imgs[:, band_idx, :, :]
                
                # Create mask of valid values
                valid_mask = ~(torch.isnan(band_data) | torch.isinf(band_data))
                
                # Calculate squared differences for valid values only
                valid_data = band_data[valid_mask]
                if valid_data.numel() > 0:  # Check if we have any valid data
                    diff = valid_data - means[band_idx].to(valid_data.dtype)
                    sum_squared_diff[band_idx] += (diff * diff).sum().to(torch.float64)
    
    # Calculate standard deviations
    stds = torch.sqrt(sum_squared_diff / count_valid_pixels)
    
    # Ensure no zero standard deviations
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    
    return means, stds, invalid_counts

##########################################
# Imagery Preprocessing Functions
##########################################

def preprocess_naip_imagery(tile: Dict[str, Any], reference_date: datetime, naip_means=None, naip_stds=None) -> Dict[str, Any]:
    """
    Preprocess NAIP imagery from the flattened data structure.
    Handles NA, NaN, and Inf values by setting them to 0 after normalization.
    Converts normalized values to float16 for memory efficiency.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'naip_imgs': Tensor of shape [n_images, 4, h, w] (4 spectral bands)
         - 'naip_dates': List of date strings
         - 'naip_ids': List of image IDs
         - 'naip_img_bbox': NAIP imagery bounding box [minx, miny, maxx, maxy]
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      naip_means: Optional tensor of shape [n_bands] with mean values for each band.
      naip_stds: Optional tensor of shape [n_bands] with standard deviation values for each band.
      
    Returns:
      A dictionary with:
         - 'images': The normalized NAIP imagery tensor in float16 format
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The NAIP imagery bounding box
    """
    # Get NAIP imagery tensor
    images = tile['naip_imgs'].clone()  # Tensor: [n_images, 4, h, w]
    
    # Identify invalid values before normalization
    invalid_mask = torch.isnan(images) | torch.isinf(images)
    
    # Normalize imagery if statistics are provided
    if naip_means is not None and naip_stds is not None:
        # Set invalid values to means temporarily for normalization
        # This ensures they become 0 after normalization
        for band_idx in range(images.shape[1]):
            band_invalid_mask = invalid_mask[:, band_idx, :, :]
            if band_invalid_mask.any():
                images[:, band_idx, :, :][band_invalid_mask] = naip_means[band_idx].to(images.dtype)
        
        # Ensure means and stds have the right shape for broadcasting
        means = naip_means.view(1, -1, 1, 1).to(images.dtype)
        stds = naip_stds.view(1, -1, 1, 1).to(images.dtype)
        
        # Normalize
        images = (images - means) / stds
        
        # Set any new invalid values (that might appear after normalization) to 0
        # which is the mean in normalized space
        images[torch.isnan(images) | torch.isinf(images)] = 0.0
        
        # Convert to float16 for memory efficiency
        images = images.to(torch.float16)
    
    # Get dates and compute relative dates
    dates = tile['naip_dates']
    relative_dates = compute_relative_dates(dates, reference_date)
    
    return {
        'images': images,                  # Normalized image tensor: [n_images, 4, h, w] in float16
        'ids': tile['naip_ids'],           # List of image IDs
        'dates': dates,
        'relative_dates': relative_dates,  # Tensor: [n_images, 1]
        'img_bbox': tile['naip_img_bbox'], # Bounding box
        'bands': tile['naip_bands']        # Band information
    }

def preprocess_uavsar_imagery(tile: Dict[str, Any], reference_date: datetime, uavsar_means=None, uavsar_stds=None) -> Dict[str, Any]:
    """
    Preprocess UAVSAR imagery from the flattened data structure.
    Handles NA, NaN, and Inf values by setting them to 0 after normalization.
    Converts normalized values to float16 for memory efficiency.
    Removes images that have all invalid values across all pixels and bands.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'uavsar_imgs': Tensor of shape [n_images, n_bands, h, w]
         - 'uavsar_dates': List of date strings
         - 'uavsar_ids': List of image IDs
         - 'uavsar_img_bbox': UAVSAR imagery bounding box
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      uavsar_means: Optional tensor of shape [n_bands] with mean values for each band.
      uavsar_stds: Optional tensor of shape [n_bands] with standard deviation values for each band.
      
    Returns:
      A dictionary with:
         - 'images': The normalized UAVSAR imagery tensor in float16 format
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The UAVSAR imagery bounding box
    """
    # Get UAVSAR imagery tensor
    images = tile['uavsar_imgs'].clone()  # Tensor: [n_images, n_bands, h, w]
    dates = tile['uavsar_dates']
    ids = tile['uavsar_ids']
    
    # Create a mask to identify images that have at least some valid pixels
    n_images = images.shape[0]
    valid_image_mask = torch.zeros(n_images, dtype=torch.bool, device=images.device)
    
    # Check each image for any valid pixels
    for img_idx in range(n_images):
        img = images[img_idx]  # Shape: [n_bands, h, w]
        # If ANY value in the image is valid, keep the image
        valid_image_mask[img_idx] = not (torch.isnan(img) | torch.isinf(img)).all()
    
    # Count and report invalid images
    invalid_count = n_images - valid_image_mask.sum().item()
    # if invalid_count > 0:
        # print(f"Removing {invalid_count} UAVSAR images with all invalid values.")
    
    # Return early if all images are invalid
    if valid_image_mask.sum() == 0:
        print("WARNING: All UAVSAR images have invalid values only!")
        return None
    
    # Filter images and corresponding metadata
    images = images[valid_image_mask]
    dates = [date for i, date in enumerate(dates) if valid_image_mask[i]]
    ids = [id for i, id in enumerate(ids) if valid_image_mask[i]]
    
    # Identify remaining invalid values before normalization
    invalid_mask = torch.isnan(images) | torch.isinf(images)
    
    # Normalize imagery if statistics are provided
    if uavsar_means is not None and uavsar_stds is not None:
        # Set invalid values to means temporarily for normalization
        # This ensures they become 0 after normalization
        for band_idx in range(images.shape[1]):
            band_invalid_mask = invalid_mask[:, band_idx, :, :]
            if band_invalid_mask.any():
                images[:, band_idx, :, :][band_invalid_mask] = uavsar_means[band_idx].to(images.dtype)
        
        # Ensure means and stds have the right shape for broadcasting
        means = uavsar_means.view(1, -1, 1, 1).to(images.dtype)
        stds = uavsar_stds.view(1, -1, 1, 1).to(images.dtype)
        
        # Normalize
        images = (images - means) / stds
        
        # Set any new invalid values (that might appear after normalization) to 0
        # which is the mean in normalized space
        images[torch.isnan(images) | torch.isinf(images)] = 0.0
        
        # Convert to float16 for memory efficiency
        images = images.to(torch.float16)
    
    # Compute relative dates for the filtered dates
    relative_dates = compute_relative_dates(dates, reference_date)
    
    return {
        'images': images,                     # Normalized image tensor: [n_valid_images, n_bands, h, w] in float16
        'ids': ids,                           # Filtered list of image IDs
        'dates': dates,                       # Filtered list of dates
        'relative_dates': relative_dates,     # Tensor: [n_valid_images, 1]
        'img_bbox': tile['uavsar_img_bbox'],  # Bounding box
        'bands': tile['uavsar_bands']         # Band information
    }


##########################################
# Point Cloud and Attribute Normalization Functions
##########################################

def compute_point_attr_statistics(data_list):
    """
    Compute global mean and standard deviation for point attributes across the entire dataset.
    Handles NA, NaN, and Inf values by ignoring them in the computation.
    
    Inputs:
        data_list: List of sample dictionaries.
        
    Returns:
        means: Tensor of shape [n_attr] with mean values for each attribute.
        stds: Tensor of shape [n_attr] with standard deviation values for each attribute.
        invalid_counts: Dict with counts of invalid values per attribute.
    """
    # Check if dep_pnt_attr exists in the dataset
    if not any('dep_pnt_attr' in sample for sample in data_list):
        print("No point attributes found in the dataset.")
        return None, None, None
    
    # First find number of attributes from first valid sample
    n_attr = None
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            n_attr = sample['dep_pnt_attr'].shape[1]
            break
    
    if n_attr is None:
        print("Could not determine number of point attributes.")
        return None, None, None
    
    # Initialize statistics
    sum_attrs = torch.zeros(n_attr, dtype=torch.float64)
    count_valid_attrs = torch.zeros(n_attr, dtype=torch.int64)
    invalid_counts = {
        'nan': torch.zeros(n_attr, dtype=torch.int64),
        'inf': torch.zeros(n_attr, dtype=torch.int64)
    }
    
    # First pass: compute mean and count invalid values
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            attrs = sample['dep_pnt_attr']  # Shape: [n_points, n_attr]
            
            # Process each attribute separately to handle invalid values
            for attr_idx in range(n_attr):
                attr_data = attrs[:, attr_idx]  # Shape: [n_points]
                
                # Count NaN values
                nan_mask = torch.isnan(attr_data)
                invalid_counts['nan'][attr_idx] += nan_mask.sum().item()
                
                # Count Inf values
                inf_mask = torch.isinf(attr_data)
                invalid_counts['inf'][attr_idx] += inf_mask.sum().item()
                
                # Create combined mask of valid values
                valid_mask = ~(nan_mask | inf_mask)
                
                # Sum valid values and count them
                sum_attrs[attr_idx] += attr_data[valid_mask].sum().to(torch.float64)
                count_valid_attrs[attr_idx] += valid_mask.sum().item()
    
    # Print invalid value statistics
    total_attrs = sum(count_valid_attrs).item()
    for attr_idx in range(n_attr):
        total_invalid = invalid_counts['nan'][attr_idx] + invalid_counts['inf'][attr_idx]
        print(f"Point attribute {attr_idx}: {total_invalid} invalid values ({invalid_counts['nan'][attr_idx]} NaN, {invalid_counts['inf'][attr_idx]} Inf)")
    
    # Check if we have enough valid attributes
    if torch.any(count_valid_attrs == 0):
        print(f"Some point attributes have no valid values!")
        zero_attrs = torch.where(count_valid_attrs == 0)[0].tolist()
        print(f"Attributes with no valid values: {zero_attrs}")
        return None, None, invalid_counts
    
    # Calculate means
    means = sum_attrs / count_valid_attrs
    
    # Second pass: compute standard deviation
    sum_squared_diff = torch.zeros_like(means)
    
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            attrs = sample['dep_pnt_attr']
            
            # Process each attribute separately
            for attr_idx in range(n_attr):
                attr_data = attrs[:, attr_idx]
                
                # Create mask of valid values
                valid_mask = ~(torch.isnan(attr_data) | torch.isinf(attr_data))
                
                # Calculate squared differences for valid values only
                valid_data = attr_data[valid_mask]
                if valid_data.numel() > 0:  # Check if we have any valid data
                    diff = valid_data - means[attr_idx].to(valid_data.dtype)
                    sum_squared_diff[attr_idx] += (diff * diff).sum().to(torch.float64)
    
    # Calculate standard deviations
    stds = torch.sqrt(sum_squared_diff / count_valid_attrs)
    
    # Ensure no zero standard deviations
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    
    return means, stds, invalid_counts

def normalize_point_clouds_with_bbox(dep_points: torch.Tensor,
                                     uav_points: torch.Tensor,
                                     bbox: tuple,
                                     normalization_type: str = 'bbox'):
    """
    Normalizes 3DEP (dep_points) and UAV point clouds to a common spatial coordinate
    system defined by a 2D bounding box.
    
    Inputs:
      dep_points: [N_dep, 3] tensor of 3DEP point coordinates.
      uav_points: [N_uav, 3] tensor of UAV point coordinates.
      bbox: Tuple (xmin, ymin, xmax, ymax) defining the spatial extent.
      normalization_type: 'mean_std' or 'bbox'. 'bbox' normalizes x,y using bbox and z using data stats.
      
    Returns:
      dep_points_norm: [N_dep, 3] normalized 3DEP points.
      uav_points_norm: [N_uav, 3] normalized UAV points.
      center: [1, 3] tensor representing the normalization center.
      scale: Scalar tensor used for normalization.
    """
    xmin, ymin, xmax, ymax = bbox

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
    

    return dep_points_norm, uav_points_norm, center, scale

##########################################
# Precomputation Function
##########################################

def precompute_dataset(data_list, naip_means=None, naip_stds=None, uavsar_means=None, uavsar_stds=None, 
                      point_attr_means=None, point_attr_stds=None, normalization_type: str = 'bbox'):
    """
    Precompute all necessary features for each tile in the dataset.
    This function is updated to work with the flattened data structure.
    
    For each sample (tile), this function performs:
      - Downsampling of UAV points using the first available downsample mask.
      - Normalization of 3DEP (dep_points) and downsampled UAV points to a common coordinate system,
        using the provided 2D bounding box.
      - Normalization of point attributes using global mean and standard deviation.
      - Parsing of the UAV acquisition date and preprocessing of NAIP and UAVSAR imagery,
        including temporal sorting, computing relative acquisition dates, and band normalization.
      - Computation of KNN edge indices for 3DEP points for each k in the list.
    
    Inputs:
      data_list: List of sample dictionaries.
      naip_means: Optional tensor of shape [n_bands] with mean values for NAIP bands.
      naip_stds: Optional tensor of shape [n_bands] with standard deviation values for NAIP bands.
      uavsar_means: Optional tensor of shape [n_bands] with mean values for UAVSAR bands.
      uavsar_stds: Optional tensor of shape [n_bands] with standard deviation values for UAVSAR bands.
      point_attr_means: Optional tensor of shape [n_attr] with mean values for point attributes.
      point_attr_stds: Optional tensor of shape [n_attr] with standard deviation values for point attributes.
      normalization_type: 'mean_std' or 'bbox'. 'bbox' normalizes x,y using bbox and z using data stats.
      
    Returns:
      precomputed_data_list: List of dictionaries with keys:
         - 'dep_points_norm': [N_dep, 3] normalized 3DEP points in float16.
         - 'uav_points_norm': [N_uav, 3] downsampled & normalized UAV points in float16.
         - 'dep_points_attr_norm': [N_dep, n_attr] normalized point attributes in float16.
         - 'center': [1, 3] normalization center.
         - 'scale': Scalar normalization factor.
         - 'knn_edge_indices': Dict mapping k to KNN edge index tensors of shape [2, E] for 3DEP points.
         - 'naip': Preprocessed and normalized NAIP imagery data in float16.
         - 'uavsar': Preprocessed and normalized UAVSAR imagery data in float16.
         - 'tile_id': Optional tile identifier.
    """
    precomputed_data_list = []
    # Define list of k values for KNN computation.
    # k_values = [10, 15]
    k_values = [15]
    
    for sample in data_list:
        # --- Point Cloud Preprocessing ---
        dep_points = sample['dep_points']       # [N_dep, 3] (3DEP points)
        dep_pnt_attr = sample['dep_pnt_attr']   # [N_dep, n_attr] (3DEP point attributes)
        uav_points = sample['uav_points']       # [N_uav, 3] (UAV points)
        uav_pnt_attr = sample['uav_pnt_attr']   # [N_uav, n_attr] (UAV point attributes)
        bbox = sample['bbox']                   # (xmin, ymin, xmax, ymax)
        
        # Downsample UAV points using the provided downsample mask
        if 'uav_downsample_mask' in sample and sample['uav_downsample_mask'] is not None:
            mask = sample['uav_downsample_mask']
            if mask.dtype != torch.bool:
                mask = mask.bool()
            uav_points = uav_points[mask]  # [N_uav_down, 3]
            if uav_pnt_attr is not None:
                uav_pnt_attr = uav_pnt_attr[mask]  # [N_uav_down, n_attr]
        # Alternatively, use the first mask from the list if available
        elif 'uav_downsample_masks' in sample and len(sample['uav_downsample_masks']) > 0:
            mask = sample['uav_downsample_masks'][0]
            if mask.dtype != torch.bool:
                mask = mask.bool()
            uav_points = uav_points[mask]  # [N_uav_down, 3]
            if uav_pnt_attr is not None:
                uav_pnt_attr = uav_pnt_attr[mask]  # [N_uav_down, n_attr]
        
        # Normalize point clouds
        dep_points_norm, uav_points_norm, center, scale = normalize_point_clouds_with_bbox(
            dep_points, uav_points, bbox, normalization_type=normalization_type
        )
        
        # Normalize point attributes if statistics are provided
        dep_points_attr_norm = None
        if dep_pnt_attr is not None and point_attr_means is not None and point_attr_stds is not None:
            # Create a copy to avoid modifying the original
            dep_points_attr_norm = dep_pnt_attr.clone()
            
            # Handle invalid values
            invalid_mask = torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)
            
            # Apply normalization
            if invalid_mask.any():
                # Set invalid values to means temporarily
                for attr_idx in range(dep_points_attr_norm.shape[1]):
                    attr_invalid_mask = invalid_mask[:, attr_idx]
                    if attr_invalid_mask.any():
                        dep_points_attr_norm[:, attr_idx][attr_invalid_mask] = point_attr_means[attr_idx].to(dep_points_attr_norm.dtype)
            
            # Normalize using global stats
            means = point_attr_means.to(dep_points_attr_norm.dtype)
            stds = point_attr_stds.to(dep_points_attr_norm.dtype)
            dep_points_attr_norm = (dep_points_attr_norm - means) / stds
            
            # Set any remaining invalid values to 0 (normalized mean)
            dep_points_attr_norm[torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)] = 0.0
            
            # Convert to float16
            dep_points_attr_norm = dep_points_attr_norm.to(torch.float16)
        
        # --- KNN Edge Indices Computation ---
        # Compute KNN graphs for the normalized 3DEP points for each k in k_values.
        knn_edge_indices = {}
        for k in k_values:
            # Compute knn_graph: returns edge_index of shape [2, E]
            edge_index_k = knn_graph(dep_points_norm, k=k, loop=False)
            # Make graph undirected.
            edge_index_k = to_undirected(edge_index_k, num_nodes=dep_points_norm.size(0))
            knn_edge_indices[k] = edge_index_k
        
        # --- Imagery Preprocessing ---
        # Get reference date from UAV metadata
        ref_date_str = sample['uav_meta']['datetime']
        ref_date = parse_date(ref_date_str)
        
        # Process NAIP and UAVSAR imagery if available
        naip_preprocessed = None
        uavsar_preprocessed = None
        
        if sample.get('has_naip', False) and 'naip_imgs' in sample:
            naip_preprocessed = preprocess_naip_imagery(sample, ref_date, naip_means, naip_stds)
            
        if sample.get('has_uavsar', False) and 'uavsar_imgs' in sample:
            uavsar_preprocessed = preprocess_uavsar_imagery(sample, ref_date, uavsar_means, uavsar_stds)
        
        # Create unified precomputed sample.
        precomputed_sample = {
            'dep_points_norm': dep_points_norm,         # [N_dep, 3] in float16
            'uav_points_norm': uav_points_norm,         # [N_uav_down, 3] in float16
            'dep_points_attr_norm': dep_points_attr_norm, # [N_dep, n_attr] in float16
            'center': center,                           # [1, 3]
            'scale': scale,                             # Scalar
            'knn_edge_indices': knn_edge_indices,       # Dict: k -> [2, E] edge indices for 3DEP
            'naip': naip_preprocessed,                  # Preprocessed and normalized NAIP imagery data in float16
            'uavsar': uavsar_preprocessed,              # Preprocessed and normalized UAVSAR imagery data in float16
            'tile_id': sample.get('tile_id', None)      # Optional identifier
        }
        
        precomputed_data_list.append(precomputed_sample)
    
    return precomputed_data_list

##########################################
# Main Script to Load, Precompute, and Save Data
##########################################

import warnings

def main():
    # File paths for the input datasets.
    training_file = 'data/processed/model_data/training_tiles.pt'
    validation_file = 'data/processed/model_data/validation_tiles.pt'
    test_file = 'data/processed/model_data/test_tiles.pt'
    
    # Load the PyTorch file containing the tiles
    print("Loading the PyTorch file...")
    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        # # Explicitly set weights_only=False to fully load the file.
        training_tiles = torch.load(training_file, weights_only=False)
        validation_tiles = torch.load(validation_file, weights_only=False)
        test_tiles = torch.load(test_file, weights_only=False)

    # Combine all datasets for computing statistics
    all_tiles = training_tiles + validation_tiles + test_tiles

    
    # Compute global band statistics for NAIP and UAVSAR imagery
    print("Computing global band statistics...")
    naip_means, naip_stds, naip_invalid_counts = compute_global_band_statistics(all_tiles, imagery_type='naip')
    uavsar_means, uavsar_stds, uavsar_invalid_counts = compute_global_band_statistics(all_tiles, imagery_type='uavsar')
    
    # Compute global point attribute statistics
    print("Computing point attribute statistics...")
    point_attr_means, point_attr_stds, point_attr_invalid_counts = compute_point_attr_statistics(all_tiles)
    
    print("NAIP band statistics:")
    if naip_means is not None and naip_stds is not None:
        for i, (mean, std) in enumerate(zip(naip_means, naip_stds)):
            print(f"  Band {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(naip_means)):
            nan_count = naip_invalid_counts['nan'][i].item() if naip_invalid_counts else 0
            inf_count = naip_invalid_counts['inf'][i].item() if naip_invalid_counts else 0
            print(f"    Band {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No NAIP imagery found.")
        
    print("UAVSAR band statistics:")
    if uavsar_means is not None and uavsar_stds is not None:
        for i, (mean, std) in enumerate(zip(uavsar_means, uavsar_stds)):
            print(f"  Band {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(uavsar_means)):
            nan_count = uavsar_invalid_counts['nan'][i].item() if uavsar_invalid_counts else 0
            inf_count = uavsar_invalid_counts['inf'][i].item() if uavsar_invalid_counts else 0
            print(f"    Band {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No UAVSAR imagery found.")
    
    print("Point attribute statistics:")
    if point_attr_means is not None and point_attr_stds is not None:
        for i, (mean, std) in enumerate(zip(point_attr_means, point_attr_stds)):
            print(f"  Attribute {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(point_attr_means)):
            nan_count = point_attr_invalid_counts['nan'][i].item() if point_attr_invalid_counts else 0
            inf_count = point_attr_invalid_counts['inf'][i].item() if point_attr_invalid_counts else 0
            print(f"    Attribute {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No point attributes found.")
        

    print("Precomputing validation dataset...")
    precomputed_validation = precompute_dataset(
        validation_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox'
    )
    print("Precomputing test dataset...")
    precomputed_test = precompute_dataset(
        test_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox'
    )
    print("Precomputing training dataset...")
    precomputed_training = precompute_dataset(
        training_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox'
    )  

    # Save the precomputed datasets.
    torch.save(precomputed_training, 'data/processed/model_data/precomputed_training_tiles.pt')
    torch.save(precomputed_validation, 'data/processed/model_data/precomputed_validation_tiles.pt')
    torch.save(precomputed_test, 'data/processed/model_data/precomputed_test_tiles.pt')
    
    # Also save the normalization statistics for future use
    if naip_means is not None and naip_stds is not None:
        torch.save({
            'means': naip_means, 
            'stds': naip_stds, 
            'invalid_counts': naip_invalid_counts if naip_invalid_counts else None
        }, 'data/processed/model_data/naip_normalization_stats.pt')
    if uavsar_means is not None and uavsar_stds is not None:
        torch.save({
            'means': uavsar_means, 
            'stds': uavsar_stds,
            'invalid_counts': uavsar_invalid_counts if uavsar_invalid_counts else None
        }, 'data/processed/model_data/uavsar_normalization_stats.pt')
    if point_attr_means is not None and point_attr_stds is not None:
        torch.save({
            'means': point_attr_means, 
            'stds': point_attr_stds, 
            'invalid_counts': point_attr_invalid_counts if point_attr_invalid_counts else None
        }, 'data/processed/model_data/point_attr_normalization_stats.pt')
    
    print("Precomputation complete. Precomputed files saved.")

if __name__ == '__main__':
    main()