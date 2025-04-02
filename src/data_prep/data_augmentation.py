import torch
import numpy as np
import random
import copy
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

def randomly_remove_points(tile, ratio=0.1):
    """
    Randomly remove a ratio of DEP points.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        ratio (float): Ratio of points to remove (0.0 to 1.0)
        
    Returns:
        dict: The modified tile with fewer DEP points
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Get DEP points
    dep_points = tile_copy['dep_points_norm']
    n_dep = dep_points.shape[0]
    
    # Calculate number of points to keep
    n_keep = int(n_dep * (1.0 - ratio))
    
    # Ensure we keep at least 100 points (or all points if there are fewer)
    n_keep = max(min(n_keep, n_dep), min(100, n_dep))
    
    if n_keep < n_dep:
        # Generate random indices to keep
        keep_indices = torch.randperm(n_dep)[:n_keep]
        
        # Keep only selected points
        tile_copy['dep_points_norm'] = dep_points[keep_indices]
        
        # Also update point attributes if they exist
        if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
            tile_copy['dep_points_attr_norm'] = tile_copy['dep_points_attr_norm'][keep_indices]
    
    # Note: KNN indices will be regenerated at the end of the augmentation pipeline
    
    return tile_copy

def add_nearby_points(tile, ratio=0.1, max_distance=0.02):
    """
    Add points within a specified small distance of existing DEP points.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        ratio (float): Ratio of new points to add relative to existing points
        max_distance (float): Maximum distance to place new points
        
    Returns:
        dict: The modified tile with additional DEP points
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Get existing DEP points
    dep_points = tile_copy['dep_points_norm']
    n_dep = dep_points.shape[0]
    
    # Calculate number of points to add
    n_add = int(n_dep * ratio)
    
    if n_add > 0:
        # Randomly select source points to duplicate from
        source_indices = torch.randint(0, n_dep, (n_add,))
        source_points = dep_points[source_indices]
        
        # Generate random offsets (in all directions)
        # Create random directions by generating points on a unit sphere
        random_directions = torch.randn(n_add, 3)
        # Fix 1: Add epsilon to prevent division by zero
        random_directions = random_directions / (random_directions.norm(dim=1, keepdim=True) + 1e-8)
        
        # Generate random distances (from 0 to max_distance)
        random_distances = torch.rand(n_add, 1) * max_distance
        
        # Compute offset vectors
        offsets = random_directions * random_distances
        
        # Create new points
        new_points = source_points + offsets
        
        # Concatenate with existing points
        tile_copy['dep_points_norm'] = torch.cat([dep_points, new_points], dim=0)
        
        # If point attributes exist, duplicate them for the new points
        if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
            source_attrs = tile_copy['dep_points_attr_norm'][source_indices]
            tile_copy['dep_points_attr_norm'] = torch.cat([tile_copy['dep_points_attr_norm'], source_attrs], dim=0)
    
    # Note: KNN indices will be regenerated at the end of the augmentation pipeline
    
    return tile_copy

def mask_points(tile, min_radius=0.05, max_radius=0.2, n_masks=1, min_removal_ratio=0.7, max_removal_ratio=1.0):
    """
    Remove a portion of points within circular regions (creating "holes" in the point cloud).
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        min_radius (float): Minimum radius for the masked area
        max_radius (float): Maximum radius for the masked area
        n_masks (int): Number of masked areas to create
        min_removal_ratio (float): Minimum ratio of points to remove within each mask (0-1)
        max_removal_ratio (float): Maximum ratio of points to remove within each mask (0-1)
        
    Returns:
        dict: The modified tile with masked areas
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Get DEP points
    dep_points = tile_copy['dep_points_norm']
    n_dep = dep_points.shape[0]
    
    # Create a mask (True for points to keep, initially all True)
    keep_mask = torch.ones(n_dep, dtype=torch.bool)
    
    for _ in range(n_masks):
        # Randomly select a center point
        center_idx = torch.randint(0, n_dep, (1,)).item()
        center_point = dep_points[center_idx, :2]  # Only use x,y coordinates
        
        # Generate a random radius
        radius = torch.FloatTensor(1).uniform_(min_radius, max_radius).item()
        
        # Calculate distances (in x,y plane only) from all points to the center
        distances = torch.norm(dep_points[:, :2] - center_point, dim=1)
        
        # Identify points within the radius
        mask_indices = torch.where(distances < radius)[0]
        n_mask_points = len(mask_indices)
        
        if n_mask_points > 0:
            # Determine the removal ratio randomly
            removal_ratio = torch.FloatTensor(1).uniform_(min_removal_ratio, max_removal_ratio).item()
            
            # Calculate how many points to remove
            n_remove = int(n_mask_points * removal_ratio)
            
            if n_remove > 0:
                # Randomly select points to remove from the masked area
                remove_indices = mask_indices[torch.randperm(n_mask_points)[:n_remove]]
                
                # Create a temporary mask for this circular region
                temp_mask = torch.ones(n_dep, dtype=torch.bool)
                temp_mask[remove_indices] = False
                
                # Combine with the overall keep mask
                keep_mask = keep_mask & temp_mask
    
    # Ensure we keep at least 100 points (or all points if there are fewer)
    if keep_mask.sum() < min(100, n_dep):
        # If too many points would be removed, just randomly select points to keep
        n_keep = min(100, n_dep)
        keep_indices = torch.randperm(n_dep)[:n_keep]
        new_keep_mask = torch.zeros(n_dep, dtype=torch.bool)
        new_keep_mask[keep_indices] = True
        keep_mask = new_keep_mask
    
    # Keep only non-masked points
    tile_copy['dep_points_norm'] = dep_points[keep_mask]
    
    # Also update point attributes if they exist
    if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
        tile_copy['dep_points_attr_norm'] = tile_copy['dep_points_attr_norm'][keep_mask]
    
    return tile_copy





def shift_temporal_sequence(tile, max_shift_days=30):
    """
    Add random temporal shift to relative dates for both NAIP and UAVSAR imagery.
    
    Parameters:
        tile (dict): The tile dictionary containing imagery data
        max_shift_days (float): Maximum number of days to shift the dates by
        
    Returns:
        dict: The modified tile with shifted temporal sequences
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Add random temporal shift to relative dates for NAIP
    if tile_copy.get('naip') is not None and 'relative_dates' in tile_copy['naip']:
        shift = torch.tensor([[random.uniform(-max_shift_days, max_shift_days)]])
        tile_copy['naip']['relative_dates'] = tile_copy['naip']['relative_dates'] + shift
    
    # Add random temporal shift to relative dates for UAVSAR
    if tile_copy.get('uavsar') is not None and 'relative_dates' in tile_copy['uavsar']:
        shift = torch.tensor([[random.uniform(-max_shift_days, max_shift_days)]])
        tile_copy['uavsar']['relative_dates'] = tile_copy['uavsar']['relative_dates'] + shift
    
    return tile_copy


def augment_attributes(tile, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    """
    Apply scaling and shifting to point attributes while preserving discrete values.
    
    Parameters:
        tile (dict): The tile dictionary containing point attributes
        scale_range (tuple): Range for random scaling factors
        shift_range (tuple): Range for random shift values
        
    Returns:
        dict: The modified tile with augmented point attributes
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
        # Generate random scales and shifts for each attribute
        n_attr = tile_copy['dep_points_attr_norm'].shape[1]
        scales = torch.FloatTensor(n_attr).uniform_(*scale_range)
        shifts = torch.FloatTensor(n_attr).uniform_(*shift_range)
        
        # Apply scaling and shifting
        for i in range(n_attr):
            # Save original values for discrete attributes like return numbers
            if i in [1, 2]:  # ReturnNumber and NumberOfReturns
                original_values = tile_copy['dep_points_attr_norm'][:, i].clone()
            
            # Apply transformation
            tile_copy['dep_points_attr_norm'][:, i] = tile_copy['dep_points_attr_norm'][:, i] * scales[i] + shifts[i]
            
            # Restore discrete attributes to valid values
            if i in [1, 2]:
                tile_copy['dep_points_attr_norm'][:, i] = original_values
        
        # Ensure values remain in valid range (-1 to 1 for normalized data)
        tile_copy['dep_points_attr_norm'] = torch.clamp(tile_copy['dep_points_attr_norm'], -1, 1)
    
    return tile_copy


def rotate_tile(tile, angle_degrees=None):
    """
    Rotate point clouds and imagery by a specified angle.
    For simplicity with imagery, restricts rotations to 90, 180, or 270 degrees.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds and imagery
        angle_degrees (float, optional): Rotation angle in degrees. If None, randomly selects 90, 180, or 270.
        
    Returns:
        dict: The modified tile with rotated point clouds and imagery
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # If no angle specified, randomly choose 90, 180, or 270 degrees
    if angle_degrees is None:
        angle_degrees = random.choice([90, 180, 270])
    
    # Convert angle to radians
    angle = np.radians(angle_degrees)
    
    # Determine the dtype of the point clouds to match
    if 'dep_points_norm' in tile_copy and isinstance(tile_copy['dep_points_norm'], torch.Tensor):
        # Use the dtype of the existing point cloud
        target_dtype = tile_copy['dep_points_norm'].dtype
    else:
        # Default to float32 if we can't determine
        target_dtype = torch.float32
    
    # Create rotation matrix with matching dtype
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=target_dtype)
    
    # Rotate point clouds
    if 'dep_points_norm' in tile_copy and isinstance(tile_copy['dep_points_norm'], torch.Tensor):
        tile_copy['dep_points_norm'] = torch.matmul(tile_copy['dep_points_norm'], rotation_matrix)
    
    if 'uav_points_norm' in tile_copy and isinstance(tile_copy['uav_points_norm'], torch.Tensor):
        # Make sure we're using the same dtype for this point cloud as well
        if tile_copy['uav_points_norm'].dtype != target_dtype:
            # Fix for different dtypes between dep_points_norm and uav_points_norm
            uav_rotation_matrix = rotation_matrix.to(tile_copy['uav_points_norm'].dtype)
            tile_copy['uav_points_norm'] = torch.matmul(tile_copy['uav_points_norm'], uav_rotation_matrix)
        else:
            tile_copy['uav_points_norm'] = torch.matmul(tile_copy['uav_points_norm'], rotation_matrix)
    
    # Rotate NAIP imagery (unchanged from original)
    if tile_copy.get('naip') is not None and 'images' in tile_copy['naip']:
        n_images = tile_copy['naip']['images'].shape[0]
        rotated_images = []
        
        for i in range(n_images):
            img = tile_copy['naip']['images'][i]
            
            if angle_degrees == 90:
                rotated = img.permute(0, 2, 1).flip(dims=[2])
            elif angle_degrees == 180:
                rotated = img.flip(dims=[1, 2])
            elif angle_degrees == 270:
                rotated = img.permute(0, 2, 1).flip(dims=[1])
            else:
                # For non-standard angles, keep original (should not happen)
                rotated = img
                
            rotated_images.append(rotated)
            
        if rotated_images:
            tile_copy['naip']['images'] = torch.stack(rotated_images)
    
    # Rotate UAVSAR imagery (same approach)
    if tile_copy.get('uavsar') is not None and 'images' in tile_copy['uavsar']:
        n_images = tile_copy['uavsar']['images'].shape[0]
        rotated_images = []
        
        for i in range(n_images):
            img = tile_copy['uavsar']['images'][i]
            
            if angle_degrees == 90:
                rotated = img.permute(0, 2, 1).flip(dims=[2])
            elif angle_degrees == 180:
                rotated = img.flip(dims=[1, 2])
            elif angle_degrees == 270:
                rotated = img.permute(0, 2, 1).flip(dims=[1])
            else:
                # For non-standard angles, keep original (should not happen)
                rotated = img
                
            rotated_images.append(rotated)
            
        if rotated_images:
            tile_copy['uavsar']['images'] = torch.stack(rotated_images)
    
    # Note: KNN indices will be regenerated at the end of the augmentation pipeline
    
    return tile_copy


def reflect_tile(tile, axis='x'):
    """
    Reflect point clouds and imagery across a specified axis.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds and imagery
        axis (str): Axis to reflect across ('x', 'y', or 'both')
        
    Returns:
        dict: The modified tile with reflected point clouds and imagery
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Determine the dtype of the point clouds to match
    if 'dep_points_norm' in tile_copy and isinstance(tile_copy['dep_points_norm'], torch.Tensor):
        # Use the dtype of the existing point cloud
        target_dtype = tile_copy['dep_points_norm'].dtype
    else:
        # Default to float32 if we can't determine
        target_dtype = torch.float32
    
    # Create reflection matrix based on axis with matching dtype
    if axis == 'x':
        reflection_matrix = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=target_dtype)
        flip_dims = [2]  # For imagery, flip along width (dim 2)
    elif axis == 'y':
        reflection_matrix = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=target_dtype)
        flip_dims = [1]  # For imagery, flip along height (dim 1)
    elif axis == 'both':
        reflection_matrix = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=target_dtype)
        flip_dims = [1, 2]  # For imagery, flip along both dimensions
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    # Reflect point clouds
    if 'dep_points_norm' in tile_copy and isinstance(tile_copy['dep_points_norm'], torch.Tensor):
        tile_copy['dep_points_norm'] = torch.matmul(tile_copy['dep_points_norm'], reflection_matrix)
    
    if 'uav_points_norm' in tile_copy and isinstance(tile_copy['uav_points_norm'], torch.Tensor):
        # Handle potential different dtype between point clouds
        if tile_copy['uav_points_norm'].dtype != target_dtype:
            uav_reflection_matrix = reflection_matrix.to(tile_copy['uav_points_norm'].dtype)
            tile_copy['uav_points_norm'] = torch.matmul(tile_copy['uav_points_norm'], uav_reflection_matrix)
        else:
            tile_copy['uav_points_norm'] = torch.matmul(tile_copy['uav_points_norm'], reflection_matrix)
    
    # Reflect NAIP imagery
    if tile_copy.get('naip') is not None and 'images' in tile_copy['naip']:
        n_images = tile_copy['naip']['images'].shape[0]
        for i in range(n_images):
            tile_copy['naip']['images'][i] = tile_copy['naip']['images'][i].flip(dims=flip_dims)
    
    # Reflect UAVSAR imagery
    if tile_copy.get('uavsar') is not None and 'images' in tile_copy['uavsar']:
        n_images = tile_copy['uavsar']['images'].shape[0]
        for i in range(n_images):
            tile_copy['uavsar']['images'][i] = tile_copy['uavsar']['images'][i].flip(dims=flip_dims)
    
    # Note: KNN indices will be regenerated at the end of the augmentation pipeline
    
    return tile_copy


# Add this utility function to help with other tensor creation operations
def create_tensor_with_matching_dtype(data, reference_tensor):
    """
    Create a tensor with the same dtype as the reference tensor.
    
    Parameters:
        data: The data to convert to a tensor
        reference_tensor: A tensor whose dtype we want to match
        
    Returns:
        A tensor with the same dtype as reference_tensor
    """
    if isinstance(reference_tensor, torch.Tensor):
        return torch.tensor(data, dtype=reference_tensor.dtype)
    else:
        return torch.tensor(data)  # Use default dtype


def jitter_points(tile, xy_scale=0.02, z_scale=0.01):
    """
    Add small random noise to DEP point coordinates only.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        xy_scale (float): Scale of noise for x and y coordinates
        z_scale (float): Scale of noise for z coordinate
        
    Returns:
        dict: The modified tile with jittered DEP points
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Get point count for DEP points only
    n_dep = tile_copy['dep_points_norm'].shape[0]
    
    # Get the dtype of the existing points
    target_dtype = tile_copy['dep_points_norm'].dtype
    
    # Generate noise (smaller for z dimension) with matching dtype
    dep_noise = torch.cat([
        torch.randn(n_dep, 2, dtype=target_dtype) * xy_scale,
        torch.randn(n_dep, 1, dtype=target_dtype) * z_scale
    ], dim=1)
    
    # Apply noise to DEP points only
    tile_copy['dep_points_norm'] = tile_copy['dep_points_norm'] + dep_noise
    
    # Note: KNN indices will be regenerated at the end of the augmentation pipeline
    
    return tile_copy


def augment_spectral_bands(tile, band_scale_range=(0.9, 1.1)):
    """
    Apply band-specific scaling to simulate different spectral responses.
    
    Parameters:
        tile (dict): The tile dictionary containing imagery
        band_scale_range (tuple): Range for random scaling factors
        
    Returns:
        dict: The modified tile with augmented spectral bands
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Augment NAIP spectral bands
    if tile_copy.get('naip') is not None and 'images' in tile_copy['naip']:
        n_images, n_bands, h, w = tile_copy['naip']['images'].shape
        
        for img_idx in range(n_images):
            # Generate random scaling factors for each band
            band_scales = torch.FloatTensor(n_bands).uniform_(*band_scale_range)
            
            # Apply band-specific scaling
            for band_idx in range(n_bands):
                tile_copy['naip']['images'][img_idx, band_idx] *= band_scales[band_idx]
    
    # Augment UAVSAR bands
    if tile_copy.get('uavsar') is not None and 'images' in tile_copy['uavsar']:
        n_images, n_bands, h, w = tile_copy['uavsar']['images'].shape
        
        for img_idx in range(n_images):
            # Generate random scaling factors for each band
            band_scales = torch.FloatTensor(n_bands).uniform_(*band_scale_range)
            
            # Apply band-specific scaling
            for band_idx in range(n_bands):
                tile_copy['uavsar']['images'][img_idx, band_idx] *= band_scales[band_idx]
    
    return tile_copy




def simulate_sensor_effects(tile, effect_strength=0.2, speckle_variance=0.1):
    """
    Simulate sensor effects for both NAIP and UAVSAR imagery.
    For NAIP: Creates gradient lighting effects simulating different sun angles.
    For UAVSAR: Adds multiplicative speckle noise.
    
    Parameters:
        tile (dict): The tile dictionary containing imagery
        effect_strength (float): Strength of the gradient effect for NAIP
        speckle_variance (float): Variance parameter for UAVSAR speckle noise
        
    Returns:
        dict: The modified tile with simulated sensor effects
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Simulate NAIP sensor effects
    if tile_copy.get('naip') is not None and 'images' in tile_copy['naip']:
        n_images, n_bands, h, w = tile_copy['naip']['images'].shape
        
        for img_idx in range(n_images):
            # Simulate different sun angles by adjusting brightness in a gradient
            grad_x = torch.linspace(-1, 1, w)
            grad_y = torch.linspace(-1, 1, h)
            xx, yy = torch.meshgrid(grad_x, grad_y, indexing='ij')
            
            # Random direction gradient
            angle = random.uniform(0, 2*np.pi)
            # Convert angle to tensor for PyTorch trig functions
            angle_tensor = torch.tensor(angle)
            gradient = xx*torch.cos(angle_tensor) + yy*torch.sin(angle_tensor)
            
            # Scale the gradient effect
            gradient = gradient * effect_strength + 1
            
            # Apply gradient differently to each band (simulating spectral variation with sun angle)
            for band_idx in range(n_bands):
                band_effect = 1.0 + (effect_strength * 0.5 * (band_idx / n_bands))
                # Multiply directly with the 2D band data
                tile_copy['naip']['images'][img_idx, band_idx] = tile_copy['naip']['images'][img_idx, band_idx] * gradient * band_effect
    
    # Simulate UAVSAR sensor effects (speckle noise)
    if tile_copy.get('uavsar') is not None and 'images' in tile_copy['uavsar']:
        n_images, n_bands, h, w = tile_copy['uavsar']['images'].shape
        
        for img_idx in range(n_images):
            
            # Multiplicative speckle noise (Gamma distribution)
            speckle_mean = 1.0
            # Add safety check for speckle variance
            speckle_var = max(speckle_variance, 1e-6)  # Prevent division by zero
            alpha = speckle_mean**2 / speckle_var
            beta = speckle_mean / speckle_var
            
            # Generate speckle noise using gamma distribution
            speckle = torch.tensor(
                np.random.gamma(alpha, 1/beta, (n_bands, h, w)), 
                dtype=tile_copy['uavsar']['images'].dtype
            )
            
            # Add clamping to prevent extreme values in the speckle noise
            speckle = torch.clamp(speckle, 0.1, 5.0)
            
            # Apply multiplicative noise
            tile_copy['uavsar']['images'][img_idx] *= speckle
            
            # Final clamp to ensure no extreme values after multiplication
            tile_copy['uavsar']['images'][img_idx] = torch.clamp(tile_copy['uavsar']['images'][img_idx], -500.0, 500.0)
    
    return tile_copy

def validate_augmented_tile(original_tile, augmented_tile, max_allowed_value=400.0):
    """
    Simplified validation that checks only essential properties of the augmented tile.
    
    Parameters:
        original_tile (dict): The original tile dictionary
        augmented_tile (dict): The augmented tile dictionary to validate
        max_allowed_value (float): Maximum absolute value allowed in tensors
        
    Returns:
        dict: The validated augmented tile or raises ValueError on validation failure
    """
    # 1. Check that important keys exist
    essential_keys = ['dep_points_norm']
    for key in essential_keys:
        if key in original_tile and key not in augmented_tile:
            raise ValueError(f"Augmented tile is missing essential key: {key}")
    
    # 2. Check for minimum number of points
    if 'dep_points_norm' in augmented_tile:
        point_count = augmented_tile['dep_points_norm'].shape[0]
        if point_count < 500:
            raise ValueError(f"Too few DEP points after augmentation: {point_count}")
        
        # 3. Quick check for NaN/Inf in point clouds
        points = augmented_tile['dep_points_norm']
        if torch.isnan(points).any():
            raise ValueError("NaN values found in dep_points_norm")
        if torch.isinf(points).any():
            raise ValueError("Inf values found in dep_points_norm")
        
        # 4. Check for extreme values in point clouds
        if torch.abs(points).max() > max_allowed_value:
            raise ValueError(f"Extreme values found in dep_points_norm: {torch.abs(points).max().item()}")
    
    # 5. Check that point attributes match point count if they exist
    if ('dep_points_norm' in augmented_tile and 
        'dep_points_attr_norm' in augmented_tile and 
        augmented_tile['dep_points_attr_norm'] is not None):
        if augmented_tile['dep_points_norm'].shape[0] != augmented_tile['dep_points_attr_norm'].shape[0]:
            raise ValueError(f"Point count mismatch: {augmented_tile['dep_points_norm'].shape[0]} points but {augmented_tile['dep_points_attr_norm'].shape[0]} attributes")
    
    # 6. Quick check for critical image data if it exists
    for img_type in ['naip', 'uavsar']:
        if (img_type in augmented_tile and 
            isinstance(augmented_tile[img_type], dict) and 
            'images' in augmented_tile[img_type]):
            images = augmented_tile[img_type]['images']
            # Only check for NaN/Inf - these should never occur in valid images
            if torch.isnan(images).any():
                raise ValueError(f"NaN values found in {img_type} images")
            if torch.isinf(images).any():
                raise ValueError(f"Inf values found in {img_type} images")
    
    return augmented_tile


def remove_horizontal_slice(tile, min_slice_height=0.05, max_slice_height=0.2, max_slice_position=0.5, min_removal_ratio=0.7, max_removal_ratio=1.0):
    """
    Remove a portion of points within a horizontal slice of the point cloud.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        min_slice_height (float): Minimum height of the slice to consider (in normalized units)
        max_slice_height (float): Maximum height of the slice to consider (in normalized units)
        max_slice_position (float): Maximum relative position for the slice (0-1, where 0 is bottom, 1 is top)
        min_removal_ratio (float): Minimum ratio of points to remove within the slice (0-1)
        max_removal_ratio (float): Maximum ratio of points to remove within the slice (0-1)
        
    Returns:
        dict: The modified tile with points removed from the horizontal slice
    """
    # Create a deep copy to avoid modifying the original
    tile_copy = copy.deepcopy(tile)
    
    # Get DEP points
    dep_points = tile_copy['dep_points_norm']
    n_dep = dep_points.shape[0]
    
    # Get z coordinates
    z_coords = dep_points[:, 2]
    
    # Determine the min and max z values
    min_z = 0  # As specified, bottom is always 0
    max_z = z_coords.max().item()
    z_range = max_z - min_z
    
    # Determine the maximum possible slice height based on max_slice_position
    max_possible_slice_height = z_range * max_slice_position
    
    # Adjust max_slice_height if needed to ensure it doesn't exceed the available space
    adjusted_max_slice_height = min(max_slice_height, max_possible_slice_height)
    
    # Ensure min_slice_height is also within bounds
    adjusted_min_slice_height = min(min_slice_height, adjusted_max_slice_height)
    
    # Determine the slice height (random within the adjusted range)
    slice_height = torch.FloatTensor(1).uniform_(adjusted_min_slice_height, adjusted_max_slice_height).item()
    
    # Determine the bottom z position of the slice
    max_bottom_position = min_z + (z_range * max_slice_position) - slice_height
    
    # Make sure we don't get a negative bottom position (should not happen with the adjustments above)
    max_bottom_position = max(0, max_bottom_position)
    
    # Randomly determine the slice bottom
    slice_bottom = torch.FloatTensor(1).uniform_(min_z, max_bottom_position).item()
    
    # Calculate slice top
    slice_top = slice_bottom + slice_height
    
    # Find points that are within the slice
    slice_mask = (z_coords >= slice_bottom) & (z_coords <= slice_top)
    slice_indices = torch.where(slice_mask)[0]
    n_slice_points = len(slice_indices)
    
    # Create a keep mask (initially all True)
    keep_mask = torch.ones(n_dep, dtype=torch.bool)
    
    # If there are points in the slice, remove a random ratio of them
    if n_slice_points > 0:
        # Determine the removal ratio randomly
        removal_ratio = torch.FloatTensor(1).uniform_(min_removal_ratio, max_removal_ratio).item()
        
        # Calculate how many points to remove
        n_remove = int(n_slice_points * removal_ratio)
        
        if n_remove > 0:
            # Randomly select points to remove from the slice
            remove_indices = slice_indices[torch.randperm(n_slice_points)[:n_remove]]
            keep_mask[remove_indices] = False
    
    # Ensure we keep at least 100 points (or all points if there are fewer)
    if keep_mask.sum() < min(100, n_dep):
        # If too many points would be removed, just randomly select points to keep
        n_keep = min(100, n_dep)
        keep_indices = torch.randperm(n_dep)[:n_keep]
        new_keep_mask = torch.zeros(n_dep, dtype=torch.bool)
        new_keep_mask[keep_indices] = True
        keep_mask = new_keep_mask
    
    # Keep only non-masked points
    tile_copy['dep_points_norm'] = dep_points[keep_mask]
    
    # Also update point attributes if they exist
    if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
        tile_copy['dep_points_attr_norm'] = tile_copy['dep_points_attr_norm'][keep_mask]
    
    return tile_copy


def get_tensor_dtypes(obj, path=""):
    """
    Recursively extract the dtypes of all tensors in a nested dictionary/list structure.
    
    Parameters:
        obj: The object to extract dtypes from (dict, list, tensor, or other)
        path: Current path in the nested structure (for tracking)
        
    Returns:
        dict: Dictionary mapping paths to dtypes
    """
    dtypes = {}
    
    if isinstance(obj, torch.Tensor):
        dtypes[path] = obj.dtype
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            dtypes.update(get_tensor_dtypes(value, new_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            dtypes.update(get_tensor_dtypes(item, new_path))
            
    return dtypes

def get_tensor_by_path(obj, path):
    """
    Get a tensor from a nested structure using the path.
    
    Parameters:
        obj: The object to navigate (dict, list, etc.)
        path: The path to the tensor
        
    Returns:
        The tensor at the specified path
    """
    if not path:
        return obj
    
    if "." in path:
        key, rest = path.split(".", 1)
        return get_tensor_by_path(obj[key], rest)
    elif "[" in path:
        key, index = path.split("[", 1)
        index = int(index.rstrip("]"))
        return get_tensor_by_path(obj[key][index], "")
    else:
        return obj[path]

def set_tensor_by_path(obj, path, tensor):
    """
    Set a tensor in a nested structure using the path.
    
    Parameters:
        obj: The object to navigate and modify (dict, list, etc.)
        path: The path to the tensor
        tensor: The tensor to set
    """
    if "." in path:
        key, rest = path.split(".", 1)
        set_tensor_by_path(obj[key], rest, tensor)
    elif "[" in path:
        key, index = path.split("[", 1)
        index = int(index.rstrip("]"))
        obj[key][index] = tensor
    else:
        obj[path] = tensor

def ensure_dtypes_match(obj, original_dtypes):
    """
    Ensure all tensors in obj have the same dtype as in original_dtypes.
    
    Parameters:
        obj: The object to check and modify (dict, list, tensor, etc.)
        original_dtypes: Dict mapping paths to original dtypes
        
    Returns:
        int: Number of tensors that needed correction
    """
    corrections = 0
    
    for path, dtype in original_dtypes.items():
        try:
            tensor = get_tensor_by_path(obj, path)
            if isinstance(tensor, torch.Tensor) and tensor.dtype != dtype:
                # Convert the tensor to the original dtype
                fixed_tensor = tensor.to(dtype)
                set_tensor_by_path(obj, path, fixed_tensor)
                corrections += 1
        except (KeyError, IndexError):
            # This can happen if the structure changed during augmentation
            # (e.g., some points were removed)
            continue
            
    return corrections


def get_dynamic_probabilities(tile, base_config):
    """
    Adjusts point modification probabilities based on the number of points in the point cloud.
    
    Parameters:
        tile (dict): The tile dictionary containing point clouds
        base_config (dict): Base configuration with default probabilities
        
    Returns:
        dict: Modified configuration with adjusted probabilities
    """
    # Create a copy of the base configuration
    config = copy.deepcopy(base_config)
    
    # Get the number of points in the point cloud
    if 'dep_points_norm' in tile and isinstance(tile['dep_points_norm'], torch.Tensor):
        num_points = tile['dep_points_norm'].shape[0]
        
        # Define thresholds for point counts
        low_threshold = 2500   # Below this is considered "few points"
        high_threshold = 6000  # Above this is considered "many points"
        
        # Calculate scaling factor between 0.1 (few points) and 1.0 (many points)
        scale = max(0.1, min(1.0, (num_points - low_threshold) / (high_threshold - low_threshold)))
        
        # Adjust probabilities for point-removing operations
        config['remove_points_probability'] *= scale
        config['mask_points_probability'] *= scale
        config['remove_horizontal_slice_probability'] *= scale
            
        # Also adjust the removal ratios for operations that remove points
        if 'remove_points_ratio' in config:
            config['remove_points_ratio'] *= scale
        
        if num_points < low_threshold:
            # For very small point clouds, adjust mask parameters
            if 'mask_min_removal_ratio' in config:
                config['mask_min_removal_ratio'] *= scale
            if 'mask_max_removal_ratio' in config:
                config['mask_max_removal_ratio'] *= scale
            if 'horizontal_slice_min_removal_ratio' in config:
                config['horizontal_slice_min_removal_ratio'] *= scale
            if 'horizontal_slice_max_removal_ratio' in config:
                config['horizontal_slice_max_removal_ratio'] *= scale
    
    return config



def randomly_augment_tile(tile, config=None):
    """
    Apply multiple augmentation techniques randomly based on provided probabilities.
    More efficient version with reduced copying and checking.
    
    Parameters:
        tile (dict): The tile dictionary to augment
        config (dict): Configuration with probabilities for each technique
        
    Returns:
        dict: The augmented tile
    """
    # Default configuration if none provided
    if config is None:
        config = {
            # Basic transformations
            'rotate_probability': 1,
            'reflect_probability': 0.5,
            'jitter_probability': 0.3,
            
            # Point cloud modifications
            'add_points_probability': 0.2,
            'remove_points_probability': 0.5,
            'mask_points_probability': 0.5,
            'remove_horizontal_slice_probability': 0.5,
            
            # Imagery and attributes
            'temporal_shift_probability': 0.4,
            'attribute_augment_probability': 0.4,
            'spectral_band_probability': 0.3,
            'sensor_effects_probability': 0.3,
            
            # Parameters (unchanged from original)
            'max_shift_days': 30,
            'jitter_xy_scale': 0.02,
            'jitter_z_scale': 0.01,
            'attribute_scale_range': (0.9, 1.1),
            'attribute_shift_range': (-0.1, 0.1),
            'band_scale_range': (0.9, 1.1),
            'add_points_ratio': 0.1,
            'add_points_max_distance': 0.02,
            'remove_points_ratio': 0.1,
            'mask_min_radius': 0.05,
            'mask_max_radius': 0.2,
            'mask_count': 1,
            'mask_min_removal_ratio': 0.7,
            'mask_max_removal_ratio': 1.0,
            'sensor_effect_strength': 0.2,
            'uavsar_noise_variance': 0.1,
            'horizontal_slice_min_height': 0.05,
            'horizontal_slice_max_height': 0.2,
            'horizontal_slice_max_position': 0.5,
            'horizontal_slice_min_removal_ratio': 0.7,
            'horizontal_slice_max_removal_ratio': 1.0
        }
    
    # Make a deep copy of the tile to avoid modifying original
    # This is the only deep copy we'll make
    augmented_tile = copy.deepcopy(tile)
    
    # Store original dtypes for key tensors only
    original_dtypes = {}
    if 'dep_points_norm' in augmented_tile and isinstance(augmented_tile['dep_points_norm'], torch.Tensor):
        original_dtypes['dep_points_norm'] = augmented_tile['dep_points_norm'].dtype
    if 'dep_points_attr_norm' in augmented_tile and isinstance(augmented_tile['dep_points_attr_norm'], torch.Tensor):
        original_dtypes['dep_points_attr_norm'] = augmented_tile['dep_points_attr_norm'].dtype
    
    # Adjust point modification probabilities based on point count
    if 'dep_points_norm' in augmented_tile and isinstance(augmented_tile['dep_points_norm'], torch.Tensor):
        num_points = augmented_tile['dep_points_norm'].shape[0]
        small_point_cloud = 6000
        if num_points < small_point_cloud:
            # Reduce probabilities for small point clouds
            scale = max(0.1, num_points / small_point_cloud)
            config = config.copy()  # Shallow copy is sufficient
            config['remove_points_probability'] *= scale
            config['mask_points_probability'] *= scale
            config['remove_horizontal_slice_probability'] *= scale
    
    try:
        # Apply augmentation techniques directly to augmented_tile
        # Each function gets augmented_tile and returns the modified augmented_tile
        
        # 1. Rotate points and imagery
        if random.random() < config['rotate_probability']:
            augmented_tile = rotate_tile(augmented_tile)
        
        # 2. Reflect points and imagery
        if random.random() < config['reflect_probability']:
            axis = random.choice(['x', 'y', 'both'])
            augmented_tile = reflect_tile(augmented_tile, axis=axis)
        
        # 3. Jitter DEP point coordinates
        if random.random() < config['jitter_probability']:
            augmented_tile = jitter_points(
                augmented_tile, 
                xy_scale=config['jitter_xy_scale'], 
                z_scale=config['jitter_z_scale']
            )
        
        # 4. Add nearby points to DEP point cloud
        if random.random() < config['add_points_probability']:
            augmented_tile = add_nearby_points(
                augmented_tile,
                ratio=config['add_points_ratio'],
                max_distance=config['add_points_max_distance']
            )
        
        # 5. Randomly remove DEP points
        if random.random() < config['remove_points_probability']:
            augmented_tile = randomly_remove_points(
                augmented_tile,
                ratio=config['remove_points_ratio']
            )
        
        # 6. Mask areas in DEP point cloud
        if random.random() < config['mask_points_probability']:
            augmented_tile = mask_points(
                augmented_tile,
                min_radius=config['mask_min_radius'],
                max_radius=config['mask_max_radius'],
                n_masks=config['mask_count'],
                min_removal_ratio=config['mask_min_removal_ratio'],
                max_removal_ratio=config['mask_max_removal_ratio']
            )
            
        # 7. Remove horizontal slice
        if random.random() < config['remove_horizontal_slice_probability']:
            augmented_tile = remove_horizontal_slice(
                augmented_tile,
                min_slice_height=config['horizontal_slice_min_height'],
                max_slice_height=config['horizontal_slice_max_height'],
                max_slice_position=config['horizontal_slice_max_position'],
                min_removal_ratio=config['horizontal_slice_min_removal_ratio'],
                max_removal_ratio=config['horizontal_slice_max_removal_ratio']
            )
        
        # 8. Shift temporal sequence
        if random.random() < config['temporal_shift_probability']:
            augmented_tile = shift_temporal_sequence(
                augmented_tile, 
                max_shift_days=config['max_shift_days']
            )
        
        # 9. Augment point attributes
        if random.random() < config['attribute_augment_probability']:
            augmented_tile = augment_attributes(
                augmented_tile,
                scale_range=config['attribute_scale_range'],
                shift_range=config['attribute_shift_range']
            )
        
        # 10. Augment spectral bands
        if random.random() < config['spectral_band_probability']:
            augmented_tile = augment_spectral_bands(
                augmented_tile,
                band_scale_range=config['band_scale_range']
            )
        
        # 11. Simulate sensor effects for both NAIP and UAVSAR
        if random.random() < config['sensor_effects_probability']:
            augmented_tile = simulate_sensor_effects(
                augmented_tile,
                effect_strength=config['sensor_effect_strength'],
                speckle_variance=config['uavsar_noise_variance']
            )
        
        # Regenerate KNN edge indices only once, after all transformations
        if 'knn_edge_indices' in augmented_tile:
            # Simply regenerate the indices for each k value
            for k in augmented_tile['knn_edge_indices']:
                edge_index_k = knn_graph(augmented_tile['dep_points_norm'], k=k, loop=False)
                edge_index_k = to_undirected(edge_index_k, num_nodes=augmented_tile['dep_points_norm'].size(0))
                augmented_tile['knn_edge_indices'][k] = edge_index_k
                
        return augmented_tile
        
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")
        return copy.deepcopy(tile)  # Return a copy of the original tile as fallback





# Additional function to check and report data type inconsistencies
def check_dtype_consistency(original_tile, augmented_tile):
    """
    Check for any data type inconsistencies between original and augmented tiles.
    
    Parameters:
        original_tile (dict): The original tile dictionary
        augmented_tile (dict): The augmented tile dictionary
        
    Returns:
        list: List of inconsistencies found
    """
    inconsistencies = []
    
    def check_tensor_dtype(orig, aug, path=""):
        if isinstance(orig, torch.Tensor) and isinstance(aug, torch.Tensor):
            if orig.dtype != aug.dtype:
                inconsistencies.append({
                    'path': path,
                    'original_dtype': str(orig.dtype),
                    'augmented_dtype': str(aug.dtype)
                })
        elif isinstance(orig, dict) and isinstance(aug, dict):
            # Check common keys
            for key in set(orig.keys()) & set(aug.keys()):
                new_path = f"{path}.{key}" if path else key
                check_tensor_dtype(orig[key], aug[key], new_path)
        elif isinstance(orig, list) and isinstance(aug, list):
            # Check elements of lists with matching indices
            for i in range(min(len(orig), len(aug))):
                new_path = f"{path}[{i}]"
                check_tensor_dtype(orig[i], aug[i], new_path)
    
    check_tensor_dtype(original_tile, augmented_tile)
    return inconsistencies




def augment_dataset(tiles, n_augmentations=1, config=None, sample_size=None, 
                prob_vector=None, total_augmentations=None):
    """
    Create augmented versions of tiles in a dataset with flexible sampling strategy.
    Will continue sampling until the desired number of augmentations is met,
    with a cap of 500 resampling attempts for validation failures.
    
    Parameters:
        tiles (list): List of tile dictionaries
        n_augmentations (int): Number of augmented versions to create for each selected tile
                              (used when total_augmentations is None)
        config (dict): Configuration with probabilities for each technique
        sample_size (int, optional): If provided, randomly sample this many tiles for augmentation.
                                   If None, use all tiles.
        prob_vector (list, optional): Vector of probabilities for selecting each tile.
                                    Must be same length as tiles or sample_size.
                                    If None, uniform probabilities are used.
        total_augmentations (int, optional): If provided, create exactly this many augmentations
                                           by sampling from tiles with replacement.
                                           When specified, overrides n_augmentations.
        
    Returns:
        list: List of augmented tiles (does not include original tiles)
    """
    import random
    import numpy as np
    
    # Randomly sample tiles if sample_size is specified
    if sample_size is not None and sample_size < len(tiles):
        print(f"Randomly sampling {sample_size} tiles from {len(tiles)} total tiles")
        sampled_tiles = random.sample(tiles, sample_size)
    else:
        sampled_tiles = tiles
        print(f"Using all {len(tiles)} tiles as candidates for augmentation")
    
    # Set up probability vector for sampling if specified
    if prob_vector is not None:
        if len(prob_vector) != len(sampled_tiles):
            raise ValueError(f"Probability vector length ({len(prob_vector)}) must match number of sampled tiles ({len(sampled_tiles)})")
        prob_vector = np.array(prob_vector) / np.sum(prob_vector)
    
    augmented_tiles = []
    augmentation_failures = 0
    augmentation_stats = {"total_attempted": 0, "successful": 0, "failures": 0}
    
    # Determine augmentation strategy based on parameters
    if total_augmentations is not None:
        # Strategy: Sample with replacement until we reach total_augmentations
        print(f"Creating {total_augmentations} augmentations by sampling tiles with replacement")
        
        successful_augmentations = 0
        resample_attempts = 0  # Track attempts to resample due to validation failures
        max_resample_attempts = 500  # Cap on resample attempts
        
        while successful_augmentations < total_augmentations and resample_attempts < max_resample_attempts:
            # Sample a tile based on probability vector
            if prob_vector is not None:
                tile_idx = np.random.choice(len(sampled_tiles), p=prob_vector)
            else:
                tile_idx = random.randrange(len(sampled_tiles))
                
            tile = sampled_tiles[tile_idx]
            augmentation_stats["total_attempted"] += 1
            
            # Apply random augmentations
            try:
                augmented_tile = randomly_augment_tile(tile, config)
                
                # Final validation before adding to result
                try:
                    validate_augmented_tile(tile, augmented_tile)
                    
                    # Update tile_id if present to indicate it's augmented
                    if 'tile_id' in augmented_tile:
                        augmented_tile['tile_id'] = f"{augmented_tile['tile_id']}_aug_{successful_augmentations+1}"
                    
                    augmented_tiles.append(augmented_tile)
                    augmentation_stats["successful"] += 1
                    successful_augmentations += 1
                    
                    # Print progress periodically
                    if successful_augmentations % 100 == 0:
                        success_rate = (augmentation_stats["successful"] / augmentation_stats["total_attempted"]) * 100
                        print(f"Generated {successful_augmentations}/{total_augmentations} augmentations, " +
                              f"success rate: {success_rate:.2f}%, resampling attempts: {resample_attempts}")
                    
                except ValueError as e:
                    print(f"Validation failed for tile {tile_idx}, attempt {augmentation_stats['total_attempted']}: {str(e)}")
                    augmentation_stats["failures"] += 1
                    augmentation_failures += 1
                    resample_attempts += 1
                    
            except Exception as e:
                print(f"Augmentation error for tile {tile_idx}, attempt {augmentation_stats['total_attempted']}: {str(e)}")
                augmentation_stats["failures"] += 1
                augmentation_failures += 1
                resample_attempts += 1
        
        # Check if we reached the cap on resampling attempts
        if resample_attempts >= max_resample_attempts:
            print(f"WARNING: Reached the maximum number of resampling attempts ({max_resample_attempts}).")
            print(f"Only generated {successful_augmentations}/{total_augmentations} requested augmentations.")
    
    else:
        # Original strategy: Process each sampled tile n_augmentations times
        # (keeping the original implementation for this part)
        for i, tile in enumerate(sampled_tiles):
            tile_failures = 0
            
            for j in range(n_augmentations):
                augmentation_stats["total_attempted"] += 1
                
                # Apply random augmentations
                try:
                    augmented_tile = randomly_augment_tile(tile, config)
                    
                    # Final validation before adding to result
                    try:
                        validate_augmented_tile(tile, augmented_tile)
                        
                        # Update tile_id if present to indicate it's augmented
                        if 'tile_id' in augmented_tile:
                            augmented_tile['tile_id'] = f"{augmented_tile['tile_id']}_aug_{j+1}"
                        
                        augmented_tiles.append(augmented_tile)
                        augmentation_stats["successful"] += 1
                        
                    except ValueError as e:
                        print(f"Validation failed for tile {i}, augmentation {j+1}: {str(e)}")
                        augmentation_stats["failures"] += 1
                        tile_failures += 1
                        
                except Exception as e:
                    print(f"Augmentation error for tile {i}, augmentation {j+1}: {str(e)}")
                    augmentation_stats["failures"] += 1
                    tile_failures += 1
            
            # Print detailed diagnostic if all augmentations of this tile failed
            if tile_failures == n_augmentations:
                print(f"WARNING: All {n_augmentations} augmentations failed for tile {i}")
                
                # Diagnostic information about the tile
                try:
                    print(f"Tile {i} information:")
                    for key, value in tile.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, " +
                                  f"contains_nan={torch.isnan(value).any().item()}, " +
                                  f"contains_inf={torch.isinf(value).any().item()}, " +
                                  f"min={value.min().item()}, max={value.max().item()}")
                        elif isinstance(value, dict):
                            print(f"  {key}: (dictionary with {len(value)} items)")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, torch.Tensor):
                                    print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}, " +
                                          f"contains_nan={torch.isnan(sub_value).any().item()}, " +
                                          f"contains_inf={torch.isinf(sub_value).any().item()}, " +
                                          f"min={sub_value.min().item()}, max={sub_value.max().item()}")
                except Exception as e:
                    print(f"Error during diagnostic: {str(e)}")
            
            augmentation_failures += tile_failures
            
            # Print progress periodically
            if (i+1) % 500 == 0:
                success_rate = (augmentation_stats["successful"] / augmentation_stats["total_attempted"]) * 100
                print(f"Processed {i+1}/{len(sampled_tiles)} tiles, success rate: {success_rate:.2f}%, failures: {augmentation_failures}")
    
    # Final stats
    success_rate = (augmentation_stats["successful"] / augmentation_stats["total_attempted"]) * 100
    print(f"Dataset augmentation complete:")
    if total_augmentations is not None:
        print(f"  Target augmentations: {total_augmentations}")
        print(f"  Successfully created: {len(augmented_tiles)}")
        if len(augmented_tiles) < total_augmentations:
            print(f"  Missing augmentations: {total_augmentations - len(augmented_tiles)}")
    else:
        print(f"  Original tiles used for augmentation: {len(sampled_tiles)}")
    print(f"  Successfully augmented tiles: {len(augmented_tiles)}")
    print(f"  Failed augmentations: {augmentation_failures}")
    print(f"  Overall success rate: {success_rate:.2f}%")
    
    return augmented_tiles




if __name__ == "__main__":
    # Load your dataset
    import torch
    print("Beginning data augmentation...  ")
    print("Loading training tiles...")
    training_tiles = torch.load('data/processed/model_data/precomputed_training_tiles.pt', weights_only=False)


    ### Calculate sampling probabilities based on standard deviations of point clouds ###
    # Calculate the standard deviation for each point cloud
    std_devs = []
    for tile in training_tiles:
        points = tile['uav_points_norm']  # (N, 3)
        std_dev = (torch.std(points[:, :2], dim=0).mean() +  torch.std(points[:, 2], dim=0)).item()  # Weight z std dev higher
        std_devs.append(std_dev)

    # Convert to a NumPy array for convenience
    std_devs = np.array(std_devs)
    # Apply an alternative to softmax: temperature-scaled softmax

    print("calculating sampling probabilities based on uav_points_norm standard deviations...")
    # Clamp the max std_devs to the 95th percentile
    percentile_95 = np.percentile(std_devs, 95)
    std_devs = np.clip(std_devs, None, percentile_95)

    temperature = 4  # Increase the temperature for a more gradual change
    scaled_probs = np.exp(std_devs / temperature) / np.sum(np.exp(std_devs / temperature))

    # Print the top 10 and bottom 10 values of scaled_probs
    print("Top 10 scaled probabilities:", np.sort(scaled_probs)[-10:])
    print("Bottom 10 scaled probabilities:", np.sort(scaled_probs)[:10])


    # Configure augmentation parameters
    config = {
        # Basic transformations
        'rotate_probability': 1,
        'reflect_probability': 0.5,
        'jitter_probability': 0.3,
        
        # Point cloud modifications
        'add_points_probability': 0.2,
        'remove_points_probability': 0.5,
        'mask_points_probability': 0.5,
        'remove_horizontal_slice_probability': 0.5,  
        
        # Imagery and attributes
        'temporal_shift_probability': 0.5,
        'attribute_augment_probability': 0.3,
        'spectral_band_probability': 0.3,
        'sensor_effects_probability': 0.3,
        
        # Parameters
        'max_shift_days': 60,
        'jitter_xy_scale': 0.02,
        'jitter_z_scale': 0.01,
        'attribute_scale_range': (0.9, 1.1),
        'attribute_shift_range': (-0.1, 0.1),
        'band_scale_range': (0.9, 1.1),
        'add_points_ratio': 0.1,
        'add_points_max_distance': 0.01,
        'remove_points_ratio': 0.1,
        'mask_min_radius': 0.10,
        'mask_max_radius': 0.75,
        'mask_min_removal_ratio': 0.2, 
        'mask_max_removal_ratio': 0.6,  
        'mask_count': 2,
        'sensor_effect_strength': 0.2,
        'uavsar_noise_variance': 0.1,
        'horizontal_slice_min_height': 0.3,  
        'horizontal_slice_max_height': 2,   
        'horizontal_slice_max_position': 0.5, #0.5 is middle of point cloud, 1 is top
        'horizontal_slice_min_removal_ratio': 0.3, 
        'horizontal_slice_max_removal_ratio': 0.7     }

    
    print("Augmenting dataset...")
    desired_total_tiles = 50000
    total_augmentations = desired_total_tiles - len(training_tiles) 
    augmented_tiles = augment_dataset(training_tiles, n_augmentations=1, config=config, prob_vector=scaled_probs, total_augmentations=total_augmentations)

    # augmented_training_tiles = training_tiles + augmented_tiles
    # Save the augmented dataset
    print("Saving augmented dataset...")
    torch.save(augmented_tiles, 'data/processed/model_data/augmented_tiles_25k.pt')