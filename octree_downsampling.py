import matplotlib.pyplot as plt
import numpy as np
import torch


def filter_invalid_shapes(data):
    """
    Filters and logs examples where the sparse or dense point clouds are invalid.

    Args:
        data (list): Dataset containing 'uav_points' (dense) and 'dep_points' (sparse).

    Returns:
        list: Indices of invalid examples.
    """
    invalid_indices = []

    print("Inspecting dataset for invalid shapes...")
    for i, item in enumerate(data):
        uav_points = item.get('uav_points')
        dep_points = item.get('dep_points')
        approx_dsm = item.get('dsm_1m')

        
        # Check if tensors are None or empty
        if uav_points is None or dep_points is None or uav_points.numel() == 0 or dep_points.numel() == 0:
            print(f"Example {i + 1}: Empty tensor detected.")
            invalid_indices.append(i)
            continue

        uav_points_shape = uav_points.shape
        dep_points_shape = dep_points.shape
        approx_dsm_shape = approx_dsm.shape
        # print(approx_dsm_shape[0], approx_dsm_shape[1], (approx_dsm_shape[0] != 10 or approx_dsm_shape[1] != 10 ))
        # Check for shape validity
        if (approx_dsm_shape[0] != 10 or approx_dsm_shape[1] != 10 ):
            print(approx_dsm_shape)
        # Check for shape validity
        is_invalid = (
            len(uav_points_shape) != 2 or uav_points_shape[1] != 3 or uav_points_shape[0] < 2000 or dep_points_shape[0]*1.5 > uav_points_shape[0] or
            len(dep_points_shape) != 2 or dep_points_shape[1] != 3 or dep_points_shape[0] < 50 or approx_dsm_shape[0] != 10 or approx_dsm_shape[1] != 10 
        )

        if is_invalid:
            print(f"Example {i + 1}: Invalid. UAV shape: {uav_points_shape} | 3DEP shape: {dep_points_shape} | dsm shape {approx_dsm_shape} ")
            # print(f"  UAV points shape: {uav_points_shape}")
            # print(f"  DEP points shape: {dep_points_shape}")
            invalid_indices.append(i)

    print(f"\nTotal invalid examples: {len(invalid_indices)}")
    # print(f"Invalid indices: {invalid_indices}")
    return invalid_indices

# # Identify and remove invalid examples
# invalid_indices = filter_invalid_shapes(training_data)
# filtered_data = [item for i, item in enumerate(training_data) if i not in invalid_indices]
# print(len(filtered_data))





def octree_downsample_indices(points, indices, max_depth, current_depth=0, bbox=None):
    """
    Recursively partitions the point cloud into octants and returns one representative
    (the one with the lowest z) per leaf.
      
    Parameters:
      points (np.ndarray): (N,3) array of point coordinates.
      indices (np.ndarray): (N,) array of the original indices corresponding to points.
      max_depth (int): maximum depth of recursion (controls how many subdivisions occur).
      current_depth (int): the current recursion depth.
      bbox (tuple): tuple (min_bound, max_bound) defining the bounding box for the current node.
    
    Returns:
      list: a list of selected indices (one per leaf).
    """
    if len(points) == 0:
        return []
    
    # Set the bounding box if not provided.
    if bbox is None:
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        bbox = (min_bound, max_bound)
        
    # Base case: maximum depth reached or only one point remains.
    if current_depth == max_depth or len(points) == 1:
        rep_index = indices[np.argmin(points[:, 2])]  # select the point with the lowest z
        return [rep_index]
    
    min_bound, max_bound = bbox
    mid = (min_bound + max_bound) / 2.0
    rep_indices = []
    
    # Loop over the 8 octants.
    for i in range(8):
        new_min = np.array(min_bound)
        new_max = np.array(max_bound)
        # Subdivide x dimension (bit 0).
        if i & 1:
            new_min[0] = mid[0]
        else:
            new_max[0] = mid[0]
        # Subdivide y dimension (bit 1).
        if i & 2:
            new_min[1] = mid[1]
        else:
            new_max[1] = mid[1]
        # Subdivide z dimension (bit 2).
        if i & 4:
            new_min[2] = mid[2]
        else:
            new_max[2] = mid[2]
        
        # Identify points within the current octant.
        in_octant = np.all((points >= new_min) & (points <= new_max), axis=1)
        if not np.any(in_octant):
            continue
        
        sub_points = points[in_octant]
        sub_indices = indices[in_octant]
        sub_bbox = (new_min, new_max)
        rep_indices.extend(octree_downsample_indices(sub_points, sub_indices, max_depth,
                                                       current_depth + 1, sub_bbox))
    return rep_indices

def downsample_uav_points_dict(data, max_depth):
    """
    Downsamples only the 'uav_points' (and associated uav_* keys) from the input dictionary
    using an octree‐based method with a fixed maximum depth.
    
    Parameters:
      data (dict): Dictionary containing keys such as:
                   - 'uav_points': tensor of shape (N_uav, 3)
                   - 'uav_intensity', 'uav_return_number', 'uav_num_returns': each of shape (N_uav,)
                   plus other keys that are left untouched.
      max_depth (int): Maximum recursion depth for the octree.
      
    Returns:
      dict: A new dictionary with the downsampled 'uav_points' and associated UAV fields.
    """
    new_data = data.copy()
    
    uav_points = data['uav_points']  # expected shape: (N_uav, 3)
    points_np = uav_points.cpu().numpy()
    indices = np.arange(uav_points.shape[0])
    
    rep_indices = octree_downsample_indices(points_np, indices, max_depth)
    rep_indices = np.array(rep_indices)
    rep_indices = np.sort(rep_indices)  # sort if desired

    # Downsample the UAV points.
    new_data['uav_points'] = uav_points[rep_indices]
    
    # For each UAV attribute, only apply the downsampling if its length matches uav_points.
    for key in ['uav_intensity', 'uav_return_number', 'uav_num_returns']:
        if key in data:
            if data[key].shape[0] == uav_points.shape[0]:
                new_data[key] = data[key][rep_indices]
            else:
                # If the attribute doesn't match the number of points,
                # you might log a warning or leave it unchanged.
                new_data[key] = data[key]
    
    return new_data


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensures 3D plotting is available.
import torch

def plot_uav_points(original_dict, downsampled_dict, elev=30, azim=45):
    """
    Visualize the UAV point cloud before and after downsampling in 3D.
    
    Parameters:
        original_dict (dict): Dictionary with the original UAV point cloud.
            Expected key 'uav_points' containing a torch.Tensor of shape (N, 3).
        downsampled_dict (dict): Dictionary with the downsampled UAV point cloud.
            Expected key 'uav_points' containing a torch.Tensor of shape (M, 3).
        elev (float): Elevation angle (in degrees) for the 3D view (default: 30).
        azim (float): Azimuth angle (in degrees) for the 3D view (default: 45).
    """
    # Convert the UAV points to NumPy arrays.
    orig = original_dict['uav_points'].cpu().numpy()  # Expected shape: (N, 3)
    down = downsampled_dict['uav_points'].cpu().numpy()  # Expected shape: (M, 3)
    
    # Create a figure with two 3D subplots side by side.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    
    # Plot original UAV points.
    sc0 = axs[0].scatter(orig[:, 0], orig[:, 1], orig[:, 2], 
                         c=orig[:, 2], cmap='viridis', s=2)
    axs[0].set_title("Original UAV Points (N = {})".format(orig.shape[0]))
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[0].view_init(elev=elev, azim=azim)
    fig.colorbar(sc0, ax=axs[0], label="Z")
    
    # Plot downsampled UAV points.
    sc1 = axs[1].scatter(down[:, 0], down[:, 1], down[:, 2],
                         c=down[:, 2], cmap='viridis', s=2)
    axs[1].set_title("Downsampled UAV Points (N = {})".format(down.shape[0]))
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[1].view_init(elev=elev, azim=azim)
    fig.colorbar(sc1, ax=axs[1], label="Z")
    
    plt.tight_layout()
    plt.show()




# i = 20
# downsample_training_data = downsample_uav_dicts(training_data[0:50], 40000)
# Visualize the before and after:
# plot_uav_points(training_data[i], downsample_training_data[i])

