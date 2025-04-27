import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import random
from tqdm import tqdm
from matplotlib.patches import Rectangle
import math

# Configure matplotlib for better PDF output with rasterization
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better text rendering
mpl.rcParams['figure.dpi'] = 100   # Default figure DPI
mpl.rcParams['savefig.dpi'] = 150  # Default save DPI
mpl.rcParams['figure.autolayout'] = True  # Better layout
mpl.rcParams['path.simplify'] = True  # Simplify paths for better rendering
mpl.rcParams['path.simplify_threshold'] = 0.8  # Higher threshold for more simplification

def create_dimension_boxen_plots(dep_points, uav_points, pred_points, figsize=(15, 5), max_points=5000):
    """
    Create boxen plots for the x, y, and z dimensions of the point clouds.
    """
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Dimension labels
    dim_labels = ['X', 'Y', 'Z']
    
    # Sample points if the point clouds are large (for better performance)
    def sample_points(points, max_count):
        if len(points) > max_count:
            indices = np.random.choice(len(points), max_count, replace=False)
            return points[indices]
        return points
    
    dep_points_sampled = sample_points(dep_points, max_points)
    uav_points_sampled = sample_points(uav_points, max_points)
    pred_points_sampled = sample_points(pred_points, max_points)
    
    # Set up the data for each dimension
    for dim_idx, dim_label in enumerate(dim_labels):
        # Create dataframes for each point cloud type
        dep_df = pd.DataFrame({
            'Value': dep_points_sampled[:, dim_idx],
            'Source': 'DEP Input'
        })
        
        uav_df = pd.DataFrame({
            'Value': uav_points_sampled[:, dim_idx],
            'Source': 'UAV Ground Truth'
        })
        
        pred_df = pd.DataFrame({
            'Value': pred_points_sampled[:, dim_idx],
            'Source': 'Model Prediction'
        })
        
        # Combine dataframes
        df = pd.concat([dep_df, uav_df, pred_df])
        
        # Create boxen plot - Fixed to avoid warning
        sns.boxenplot(x='Source', y='Value', hue='Source', data=df, ax=axes[dim_idx], 
                      palette={'DEP Input': 'blue', 'UAV Ground Truth': 'green', 'Model Prediction': 'red'},
                      legend=False)
        axes[dim_idx].set_title(f'{dim_label} Dimension Distribution')
        axes[dim_idx].tick_params(axis='x', rotation=45)
        axes[dim_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules from val_eval.py
from val_eval import (
    load_model, RawDataset, evaluate_validation_samples, 
    multimodal_variable_size_collate, MultimodalModelConfig
)
from torch.utils.data import DataLoader




def plot_naip_imagery(naip_data, naip_norm_stats=None, bbox_overlay=None, figsize=(15, 4), title=None):
    """
    Create a single plot with subplots for each NAIP image, ordered by date.
    
    Parameters:
    -----------
    naip_data : dict
        NAIP dictionary containing:
        - 'images': Tensor [n_images, 4, h, w] - NAIP imagery with 4 spectral bands (scaled 0-1)
        - 'dates': List[str] - List of NAIP acquisition date strings
        - 'relative_dates': Tensor [n_images, 1] - Relative days from UAV acquisition
        - 'img_bbox': List/Tuple [4] - NAIP imagery bounding box [minx, miny, maxx, maxy]
        - 'ids': List[str] - List of NAIP image IDs
        - 'bands': List - NAIP band information
    naip_norm_stats : dict, optional
        Dictionary containing normalization statistics (not used with 0-1 scaled data)
    bbox_overlay : tuple, optional
        Bounding box to overlay on each image [minx, miny, maxx, maxy] (deprecated, 
        now automatically calculated as 10x10m box centered on img_bbox)
    figsize : tuple, optional
        Base figure size (width, height) - will be adjusted based on number of rows
    title : str, optional
        Overall title for the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure containing all NAIP images
    """
    # Extract the components from the NAIP dictionary
    imgs = naip_data['images']  # [n_images, 4, h, w]
    dates = naip_data.get('dates', [])
    img_bbox = naip_data.get('img_bbox')
    
    # Get the number of images
    n_images = imgs.shape[0]
    
    # Check if we have dates for all images
    if len(dates) < n_images:
        # Fill in missing dates
        dates = dates + [f"Unknown Date {i+1}" for i in range(len(dates), n_images)]
    
    # Sort the images by date if possible
    try:
        # Convert string dates to datetime objects for sorting
        datetime_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        # Create pairs of (index, datetime)
        pairs = list(zip(range(n_images), datetime_objects))
        # Sort pairs by datetime
        pairs.sort(key=lambda x: x[1])
        # Get sorted indices
        sorted_indices = [pair[0] for pair in pairs]
    except (ValueError, TypeError):
        # If date parsing fails, just use original order
        sorted_indices = list(range(n_images))
    
    # Calculate rows and columns for the subplot grid (3 columns)
    n_cols = 3
    n_rows = math.ceil(n_images / n_cols)
    
    # Adjust figsize based on number of rows
    adjusted_figsize = (figsize[0], figsize[1] * max(1, n_rows / 2))
    
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Ensure axes is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each image in order
    for i, idx in enumerate(sorted_indices):
        # Calculate row and column for this subplot
        row = i // n_cols
        col = i % n_cols
        
        # Get the image and convert to numpy if needed
        img = imgs[idx]
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy() if img.is_cuda else img.numpy()
        else:
            img_np = img
        
        # Create RGB image (use first 3 bands)
        if img_np.shape[0] >= 3:
            rgb = np.transpose(img_np[:3], (1, 2, 0))
            
            # Apply percentile-based contrast stretching for better visualization
            # Since data is already 0-1, we don't need to scale again, just enhance contrast
            if rgb.max() > 0:
                p_low, p_high = np.percentile(rgb[rgb > 0], [2, 98])
                rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
                
                # Scale to 0-255 for display
                rgb = rgb_stretched * 255
            else:
                rgb = np.zeros_like(rgb)
            
            rgb = rgb.astype(np.uint8)
        else:
            # Single band grayscale image
            rgb = img_np[0]
            
            # Apply contrast stretching
            if rgb.max() > rgb.min():
                p_low, p_high = np.percentile(rgb, [2, 98])
                rgb = np.clip((rgb - p_low) / (p_high - p_low), 0, 1) * 255
                rgb = rgb.astype(np.uint8)
        
        # Calculate extent from img_bbox
        if img_bbox is not None:
            extent = (img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3])
            
            # Calculate the centroid of the img_bbox (assumed to be 20x20 meters)
            centroid_x = (img_bbox[0] + img_bbox[2]) / 2
            centroid_y = (img_bbox[1] + img_bbox[3]) / 2
            
            # Calculate the 10x10 meter bbox that shares the centroid with img_bbox
            half_width = 5.0  # half of 10 meters
            half_height = 5.0  # half of 10 meters
            
            inner_bbox = [
                centroid_x - half_width,   # minx
                centroid_y - half_height,  # miny
                centroid_x + half_width,   # maxx
                centroid_y + half_height   # maxy
            ]
        else:
            extent = (0, img_np.shape[2], 0, img_np.shape[1])
            inner_bbox = None
        
        # Plot the image
        axes[row, col].imshow(rgb, extent=extent, origin='upper')
        
        # Draw the 10x10 meter bounding box centered on img_bbox
        if inner_bbox is not None:
            rect = Rectangle((inner_bbox[0], inner_bbox[1]),
                            inner_bbox[2] - inner_bbox[0],
                            inner_bbox[3] - inner_bbox[1],
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[row, col].add_patch(rect)
        
        # Add date as title
        date_str = dates[idx] if idx < len(dates) else f"Image {idx+1}"
        
        # Add relative date if available
        relative_date = ""
        if 'relative_dates' in naip_data and idx < len(naip_data['relative_dates']):
            if isinstance(naip_data['relative_dates'], torch.Tensor):
                rel_days = naip_data['relative_dates'][idx].item()
            else:
                rel_days = naip_data['relative_dates'][idx]
            
            # Format as +X or -X days relative to UAV capture
            sign = "+" if rel_days >= 0 else ""
            relative_date = f" ({sign}{rel_days:.0f} days)"
        
        # Add ID if available
        id_str = ""
        if 'ids' in naip_data and idx < len(naip_data['ids']):
            id_str = f"\n{naip_data['ids'][idx]}"
        
        axes[row, col].set_title(f"{date_str}{relative_date}{id_str}", fontsize=10)
        
        # Set ticks
        axes[row, col].set_xticks(np.linspace(extent[0], extent[1], num=5))
        axes[row, col].set_yticks(np.linspace(extent[2], extent[3], num=5))
        axes[row, col].tick_params(axis='both', which='both', labelsize=8)
    
    # Hide unused axes
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for the overall title
    
    return fig

def calculate_chamfer_distances(results, batch_size=4, device=None):
    """
    Calculate Chamfer distances with efficient memory usage and robust error handling.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries containing point cloud data
    batch_size : int
        Size of batches for processing (default: 1, increase cautiously)
    device : torch.device
        Device to use for calculations (if None, will use CUDA if available)
    
    Returns:
    --------
    results : list
        Updated results list with input_loss and improvement_ratio fields
    """
    import torch
    import numpy as np
    from src.utils.chamfer_distance import chamfer_distance
    import gc  # For garbage collection
    
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}, batch size: {batch_size}")
    
    # Initialize a counter for tracking progress
    processed_count = 0
    total_count = len(results)
    
    # Process small batches (or individual samples) to avoid memory issues
    for i in range(0, total_count, batch_size):
        # Get a slice of results for this batch
        batch_end = min(i + batch_size, total_count)
        current_batch = results[i:batch_end]
        current_batch_size = len(current_batch)
        
        try:
            if batch_size > 1:
                # Try batch processing first (if batch_size > 1)
                # Memory-efficient approach: don't pad to max size globally, just process each batch
                
                # Collect point clouds for this batch
                dep_points_list = []
                uav_points_list = []
                dep_lengths = []
                uav_lengths = []
                
                for result in current_batch:
                    dep_points = result['dep_points']
                    uav_points = result['uav_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(dep_points, torch.Tensor):
                        dep_points = torch.tensor(dep_points, dtype=torch.float32)
                    if not isinstance(uav_points, torch.Tensor):
                        uav_points = torch.tensor(uav_points, dtype=torch.float32)
                    
                    # Move to device
                    dep_points = dep_points.to(device)
                    uav_points = uav_points.to(device)
                    
                    # Add batch dimension
                    dep_points = dep_points.unsqueeze(0)
                    uav_points = uav_points.unsqueeze(0)
                    
                    dep_points_list.append(dep_points)
                    uav_points_list.append(uav_points)
                    dep_lengths.append(torch.tensor([dep_points.shape[1]], device=device))
                    uav_lengths.append(torch.tensor([uav_points.shape[1]], device=device))
                
                # Concatenate along batch dimension
                batch_dep_points = torch.cat(dep_points_list, dim=0)
                batch_uav_points = torch.cat(uav_points_list, dim=0)
                batch_dep_lengths = torch.cat(dep_lengths, dim=0)
                batch_uav_lengths = torch.cat(uav_lengths, dim=0)
                
                # Calculate Chamfer distances
                batch_losses, _ = chamfer_distance(
                    batch_dep_points,
                    batch_uav_points,
                    x_lengths=batch_dep_lengths,
                    y_lengths=batch_uav_lengths
                )
                
                # Update results
                for j, result in enumerate(current_batch):
                    result['input_loss'] = batch_losses[j].item()
                    
                    # Handle infinite loss values
                    if np.isinf(result['loss']):
                        print(f"Replacing Inf loss value for sample {result.get('tile_id', j)} with 32000")
                        result['loss'] = 32000.0
                    
                    # Calculate improvement ratio
                    result['improvement_ratio'] = result['input_loss'] / result['loss'] if result['loss'] > 0 else float('inf')
            else:
                # Process samples individually (fallback or if batch_size=1)
                raise Exception("Using individual processing for safety")
                
        except Exception as e:
            # Fallback to individual processing
            print(f"Batch processing error: {e}, falling back to individual processing")
            
            # Process each sample individually
            for result in current_batch:
                try:
                    # Extract point clouds
                    dep_points = result['dep_points']
                    uav_points = result['uav_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(dep_points, torch.Tensor):
                        dep_points = torch.tensor(dep_points, dtype=torch.float32)
                    if not isinstance(uav_points, torch.Tensor):
                        uav_points = torch.tensor(uav_points, dtype=torch.float32)
                    
                    # Move to device
                    dep_points = dep_points.to(device)
                    uav_points = uav_points.to(device)
                    
                    # Add batch dimension
                    dep_points_tensor = dep_points.unsqueeze(0)
                    uav_points_tensor = uav_points.unsqueeze(0)
                    
                    # Create length tensors
                    dep_length = torch.tensor([dep_points.shape[0]], device=device)
                    uav_length = torch.tensor([uav_points.shape[0]], device=device)
                    
                    # Calculate Chamfer distance
                    input_loss_calc, _ = chamfer_distance(
                        dep_points_tensor,
                        uav_points_tensor,
                        x_lengths=dep_length,
                        y_lengths=uav_length
                    )
                    
                    # Get scalar loss value
                    input_loss = input_loss_calc.item()
                    
                    # Store results
                    result['input_loss'] = input_loss
                    
                    # Handle infinite loss values
                    if np.isinf(result['loss']):
                        print(f"Replacing Inf loss value for sample {result.get('tile_id', 'unknown')} with 32000")
                        result['loss'] = 32000.0
                    
                    # Calculate improvement ratio
                    result['improvement_ratio'] = input_loss / result['loss'] if result['loss'] > 0 else float('inf')
                    
                except Exception as inner_e:
                    print(f"Error in individual processing for sample {result.get('tile_id', 'unknown')}: {inner_e}")
                    # Set default values
                    result['input_loss'] = 1.0
                    result['improvement_ratio'] = 1.0 / result['loss'] if result['loss'] > 0 else float('inf')
                
                # Force garbage collection after each sample to reduce memory pressure
                torch.cuda.empty_cache()
                gc.collect()
        
        # Update progress
        processed_count += current_batch_size
        print(f"Processed {processed_count}/{total_count} samples ({processed_count/total_count*100:.1f}%)")
        
        # Force garbage collection after each batch
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Completed Chamfer distance calculations for {processed_count} samples")
    return results



def calculate_chamfer_with_knn(results, batch_size=50, device=None):
    """
    Calculate Chamfer distances using KNN with batch processing for efficiency.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries containing point cloud data
    batch_size : int
        Number of samples to process in each batch (default: 10)
    device : torch.device
        Device to use for calculations (if None, will use CUDA if available)
    
    Returns:
    --------
    results : list
        Updated results list with input_loss and improvement_ratio fields
    """
    import torch
    import numpy as np
    import gc  # For garbage collection
    
    try:
        from pytorch3d.ops import knn_points
    except ImportError:
        print("PyTorch3D not found. Please install it with: pip install pytorch3d")
        # Fallback to single-sample processing
        print("Falling back to individual processing")
        batch_size = 1
    
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using KNN-based Chamfer calculation on {device} with batch size {batch_size}")
    
    # Function to calculate one-sided Chamfer distance using KNN
    def one_sided_chamfer(x, y, x_lengths=None, y_lengths=None):
        """
        Calculate one-sided Chamfer distance from x to y.
        
        Parameters:
        x : tensor of shape (B, N, 3) - source points
        y : tensor of shape (B, M, 3) - target points
        x_lengths : tensor of shape (B,) - actual lengths of x batches
        y_lengths : tensor of shape (B,) - actual lengths of y batches
        
        Returns:
        distances : tensor of shape (B,) - mean distance from each x to nearest y
        """
        # Find nearest neighbor in y for each point in x
        nearest = knn_points(x, y, K=1, lengths1=x_lengths, lengths2=y_lengths)
        
        # Get the squared distances to nearest neighbors
        dist = nearest.dists.squeeze(-1)  # (B, N)
        
        # If we have length information, we need to mask the padded values
        if x_lengths is not None:
            mask = torch.zeros_like(dist, dtype=torch.bool)
            for i, length in enumerate(x_lengths):
                mask[i, :length] = True
            # Apply mask to only consider valid points
            valid_dists = dist.masked_select(mask)
            # Reshape to (B, max_length) and compute mean only over valid points
            mean_dists = torch.zeros(dist.shape[0], device=dist.device)
            for i, length in enumerate(x_lengths):
                if length > 0:  # Avoid division by zero
                    mean_dists[i] = dist[i, :length].mean()
            return mean_dists
        else:
            # If no length information, just compute mean over all points
            return dist.mean(dim=1)  # (B,)
    
    # Process samples in batches
    total_count = len(results)
    processed_count = 0
    
    # Process in batches
    for i in range(0, total_count, batch_size):
        # Extract the current batch
        end_idx = min(i + batch_size, total_count)
        current_batch = results[i:end_idx]
        current_batch_size = len(current_batch)
        
        try:
            # First determine if we can process this as a real batch
            # (all point clouds must have the same dimensions for true batching)
            dep_sizes = [result['dep_points'].shape[0] for result in current_batch]
            uav_sizes = [result['uav_points'].shape[0] for result in current_batch]
            
            # Check if all sizes in the batch are the same
            uniform_dep_size = all(size == dep_sizes[0] for size in dep_sizes)
            uniform_uav_size = all(size == uav_sizes[0] for size in uav_sizes)
            
            if uniform_dep_size and uniform_uav_size and batch_size > 1:
                # We can use true batching (all point clouds have same dimensions)
                dep_points_list = []
                uav_points_list = []
                
                for result in current_batch:
                    # Extract point clouds
                    dep_points = result['dep_points']
                    uav_points = result['uav_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(dep_points, torch.Tensor):
                        dep_points = torch.tensor(dep_points, dtype=torch.float32)
                    if not isinstance(uav_points, torch.Tensor):
                        uav_points = torch.tensor(uav_points, dtype=torch.float32)
                    
                    # Add to lists
                    dep_points_list.append(dep_points)
                    uav_points_list.append(uav_points)
                
                # Stack along batch dimension
                batch_dep_points = torch.stack(dep_points_list).to(device)  # (B, N, 3)
                batch_uav_points = torch.stack(uav_points_list).to(device)  # (B, M, 3)
                
                # Calculate two-sided Chamfer distance
                forward_dist = one_sided_chamfer(batch_dep_points, batch_uav_points)
                backward_dist = one_sided_chamfer(batch_uav_points, batch_dep_points)
                
                # Full Chamfer distance
                chamfer_dists = forward_dist + backward_dist  # (B,)
                
                # Update results
                for j, (result, chamfer_dist) in enumerate(zip(current_batch, chamfer_dists)):
                    # Store input loss
                    input_loss = chamfer_dist.item()
                    result['input_loss'] = input_loss
                    
                    # Handle infinite loss values
                    if np.isinf(result['loss']):
                        print(f"Replacing Inf loss value for sample {result.get('tile_id', i+j)} with 32000")
                        result['loss'] = 32000.0
                    
                    # Calculate improvement ratio
                    result['improvement_ratio'] = input_loss / result['loss'] if result['loss'] > 0 else float('inf')
            
            else:
                # Point clouds have different sizes, use padded batching with lengths
                max_dep_size = max(dep_sizes)
                max_uav_size = max(uav_sizes)
                
                # Create padded tensors
                batch_dep_points = torch.zeros(current_batch_size, max_dep_size, 3, device=device)
                batch_uav_points = torch.zeros(current_batch_size, max_uav_size, 3, device=device)
                
                # Track actual lengths
                dep_lengths = torch.zeros(current_batch_size, dtype=torch.long, device=device)
                uav_lengths = torch.zeros(current_batch_size, dtype=torch.long, device=device)
                
                # Fill padded tensors
                for j, result in enumerate(current_batch):
                    # Extract point clouds
                    dep_points = result['dep_points']
                    uav_points = result['uav_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(dep_points, torch.Tensor):
                        dep_points = torch.tensor(dep_points, dtype=torch.float32)
                    if not isinstance(uav_points, torch.Tensor):
                        uav_points = torch.tensor(uav_points, dtype=torch.float32)
                    
                    # Move to device
                    dep_points = dep_points.to(device)
                    uav_points = uav_points.to(device)
                    
                    # Store actual lengths
                    dep_lengths[j] = dep_points.shape[0]
                    uav_lengths[j] = uav_points.shape[0]
                    
                    # Fill padded tensors
                    batch_dep_points[j, :dep_points.shape[0]] = dep_points
                    batch_uav_points[j, :uav_points.shape[0]] = uav_points
                
                # Calculate two-sided Chamfer distance with lengths
                forward_dist = one_sided_chamfer(batch_dep_points, batch_uav_points, 
                                                dep_lengths, uav_lengths)
                backward_dist = one_sided_chamfer(batch_uav_points, batch_dep_points, 
                                                 uav_lengths, dep_lengths)
                
                # Full Chamfer distance
                chamfer_dists = forward_dist + backward_dist  # (B,)
                
                # Update results
                for j, (result, chamfer_dist) in enumerate(zip(current_batch, chamfer_dists)):
                    # Store input loss
                    input_loss = chamfer_dist.item()
                    result['input_loss'] = input_loss
                    
                    # Handle infinite loss values
                    if np.isinf(result['loss']):
                        print(f"Replacing Inf loss value for sample {result.get('tile_id', i+j)} with 32000")
                        result['loss'] = 32000.0
                    
                    # Calculate improvement ratio
                    result['improvement_ratio'] = input_loss / result['loss'] if result['loss'] > 0 else float('inf')
        
        except Exception as e:
            print(f"Error in batch KNN Chamfer calculation: {e}")
            print("Falling back to individual processing for this batch")
            
            # Process each sample individually
            for j, result in enumerate(current_batch):
                try:
                    # Extract point clouds
                    dep_points = result['dep_points']
                    uav_points = result['uav_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(dep_points, torch.Tensor):
                        dep_points = torch.tensor(dep_points, dtype=torch.float32)
                    if not isinstance(uav_points, torch.Tensor):
                        uav_points = torch.tensor(uav_points, dtype=torch.float32)
                    
                    # Move to device
                    dep_points = dep_points.to(device)
                    uav_points = uav_points.to(device)
                    
                    # Add batch dimension
                    dep_points = dep_points.unsqueeze(0)  # (1, N, 3)
                    uav_points = uav_points.unsqueeze(0)  # (1, M, 3)
                    
                    # Calculate two-sided Chamfer distance
                    forward_dist = one_sided_chamfer(dep_points, uav_points)
                    backward_dist = one_sided_chamfer(uav_points, dep_points)
                    
                    # Full Chamfer distance is the sum
                    chamfer_dist = forward_dist + backward_dist  # (1,)
                    
                    # Get scalar loss value
                    input_loss = chamfer_dist.item()
                    
                    # Store results
                    result['input_loss'] = input_loss
                    
                    # Handle infinite loss values
                    if np.isinf(result['loss']):
                        print(f"Replacing Inf loss value for sample {result.get('tile_id', i+j)} with 32000")
                        result['loss'] = 32000.0
                    
                    # Calculate improvement ratio
                    result['improvement_ratio'] = input_loss / result['loss'] if result['loss'] > 0 else float('inf')
                
                except Exception as inner_e:
                    print(f"Error in individual KNN Chamfer calculation: {inner_e}")
                    # Set default values
                    result['input_loss'] = 1.0
                    result['improvement_ratio'] = 1.0 / result['loss'] if result['loss'] > 0 else float('inf')
        
        # Force garbage collection after each batch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Update progress
        processed_count += current_batch_size
        print(f"Processed {processed_count}/{total_count} samples ({processed_count/total_count*100:.1f}%)")
    
    print(f"Completed KNN-based Chamfer distance calculations for {processed_count} samples")
    return results



def generate_model_report(
    model_path,
    validation_data_path,
    output_dir="data/output/reports",
    n_high_loss_samples=200,
    n_random_samples=200,
    n_low_improvement_samples=0,
    n_high_improvement_samples=0,
    dpi=150,
    naip_norm_stats_path="data/processed/model_data/naip_normalization_stats.pt",
    model_config=None
):
    """
    Generate a comprehensive PDF report for model evaluation.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model checkpoint
    validation_data_path : str
        Path to the validation data
    output_dir : str, optional
        Directory to save the report
    n_high_loss_samples : int, optional
        Number of highest loss samples to include
    n_random_samples : int, optional
        Number of random samples to include
    n_low_improvement_samples : int, optional
        Number of lowest improvement samples to include
    n_high_improvement_samples : int, optional
        Number of highest improvement samples to include
    dpi : int, optional
        DPI for rasterized images in the report
    naip_norm_stats_path : str, optional
        Path to NAIP normalization statistics
    model_config : MultimodalModelConfig, optional
        Model configuration. If None, a default configuration will be used
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NAIP normalization statistics if available
    naip_norm_stats = None
    try:
        if os.path.exists(naip_norm_stats_path):
            naip_norm_stats = torch.load(naip_norm_stats_path, map_location='cpu')
            print(f"Loaded NAIP normalization statistics from {naip_norm_stats_path}")
    except Exception as e:
        print(f"Error loading NAIP normalization statistics: {e}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set model configuration
    if model_config is None:
        # Use default configuration if no config is provided
        config = MultimodalModelConfig(
            # Default configuration
            fnl_attn_hds=4,
            feature_dim=256,     
            k=15,                
            up_attn_hds=4,       
            up_ratio=2,          
            pos_mlp_hdn=32,      
           
            use_naip=False,
            use_uavsar=False,
            img_embed_dim=64,    
            img_num_patches=16,  
            naip_dropout=0.05,
            uavsar_dropout=0.05,
            temporal_encoder='transformer',
            fusion_type='cross_attention',
            max_dist_ratio=1.5,
            fusion_dropout=0.10,
            fusion_num_heads=4,
            position_encoding_dim=24,
    
            # Point Transformer parameters
            num_lcl_heads = 4,  # Local attention heads (for MultiHeadPointTransformerConv)
            num_glbl_heads = 4,  # Global attention heads (for PosAwareGlobalFlashAttention)
            pt_attn_dropout = 0.0
        )
    else:
        # Use provided configuration
        config = model_config
    
    # Load the model
    model = load_model(model_path, config)
    model.to(device)
    print("Model loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load validation data
    try:
        print(f"Loading validation data from: {validation_data_path}")
        validation_data = torch.load(validation_data_path, map_location='cpu')
        print("Validation data loaded successfully")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return
    
    # Create dataset
    if isinstance(validation_data, list):
        print(f"Creating dataset from list of {len(validation_data)} samples")
        validation_dataset = RawDataset(validation_data, config.k)
    else:
        print("Unexpected validation data format")
        return
    
    print(f"Created validation dataset with {len(validation_dataset)} samples")
    
    # Create DataLoader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=multimodal_variable_size_collate
    )
    
    # Evaluate the model
    print("Evaluating model on validation data...")
    results = evaluate_validation_samples(model, validation_loader, device)
    print(f"Evaluated {len(results)} samples")
    
    # Calculate input losses and improvement ratios for all samples using optimized functions
    print("Calculating Chamfer distances (this may take some time)...")
    try:
        # Try the KNN-based method first (most memory efficient)
        print("Trying KNN-based Chamfer calculation...")
        results = calculate_chamfer_with_knn(results, device=device)
    except Exception as e:
        print(f"KNN-based method failed: {e}")
        print("Falling back to standard method with small batch size...")
        # Fall back to standard method with small batch size
        results = calculate_chamfer_distances(results, batch_size=4, device=device)
    
    # Create different sample sets
    # 1. Sort by prediction loss (high to low)
    results_by_loss = sorted(results, key=lambda x: x['loss'], reverse=True)
    high_loss_samples = results_by_loss[:n_high_loss_samples]
    
    # 2. Sort by improvement ratio (low to high) for low improvement samples
    results_by_improvement = sorted(results, key=lambda x: x['improvement_ratio'])
    low_improvement_samples = results_by_improvement[:n_low_improvement_samples]
    
    # 3. Sort by improvement ratio (high to low) for high improvement samples
    high_improvement_samples = sorted(results, key=lambda x: x['improvement_ratio'], reverse=True)[:n_high_improvement_samples]
    
    # 4. Get remaining samples (not in any of the above sets)
    used_samples = set()
    for sample in high_loss_samples + low_improvement_samples + high_improvement_samples:
        used_samples.add(sample['sample_idx'])
    
    remaining_samples = [s for s in results if s['sample_idx'] not in used_samples]
    
    # Select random samples from remaining samples
    random_sample_count = min(n_random_samples, len(remaining_samples))
    random_samples = random.sample(remaining_samples, random_sample_count) if random_sample_count > 0 else []
    
    # Generate the PDF report
    report_path = os.path.join(output_dir, f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    print(f"Generating report to {report_path}...")
    with PdfPages(report_path) as pdf:
        # Generate overview pages
        generate_overview_pages(pdf, results, model_path)
        
        # Generate high loss sample pages
        if n_high_loss_samples > 0:
            print(f"Adding {len(high_loss_samples)} highest loss samples...")
            for i, sample in enumerate(tqdm(high_loss_samples)):
                generate_sample_page(pdf, sample, i, "high-loss", naip_norm_stats=naip_norm_stats)
        
        # Generate low improvement sample pages
        if n_low_improvement_samples > 0:
            print(f"Adding {len(low_improvement_samples)} lowest improvement samples...")
            for i, sample in enumerate(tqdm(low_improvement_samples)):
                generate_sample_page(pdf, sample, i, "low-improvement", naip_norm_stats=naip_norm_stats)
        
        # Generate high improvement sample pages
        if n_high_improvement_samples > 0:
            print(f"Adding {len(high_improvement_samples)} highest improvement samples...")
            for i, sample in enumerate(tqdm(high_improvement_samples)):
                generate_sample_page(pdf, sample, i, "high-improvement", naip_norm_stats=naip_norm_stats)
        
        # Generate random sample pages
        if n_random_samples > 0:
            print(f"Adding {len(random_samples)} random samples...")
            for i, sample in enumerate(tqdm(random_samples)):
                generate_sample_page(pdf, sample, i, "random", naip_norm_stats=naip_norm_stats)
    
    print(f"Report saved to {report_path}")

def generate_overview_pages(pdf, results, model_path, dpi=150):
    """
    Generate overview pages with performance metrics and plots.
    """
    # Extract losses
    losses = [result['loss'] for result in results]
    
    # Create a figure for the first overview page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Model Evaluation Report", fontsize=16)
    
    # Add report generation time
    fig.text(0.5, 0.94, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             ha='center', fontsize=12)
    
    # Add model details
    fig.text(0.5, 0.9, f"Model: {os.path.basename(model_path)}", ha='center', fontsize=12)
    
    # Add overall statistics
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    std_loss = np.std(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    
    stats_text = (
        f"Number of samples: {len(results)}\n"
        f"Mean Chamfer Loss: {mean_loss:.6f}\n"
        f"Median Chamfer Loss: {median_loss:.6f}\n"
        f"Std Dev: {std_loss:.6f}\n"
        f"Min Loss: {min_loss:.6f}\n"
        f"Max Loss: {max_loss:.6f}"
    )
    
    fig.text(0.5, 0.8, stats_text, ha='center', fontsize=12)
    
    # Add the figure to the PDF with specified DPI
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    
    # Create a second overview page with distribution plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("Loss Distribution", fontsize=16)
    
    # Filter out extreme loss values (>15) for better visualization
    filtered_losses = [loss for loss in losses if loss <= 15]
    
    # Histogram of losses (filtered)
    ax1.hist(filtered_losses, bins=50, alpha=0.7, color='royalblue')
    ax1.set_title('Chamfer Loss Histogram (Values ≤ 15)')
    ax1.set_xlabel('Loss')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Sorted losses (filtered for better visualization)
    sorted_losses = sorted(filtered_losses)
    ax2.plot(sorted_losses, marker='.', markersize=2, linestyle='-', linewidth=1, color='royalblue')
    ax2.set_title('Sorted Chamfer Losses (Values ≤ 15)')
    ax2.set_xlabel('Sample Index (sorted)')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Highlight the threshold for high-loss samples
    if len(sorted_losses) >= 200:
        threshold = sorted_losses[-200]
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                    label=f'Top 200 threshold: {threshold:.6f}')
        ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

def generate_sample_page(pdf, sample, index, sample_type, dpi=150, naip_norm_stats=None):
    """
    Generate a PDF page for a single sample.
    """
    # Extract data
    dep_points = sample['dep_points'].numpy()
    uav_points = sample['uav_points'].numpy()
    pred_points = sample['pred_points'].numpy()
    
    # Use the correct keys or calculate if needed
    prediction_loss = sample['loss']
    input_loss = sample.get('input_loss', 1.0)  # Use the one we calculated in generate_model_report
    improvement_ratio = sample.get('improvement_ratio', input_loss / prediction_loss if prediction_loss > 0 else float('inf'))
    
    tile_id = sample['tile_id']
    bbox = sample.get('bbox')
    
    # Create a figure for point cloud visualizations
    fig = plt.figure(figsize=(11, 8.5))
    
    # Add title based on sample type
    if sample_type == "high-loss":
        title = f"High-Loss Sample {index+1}: Tile {tile_id}"
    elif sample_type == "low-improvement":
        title = f"Low-Improvement Sample {index+1}: Tile {tile_id}"
    elif sample_type == "high-improvement":
        title = f"High-Improvement Sample {index+1}: Tile {tile_id}"
    else:
        title = f"Random Sample {index+1}: Tile {tile_id}"
    
    fig.suptitle(title, fontsize=14)
    
    # Add loss information
    fig.text(0.5, 0.95, 
             f"Prediction Loss: {prediction_loss:.6f} | Input Loss: {input_loss:.6f} | Improvement Ratio: {improvement_ratio:.2f}x", 
             ha='center', fontsize=10)
    
    # Create 3D scatter plots for point clouds - with rasterized=True for all scatter plots
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(dep_points[:, 0], dep_points[:, 1], dep_points[:, 2], 
                s=1, alpha=0.5, c='blue', label='3DEP Points', rasterized=True)
    ax1.set_title('3DEP Input')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(uav_points[:, 0], uav_points[:, 1], uav_points[:, 2], 
                s=1, alpha=0.5, c='green', label='UAV Ground Truth', rasterized=True)
    ax2.set_title('UAV Ground Truth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                s=1, alpha=0.5, c='red', label='Predicted Points', rasterized=True)
    ax3.set_title('Model Prediction')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Comparison of ground truth vs prediction
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(uav_points[:, 0], uav_points[:, 1], uav_points[:, 2], 
                s=1, alpha=0.3, c='green', label='UAV Ground Truth', rasterized=True)
    ax4.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                s=1, alpha=0.3, c='red', label='Predicted Points', rasterized=True)
    ax4.set_title('Comparison: Ground Truth vs Prediction')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Add the figure to the PDF with specified DPI for rasterization
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

    naip_data = sample.get('naip')
    if naip_data and 'images' in naip_data and naip_data['images'] is not None:
        try:
            # Create NAIP visualization with normalization stats
            naip_title = f"{title} - NAIP Imagery"
            naip_fig = plot_naip_imagery(
                naip_data, 
                naip_norm_stats=naip_norm_stats,
                bbox_overlay=bbox if bbox is not None else None,
                title=naip_title
            )
            
            # Add the figure to the PDF with specified DPI
            pdf.savefig(naip_fig, dpi=dpi)
            plt.close(naip_fig)
            
        except Exception as e:
            print(f"Error plotting NAIP imagery for sample {tile_id}: {e}")
            # Create a simple error page
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f"{title} - NAIP Imagery (Error)", fontsize=14)
            fig.text(0.5, 0.5, f"Error plotting NAIP imagery: {e}", 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)  
    # Create boxen plots for dimensions - if this is causing duplication, we'll let the user decide
    # whether to keep this part
    try:
        boxen_fig = create_dimension_boxen_plots(dep_points, uav_points, pred_points)
        boxen_fig.suptitle(f"{title} - Dimension Distributions", fontsize=14)
        pdf.savefig(boxen_fig, dpi=dpi)
        plt.close(boxen_fig)
    except Exception as e:
        print(f"Error creating boxen plots for sample {tile_id}: {e}")
        # Create a simple error page
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(f"{title} - Dimension Distributions (Error)", fontsize=14)
        fig.text(0.5, 0.5, f"Error creating boxen plots: {e}", 
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)
    


if __name__ == "__main__":
# Define a base model configuration
    base_config = MultimodalModelConfig(
        # Core model parameters
        feature_dim=256,     # Feature dimension
        k=16,                # Number of neighbors for KNN
        up_ratio=2,          # Upsampling ratio
        pos_mlp_hdn=16,      # Hidden dimension for positional MLP
        pt_attn_dropout=0.05,
        
        # Granular attention head configurations
        extractor_lcl_heads=8,  # Local attention heads for feature extractor
        extractor_glbl_heads=4,  # Global attention heads for feature extractor
        expansion_lcl_heads=8,  # Local attention heads for feature expansion
        expansion_glbl_heads=4,  # Global attention heads for feature expansion
        refinement_lcl_heads=4,  # Local attention heads for feature refinement
        refinement_glbl_heads=4,  # Global attention heads for feature refinement
        
        # Deprecated/legacy parameters
        num_lcl_heads=4,      # Local attention heads (for backward compatibility)
        num_glbl_heads=4,     # Global attention heads (for backward compatibility)
        up_attn_hds=4,        # Legacy parameter (upsampling attention heads)
        fnl_attn_hds=2,       # Legacy parameter (final attention heads)
        up_concat=True,       # Legacy parameter (no longer used)
        up_beta=False,        # Legacy parameter
        
        # Modality flags
        use_naip=True,
        use_uavsar=True,
        
        # Imagery encoder parameters
        img_embed_dim=128,    # Dimension of patch embeddings
        img_num_patches=16,  # Number of output patch embeddings
        naip_dropout=0.03,
        uavsar_dropout=0.03,
        temporal_encoder='gru',  # Type of temporal encoder
        
        # Fusion parameters
        fusion_type='cross_attention',
        max_dist_ratio=8,

        # Cross attention fusion parameters
        fusion_dropout=0.03,
        fusion_num_heads=4,
        position_encoding_dim=36,
        
        # Other parameters
        attr_dim=3,
    )

    model_path = "/home/jovyan/geoai_veg_map/data/output/checkpoints/0423_e110_3.5e-4lr_dsclr4_wd5e-3_naip_uavsar_k16_f256_b10_e110.pth"
    validation_data_path = "data/processed/model_data/precomputed_validation_tiles_32bit.pt"
    output_dir = "data/output/reports"


    # Generate the report
    generate_model_report(
        model_path=model_path,
        validation_data_path=validation_data_path,
        output_dir=output_dir,
        n_high_loss_samples=30,
        n_low_improvement_samples=30,
        n_high_improvement_samples=30,
        n_random_samples=50,
        dpi=150,
        model_config=base_config  
    )