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
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Union, Any

# Configure matplotlib for better PDF output with rasterization
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better text rendering
mpl.rcParams['figure.dpi'] = 100   # Default figure DPI
mpl.rcParams['savefig.dpi'] = 150  # Default save DPI
mpl.rcParams['figure.autolayout'] = True  # Better layout
mpl.rcParams['path.simplify'] = True  # Simplify paths for better rendering
mpl.rcParams['path.simplify_threshold'] = 0.8  # Higher threshold for more simplification

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Define the MultimodalModelConfig class (simplified version)
@dataclass
class MultimodalModelConfig:
    """Configuration for the MultimodalPointUpsampler model."""
    # Core model parameters
    feature_dim: int = 256
    k: int = 16
    up_ratio: int = 2
    pos_mlp_hdn: int = 32
    
    # Attention head configurations
    fnl_attn_hds: int = 4
    up_attn_hds: int = 4
    num_lcl_heads: int = 4
    num_glbl_heads: int = 4
    extractor_lcl_heads: int = 8
    extractor_glbl_heads: int = 4
    expansion_lcl_heads: int = 8
    expansion_glbl_heads: int = 4
    refinement_lcl_heads: int = 4
    refinement_glbl_heads: int = 4
    
    # Modality flags
    use_naip: bool = True
    use_uavsar: bool = True
    
    # Imagery encoder parameters
    img_embed_dim: int = 128
    img_num_patches: int = 16
    naip_dropout: float = 0.05
    uavsar_dropout: float = 0.05
    temporal_encoder: str = 'gru'
    
    # Fusion parameters
    fusion_type: str = 'cross_attention'
    max_dist_ratio: float = 8.0
    fusion_dropout: float = 0.05
    fusion_num_heads: int = 4
    position_encoding_dim: int = 36
    
    # Dropout parameters
    pt_attn_dropout: float = 0.0
    
    # Other parameters
    attr_dim: int = 3
    up_concat: bool = True
    up_beta: bool = False

class RawDataset(Dataset):
    """Dataset wrapper for a list of samples."""
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get the sample
        sample = self.samples[idx]
        
        # Set edge_index from precomputed knn_edge_indices
        if 'knn_edge_indices' in sample and self.k in sample['knn_edge_indices']:
            sample['dep_edge_index'] = sample['knn_edge_indices'][self.k]
        else:
            print(f"Warning: No precomputed k={self.k} edge indices for sample {idx}")
        
        # Return the basic sample data
        dep_points_norm = sample['dep_points_norm'] 
        uav_points_norm = sample['uav_points_norm']
        edge_index = sample['dep_edge_index']
        dep_points_attr = sample.get('dep_points_attr_norm', None)
        
        # Extract normalization parameters
        center = sample.get('center', None)
        scale = sample.get('scale', None)
        bbox = sample.get('bbox', None)
        
        # Add imagery data
        naip_data = None
        uavsar_data = None
        
        # Add tile_id for debugging
        tile_id = sample.get('tile_id', None)
        
        # NAIP data
        if 'naip' in sample:
            naip_data = {
                'images': sample['naip'].get('images', None),
                'img_bbox': sample['naip'].get('img_bbox', None),
                'relative_dates': sample['naip'].get('relative_dates', None),
                'dates': sample['naip'].get('dates', None),  # Include dates for plotting
                'ids': sample['naip'].get('ids', None)       # Include IDs for reference
            }
        
        # UAVSAR data
        if 'uavsar' in sample:
            uavsar_data = {
                'images': sample['uavsar'].get('images', None),
                'img_bbox': sample['uavsar'].get('img_bbox', None),
                'relative_dates': sample['uavsar'].get('relative_dates', None),
                'dates': sample['uavsar'].get('dates', None),
                'ids': sample['uavsar'].get('ids', None)
            }
        
        return (dep_points_norm, uav_points_norm, edge_index, dep_points_attr, naip_data, uavsar_data, center, scale, bbox, tile_id)

def multimodal_variable_size_collate(batch):
    """
    Custom collate function for batches with variable-sized point clouds and imagery.
    Keeps each item separate rather than attempting to create tensors of uniform size.
    
    Args:
        batch: List of tuples containing sample data
        
    Returns:
        Tuple of lists for each data type
    """
    dep_points_list = []
    uav_points_list = []
    edge_index_list = []
    dep_attr_list = []
    naip_data_list = []
    uavsar_data_list = []
    center_list = []
    scale_list = []
    bbox_list = []
    tile_id_list = []
    
    for dep_points, uav_points, edge_index, dep_attr, naip_data, uavsar_data, center, scale, bbox, tile_id in batch:
        dep_points_list.append(dep_points)
        uav_points_list.append(uav_points)
        edge_index_list.append(edge_index)
        dep_attr_list.append(dep_attr)
        naip_data_list.append(naip_data)
        uavsar_data_list.append(uavsar_data)
        center_list.append(center)
        scale_list.append(scale)
        bbox_list.append(bbox)
        tile_id_list.append(tile_id)
    
    return dep_points_list, uav_points_list, edge_index_list, dep_attr_list, naip_data_list, uavsar_data_list, center_list, scale_list, bbox_list, tile_id_list

def load_model(model_path, config):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
    # Import the model class
    try:
        from src.models.multimodal_model import MultimodalPointUpsampler
    except ImportError:
        print("Warning: Failed to import MultimodalPointUpsampler. Make sure the src module is in your Python path.")
        raise ImportError("Could not import MultimodalPointUpsampler. Please ensure the model implementation is available.")
    
    # Create model with the same configuration
    model = MultimodalPointUpsampler(config)
    
    # Load state dict from checkpoint
    print(f"Loading model from path: {model_path}")
    try:
        # Try loading with weights_only=True to avoid warnings
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        state_dict = torch.load(model_path, map_location='cpu')
        print("Note: Using legacy model loading method")
    
    model.load_state_dict(state_dict)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

def process_sample_for_evaluation(model, sample_data, device):
    """
    Process a single sample through the model and return both the predictions and loss.
    
    Args:
        model: The model to evaluate
        sample_data: Tuple of (dep_points, uav_points, edge_index, dep_attr, naip_data, uavsar_data, center, scale, bbox, tile_id)
        device: Device to run evaluation on
        
    Returns:
        Tuple of (pred_points, loss)
    """
    dep_points, uav_points, e_idx, dep_attr, naip_data, uavsar_data, center, scale, bbox, tile_id = sample_data
    
    # Move data to device
    dep_points = dep_points.to(device)
    uav_points = uav_points.to(device)
    e_idx = e_idx.to(device)
    if dep_attr is not None:
        dep_attr = dep_attr.to(device)
    
    # Get normalization parameters
    if center is not None:
        center = center.to(device)
    if scale is not None:
        scale = scale.to(device)
    if bbox is not None:
        bbox = bbox.to(device)
    
    # Move imagery data to device if available
    if naip_data is not None and isinstance(naip_data, dict):
        if 'images' in naip_data and naip_data['images'] is not None:
            naip_data['images'] = naip_data['images'].to(device)
    
    if uavsar_data is not None and isinstance(uavsar_data, dict):
        if 'images' in uavsar_data and uavsar_data['images'] is not None:
            uavsar_data['images'] = uavsar_data['images'].to(device)
    
    # Use automatic mixed precision to handle float16/float32 mismatches
    # This is important because the model was trained with mixed precision
    try:
        from torch.amp import autocast
        with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.float16):
            # Run the model with all available data including spatial information
            pred_points = model(
                dep_points, e_idx, dep_attr, 
                naip_data, uavsar_data,
                center, scale, bbox
            )
    except (ImportError, RuntimeError) as e:
        print(f"Warning: Mixed precision inference failed: {e}. Falling back to full precision.")
        # Fallback to regular inference
        pred_points = model(
            dep_points, e_idx, dep_attr, 
            naip_data, uavsar_data,
            center, scale, bbox
        )

    # Create batch tensors for chamfer distance calculation
    pred_points_batch = pred_points.unsqueeze(0)
    uav_points_batch = uav_points.unsqueeze(0)
    
    pred_length = torch.tensor([pred_points.shape[0]], device=device)
    uav_length = torch.tensor([uav_points.shape[0]], device=device)
    
    # Calculate Chamfer distance loss
    try:
        from src.utils.chamfer_distance import chamfer_distance
        chamfer_loss, _ = chamfer_distance(
            pred_points_batch, 
            uav_points_batch,
            x_lengths=pred_length,
            y_lengths=uav_length
        )
    except ImportError:
        # Fallback to custom implementation
        print("Warning: Unable to import chamfer_distance. Using fallback implementation.")
        # Use KNN Chamfer implementation from above
        mock_results = [{
            'dep_points': dep_points.cpu(),
            'uav_points': uav_points.cpu(),
            'pred_points': pred_points.cpu(),
            'loss': 1.0  # Placeholder
        }]
        processed_results = calculate_chamfer_with_knn(mock_results, batch_size=1, device=device)
        chamfer_loss = torch.tensor(processed_results[0]['loss'], device=device)
    
    if torch.isnan(chamfer_loss):
        print(f"WARNING: Loss for sample is NaN! {tile_id}")
        chamfer_loss = torch.tensor(1e6, device=device)  # Use a large value for NaN

    if torch.isinf(chamfer_loss):
        print(f"WARNING: Loss for sample is Inf! {tile_id}")
        chamfer_loss = torch.tensor(1e6, device=device)  # Use a large value for Inf
    
    return pred_points, chamfer_loss.item()

def evaluate_validation_samples(model, dataloader, device):
    """
    Evaluate a model on a validation dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with validation samples
        device: Device to run evaluation on
        
    Returns:
        List of result dictionaries with predictions and metrics
    """
    model.eval()  # Set model to evaluation mode
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list = batch
            
            for i in range(len(dep_list)):
                sample_data = (
                    dep_list[i], uav_list[i], edge_list[i], attr_list[i],
                    naip_list[i], uavsar_list[i], center_list[i], scale_list[i],
                    bbox_list[i], tile_id_list[i]
                )
                
                # Process sample and get predictions and loss
                try:
                    pred_points, loss = process_sample_for_evaluation(model, sample_data, device)
                
                    # Get the tile ID
                    tile_id = tile_id_list[i]
                    if isinstance(tile_id, torch.Tensor):
                        tile_id = tile_id.item() if tile_id.numel() == 1 else tile_id.cpu().numpy()
                    
                    # Store results
                    result = {
                        'sample_idx': batch_idx * len(dep_list) + i,
                        'tile_id': tile_id,
                        'pred_points': pred_points.cpu(),
                        'uav_points': uav_list[i].cpu(),
                        'dep_points': dep_list[i].cpu(),
                        'loss': loss,
                        'naip': naip_list[i],  
                        'uavsar': uavsar_list[i],  
                        'bbox': bbox_list[i] if bbox_list[i] is not None else None  # Include bbox for overlay
                    }
                    
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample {batch_idx * len(dep_list) + i}: {e}")
                    # Skip this sample
                    continue
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
    
    return results

def plot_naip_imagery_row(naip_data, naip_norm_stats=None, bbox_overlay=None, figsize=(15, 3), title=None):
    """
    Create a single row of NAIP imagery plots, ordered by date.
    
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
        Dictionary containing normalization statistics
    bbox_overlay : tuple, optional
        Bounding box to overlay on each image [minx, miny, maxx, maxy]
    figsize : tuple, optional
        Base figure size (width, height)
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
    
    # Create the figure with all plots in a single row
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Ensure axes is always an array
    if n_images == 1:
        axes = np.array([axes])
    
    # Plot each image in order
    for i, idx in enumerate(sorted_indices):
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
            if rgb.max() > 0:
                p_low, p_high = np.percentile(rgb[rgb > 0], [2, 98])
                rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
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
            
            # Calculate the centroid of the img_bbox
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
        axes[i].imshow(rgb, extent=extent, origin='upper')
        
        # Draw the 10x10 meter bounding box centered on img_bbox
        if inner_bbox is not None:
            rect = Rectangle((inner_bbox[0], inner_bbox[1]),
                            inner_bbox[2] - inner_bbox[0],
                            inner_bbox[3] - inner_bbox[1],
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
        
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
        
        axes[i].set_title(f"{date_str}{relative_date}", fontsize=10)
        
        # Set ticks
        axes[i].tick_params(axis='both', which='both', labelsize=8)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.85)  # Make room for the overall title
    
    return fig

def calculate_chamfer_with_knn(results, batch_size=50, device=None):
    """
    Calculate Chamfer distances using KNN with batch processing for efficiency.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries containing point cloud data
    batch_size : int
        Number of samples to process in each batch (default: 50)
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
    
    # print(f"Using KNN-based Chamfer calculation on {device} with batch size {batch_size}")
    
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
        if total_count > 1:
            print(f"Processed {processed_count}/{total_count} samples ({processed_count/total_count*100:.1f}%)")
    
    # print(f"Completed KNN-based Chamfer distance calculations for {processed_count} samples")
    return results

# Helper function to calculate Chamfer distance for a single pair of point clouds
def calculate_chamfer_distance(source_points, target_points, device=None):
    """
    Calculate Chamfer distance between two point clouds using KNN approach.
    This is a simplified wrapper around the KNN implementation.
    
    Parameters:
    -----------
    source_points : numpy.ndarray or torch.Tensor
        Source point cloud (N, 3)
    target_points : numpy.ndarray or torch.Tensor
        Target point cloud (M, 3)
    device : torch.device, optional
        Device to use for calculations
        
    Returns:
    --------
    float
        Chamfer distance
    """
    import torch
    import numpy as np
    
    # Create mock results list with a single entry
    mock_results = [{
        'dep_points': source_points,
        'uav_points': target_points,
        'loss': 1.0  # Placeholder
    }]
    
    # Use the KNN-based implementation
    results = calculate_chamfer_with_knn(mock_results, batch_size=1, device=device)
    
    # Return the input_loss from the processed result
    return results[0]['input_loss']

def generate_multi_model_report(
    model_paths,
    validation_data_path,
    output_dir="data/output/reports",
    n_high_loss_samples=30,
    n_low_improvement_samples=30,
    n_high_improvement_samples=30,
    n_random_samples=30,
    dpi=150,
    point_size=1,
    point_alpha=0.5,
    naip_norm_stats_path="data/processed/model_data/naip_normalization_stats.pt",
    base_config=None
):
    """
    Generate a comprehensive PDF report for multiple model evaluation.
    
    Parameters:
    -----------
    model_paths : dict
        Dictionary with paths to model checkpoints:
        {
            'combined': path to NAIP+UAVSAR model (or None),
            'naip': path to NAIP-only model (or None),
            'uavsar': path to UAVSAR-only model (or None),
            'baseline': path to baseline model without imagery (or None)
        }
    validation_data_path : str
        Path to the validation data
    output_dir : str, optional
        Directory to save the report
    n_high_loss_samples : int, optional
        Number of highest loss samples to include
    n_low_improvement_samples : int, optional
        Number of lowest improvement samples to include
    n_high_improvement_samples : int, optional
        Number of highest improvement samples to include
    n_random_samples : int, optional
        Number of random samples to include
    dpi : int, optional
        DPI for rasterized images in the report
    point_size : float, optional
        Size of points in scatter plots
    point_alpha : float, optional
        Alpha (transparency) of points in scatter plots
    naip_norm_stats_path : str, optional
        Path to NAIP normalization statistics
    base_config : MultimodalModelConfig, optional
        Base model configuration. If None, a default configuration will be used
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
    
    # Set base model configuration
    if base_config is None:
        # Use default configuration if no config is provided
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
            
            # Modality flags
            use_naip=True,
            use_uavsar=True,
            
            # Imagery encoder parameters
            img_embed_dim=128,    # Dimension of patch embeddings
            img_num_patches=16,  # Number of output patch embeddings
            naip_dropout=0.05,
            uavsar_dropout=0.05,
            temporal_encoder='gru',  # Type of temporal encoder
            
            # Fusion parameters
            fusion_type='cross_attention',
            max_dist_ratio=8,
            fusion_dropout=0.05,
            fusion_num_heads=4,
            position_encoding_dim=36,
            
            # Other parameters
            attr_dim=3,
        )
    
    # Load validation data
    try:
        print(f"Loading validation data from: {validation_data_path}")
        validation_data = torch.load(validation_data_path, map_location='cpu')
        print(f"Validation data loaded successfully with {len(validation_data)} samples")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return
    
    # Create dataset
    validation_dataset = RawDataset(validation_data, base_config.k)
    print(f"Created validation dataset with {len(validation_dataset)} samples")
    
    # Create DataLoader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=multimodal_variable_size_collate
    )
    
    # Load models and run inference for each model
    models_results = {}
    all_models_configs = {}
    
    # Define model configurations for each model type
    model_configs = {
        'combined': dict(use_naip=True, use_uavsar=True),
        'naip': dict(use_naip=True, use_uavsar=False),
        'uavsar': dict(use_naip=False, use_uavsar=True),
        'baseline': dict(use_naip=False, use_uavsar=False),
    }
    
    # Dictionary to track which models are available
    available_models = {}
    
    # Loop through all model types
    for model_type, model_path in model_paths.items():
        if model_path is None:
            print(f"Skipping {model_type} model (path is None)")
            available_models[model_type] = False
            continue
        
        print(f"\nProcessing {model_type} model: {model_path}")
        available_models[model_type] = True
        
        # Create a copy of the base config and update with model-specific settings
        model_config = MultimodalModelConfig(**vars(base_config))
        # Update model config with the specific modality flags
        for key, value in model_configs[model_type].items():
            setattr(model_config, key, value)
        
        # Store the config for later use
        all_models_configs[model_type] = model_config
        
        # Load the model
        try:
            model = load_model(model_path, model_config)
            model.to(device)
            model.eval()
            print(f"Model loaded successfully")
            
            # Run inference
            print(f"Evaluating {model_type} model on validation data...")
            results = evaluate_validation_samples(model, validation_loader, device)
            print(f"Evaluated {len(results)} samples")
            
            # Store results
            models_results[model_type] = results
            
            # Clean up to save memory
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading or evaluating {model_type} model: {e}")
            available_models[model_type] = False
    
    # Make sure we have at least one model with results
    if not any(available_models.values()):
        print("No models were successfully loaded. Exiting.")
        return
    
    # Determine the main model to use for sample selection (prefer combined, then naip, then uavsar, then baseline)
    main_model = None
    for model_type in ['combined', 'naip', 'uavsar', 'baseline']:
        if available_models.get(model_type, False):
            main_model = model_type
            break
    
    if main_model is None:
        print("No models available for sample selection. Exiting.")
        return
    
    print(f"Using {main_model} model for sample selection")
    results = models_results[main_model]
    
    # Create different sample sets based on the main model results
    # 1. Sort by prediction loss (high to low)
    results_by_loss = sorted(results, key=lambda x: x['loss'], reverse=True)
    high_loss_samples = results_by_loss[:n_high_loss_samples]
    
    # Calculate input-to-prediction improvement ratio for all samples using the KNN implementation
    print("Calculating Chamfer distances using KNN implementation (this may take some time)...")
    try:
        # Use the KNN-based method for batch processing
        results = calculate_chamfer_with_knn(results, batch_size=50, device=device)
    except Exception as e:
        print(f"KNN-based method failed: {e}")
        print("Falling back to standard method with small batch size...")
        # Try again with smaller batch size
        results = calculate_chamfer_with_knn(results, batch_size=10, device=device)
        
    # Ensure all samples have improvement ratio calculated
    for result in results:
        # Calculate improvement ratio if not already present
        if 'improvement_ratio' not in result:
            pred_loss = result['loss']
            input_loss = result.get('input_loss', 1.0)
            
            # Handle edge cases
            if np.isnan(pred_loss) or np.isinf(pred_loss):
                pred_loss = 32000.0
            
            if input_loss > 0 and pred_loss > 0:
                result['improvement_ratio'] = input_loss / pred_loss
            else:
                result['improvement_ratio'] = 1.0
    
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
    report_path = os.path.join(output_dir, f"multi_model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    print(f"Generating report to {report_path}...")
    with PdfPages(report_path) as pdf:
        # Generate overview pages
        generate_overview_pages(pdf, results, model_paths, dpi)
        
        # Generate high loss sample pages
        if n_high_loss_samples > 0 and high_loss_samples:
            print(f"Adding {len(high_loss_samples)} highest loss samples...")
            for i, sample in enumerate(tqdm(high_loss_samples)):
                generate_multi_model_sample_page(
                    pdf, 
                    sample, 
                    i, 
                    "high-loss", 
                    models_results, 
                    available_models,
                    naip_norm_stats=naip_norm_stats,
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha
                )
        
        # Generate low improvement sample pages
        if n_low_improvement_samples > 0 and low_improvement_samples:
            print(f"Adding {len(low_improvement_samples)} lowest improvement samples...")
            for i, sample in enumerate(tqdm(low_improvement_samples)):
                generate_multi_model_sample_page(
                    pdf, 
                    sample, 
                    i, 
                    "low-improvement", 
                    models_results, 
                    available_models,
                    naip_norm_stats=naip_norm_stats,
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha
                )
        
        # Generate high improvement sample pages
        if n_high_improvement_samples > 0 and high_improvement_samples:
            print(f"Adding {len(high_improvement_samples)} highest improvement samples...")
            for i, sample in enumerate(tqdm(high_improvement_samples)):
                generate_multi_model_sample_page(
                    pdf, 
                    sample, 
                    i, 
                    "high-improvement", 
                    models_results, 
                    available_models,
                    naip_norm_stats=naip_norm_stats,
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha
                )
        
        # Generate random sample pages
        if n_random_samples > 0 and random_samples:
            print(f"Adding {len(random_samples)} random samples...")
            for i, sample in enumerate(tqdm(random_samples)):
                generate_multi_model_sample_page(
                    pdf, 
                    sample, 
                    i, 
                    "random", 
                    models_results, 
                    available_models,
                    naip_norm_stats=naip_norm_stats,
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha
                )
    
    print(f"Report saved to {report_path}")

def generate_overview_pages(pdf, results, model_paths, dpi=150):
    """
    Generate overview pages with performance metrics and plots.
    """
    # Extract losses
    losses = [result['loss'] for result in results]
    
    # Create a figure for the first overview page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Multi-Model Evaluation Report", fontsize=16)
    
    # Add report generation time
    fig.text(0.5, 0.94, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             ha='center', fontsize=12)
    
    # Add model details
    model_paths_text = "\n".join([
        f"{model_type.capitalize()}: {os.path.basename(path) if path else 'Not used'}" 
        for model_type, path in model_paths.items()
    ])
    
    fig.text(0.5, 0.85, "Models used in this report:", ha='center', fontsize=12)
    fig.text(0.5, 0.8, model_paths_text, ha='center', fontsize=10)
    
    # Add overall statistics based on the main model
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
    
    fig.text(0.5, 0.6, "Main Model Statistics:", ha='center', fontsize=12)
    fig.text(0.5, 0.5, stats_text, ha='center', fontsize=10)
    
    # Add the figure to the PDF with specified DPI
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    
    # Create a second overview page with distribution plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("Loss Distribution (Main Model)", fontsize=16)
    
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
    if len(sorted_losses) >= 30:
        threshold = sorted_losses[-30]
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                    label=f'Top 30 threshold: {threshold:.6f}')
        ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

def generate_multi_model_sample_page(
    pdf, 
    sample, 
    index, 
    sample_type, 
    models_results, 
    available_models,
    dpi=150, 
    point_size=1,
    point_alpha=0.5,
    naip_norm_stats=None
):
    """
    Generate a PDF page showing multiple model predictions for a single sample.
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF document
    sample : dict
        Sample data from the main model
    index : int
        Sample index in the category
    sample_type : str
        Type of sample ('high-loss', 'low-improvement', 'high-improvement', 'random')
    models_results : dict
        Dictionary of results from different models
    available_models : dict
        Dictionary indicating which models are available
    dpi : int
        DPI for output
    point_size : float
        Size of points in scatter plots
    point_alpha : float
        Alpha (transparency) of points
    naip_norm_stats : dict
        NAIP normalization statistics
    """
    # Extract sample index to locate the same sample in other models' results
    sample_idx = sample['sample_idx']
    tile_id = sample['tile_id']
    
    # Extract data for the current sample
    dep_points = sample['dep_points'].numpy() if isinstance(sample['dep_points'], torch.Tensor) else sample['dep_points']
    uav_points = sample['uav_points'].numpy() if isinstance(sample['uav_points'], torch.Tensor) else sample['uav_points']
    
    # Find this sample in other models' results
    model_predictions = {}
    model_losses = {}
    
    for model_type, model_available in available_models.items():
        if not model_available:
            continue
        
        # Find the sample in this model's results
        model_sample = next((s for s in models_results[model_type] if s['sample_idx'] == sample_idx), None)
        
        if model_sample:
            # Extract prediction points
            pred_points = model_sample['pred_points'].numpy() if isinstance(model_sample['pred_points'], torch.Tensor) else model_sample['pred_points']
            
            # Calculate Chamfer distance to ground truth
            try:
                chamfer_dist = calculate_chamfer_distance(pred_points, uav_points)
            except Exception as e:
                print(f"Error calculating Chamfer distance for {model_type} model: {e}")
                chamfer_dist = model_sample.get('loss', 0.0)
            
            # Store prediction and loss
            model_predictions[model_type] = pred_points
            model_losses[model_type] = chamfer_dist
    
    # Create a figure for point cloud visualizations
    fig = plt.figure(figsize=(16, 9))
    
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
    
    # Define standard colormap for point clouds
    colors = {
        'dep': 'blue',
        'baseline': 'purple',
        'naip': 'orange',
        'uavsar': 'cyan',
        'combined': 'red',
        'uav': 'green'
    }
    
    # Define plot titles and data
    plot_configs = [
        {'title': '3DEP Input (2014-2016)', 'points': dep_points, 'color': colors['dep']},
    ]
    
    # Add model plots in order: baseline, naip, uavsar, combined
    for model_type in ['baseline', 'naip', 'uavsar', 'combined']:
        if model_type in model_predictions:
            plot_configs.append({
                'title': f"{model_type.upper()} Model Prediction",
                'points': model_predictions[model_type],
                'color': colors[model_type],
                'loss': model_losses[model_type],
                'num_points': len(model_predictions[model_type])
            })
    
    # Add ground truth as the last plot
    plot_configs.append({
        'title': 'UAV Ground Truth (2023-2024)', 
        'points': uav_points, 
        'color': colors['uav'],
        'num_points': len(uav_points)
    })
    
    # Determine global axis limits
    all_points = [config['points'] for config in plot_configs]
    min_x = min([np.min(points[:, 0]) for points in all_points])
    max_x = max([np.max(points[:, 0]) for points in all_points])
    min_y = min([np.min(points[:, 1]) for points in all_points])
    max_y = max([np.max(points[:, 1]) for points in all_points])
    min_z = min([np.min(points[:, 2]) for points in all_points])
    max_z = max([np.max(points[:, 2]) for points in all_points])
    
    # Add some padding to the limits
    padding = 0.05
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    min_x -= x_range * padding
    max_x += x_range * padding
    min_y -= y_range * padding
    max_y += y_range * padding
    min_z -= z_range * padding
    max_z += z_range * padding
    
    # Create point cloud plots in a single row
    num_plots = len(plot_configs)
    axes = []
    
    for i, config in enumerate(plot_configs):
        ax = fig.add_subplot(1, num_plots, i+1, projection='3d')
        axes.append(ax)
        
        points = config['points']
        color = config['color']
        
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], 
            s=point_size, alpha=point_alpha, c=color, 
            rasterized=True
        )
        
        # Create title with number of points and distance if available
        title_parts = [config['title']]
        title_parts.append(f"N={len(points)}")
        
        if 'loss' in config and config['loss'] is not None:
            title_parts.append(f"CD={config['loss']:.4f}")
        
        ax.set_title('\n'.join(title_parts), fontsize=10)
        
        # Set consistent axis limits for all plots
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 0.5])
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Add the figure to the PDF with specified DPI for rasterization
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

    # Check if we have NAIP data and plot it
    naip_data = sample.get('naip')
    if naip_data and 'images' in naip_data and naip_data['images'] is not None:
        try:
            # Get the bounding box
            bbox = sample.get('bbox')
            
            # Create NAIP visualization with normalization stats
            naip_title = f"{title} - NAIP Imagery"
            
            # Adjust figure size based on number of NAIP images
            n_images = naip_data['images'].shape[0]
            naip_figsize = (min(16, n_images * 4), 4)
            
            naip_fig = plot_naip_imagery_row(
                naip_data, 
                naip_norm_stats=naip_norm_stats,
                bbox_overlay=bbox if bbox is not None else None,
                figsize=naip_figsize,
                title=naip_title
            )
            
            # Add the figure to the PDF with specified DPI
            pdf.savefig(naip_fig, dpi=dpi)
            plt.close(naip_fig)
            
        except Exception as e:
            print(f"Error plotting NAIP imagery for sample {tile_id}: {e}")
            # Create a simple error page
            fig = plt.figure(figsize=(11, 4))
            fig.suptitle(f"{title} - NAIP Imagery (Error)", fontsize=14)
            fig.text(0.5, 0.5, f"Error plotting NAIP imagery: {e}", 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)  



if __name__ == "__main__":

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
        naip_dropout=0.05,
        uavsar_dropout=0.05,
        temporal_encoder='gru',  # Type of temporal encoder
        
        # Fusion parameters
        fusion_type='cross_attention',
        max_dist_ratio=8,

        # Cross attention fusion parameters
        fusion_dropout=0.05,
        fusion_num_heads=4,
        position_encoding_dim=36,
        
        # Other parameters
        attr_dim=3,
    )

    # Example usage
    model_paths = {
        'combined': "/home/jovyan/geoai_veg_map/data/output/checkpoints/0426_e250_1e3lr_1e3wd_b10_tau030_naip_uavsar_k16_f256_b10_e250.pth",  # NAIP+UAVSAR model
        'naip': None,          # NAIP-only model
        'uavsar': None, # '/home/jovyan/geoai_veg_map/data/output/checkpoints/0423_e110_4e-4lr_b10_uavsar_k16_f256_b10_e110.pth',      # UAVSAR-only model
        'baseline': "/home/jovyan/geoai_veg_map/data/output/checkpoints/0422_e220_4e-4_b10_32bit_infocd_med_repul_adamw_1e-2_baseline_k16_f256_b10_e220.pth",  # Baseline model (no imagery)
    }
    
    validation_data_path = "data/processed/model_data/precomputed_validation_tiles_32bit.pt"
    output_dir = "data/output/reports"
    
    # Generate the report
    generate_multi_model_report(
        model_paths=model_paths,
        validation_data_path=validation_data_path,
        output_dir=output_dir,
        n_high_loss_samples=40,
        n_low_improvement_samples=30,
        n_high_improvement_samples=50,
        n_random_samples=50,
        dpi=150,
        point_size=1.0,
        point_alpha=0.5
    )

