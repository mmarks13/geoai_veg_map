import geopandas as gpd
import pandas as pd
import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from tqdm import tqdm
import gc  # For garbage collection
from scipy.stats import scoreatpercentile
from pytorch3d.ops import knn_points   

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Define the MultimodalModelConfig class
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


_CACHED_DATA = {}  # Global cache dictionary

def load_data_cached(data_path, device='cpu'):
    """
    Load data from a file with caching to avoid repeated disk reads.
    
    Args:
        data_path: Path to the data file
        device: Device to load the data to
        
    Returns:
        Loaded data
    """
    global _CACHED_DATA
    
    # Check if data is already in cache
    if data_path in _CACHED_DATA:
        print(f"Using cached data for: {data_path}")
        return _CACHED_DATA[data_path]
    
    # Load data from disk
    print(f"Loading data from: {data_path}")
    try:
        data = torch.load(data_path, map_location=device)
        print(f"Data loaded successfully with {len(data)} samples")
        
        # Store in cache
        _CACHED_DATA[data_path] = data
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_canopy_height_metrics(input_points, gt_points):
    """
    Calculate canopy height metrics using a 2m x 2m grid.
    
    Args:
        input_points: Numpy array of input point cloud
        gt_points: Numpy array of ground truth point cloud
        
    Returns:
        Tuple of (input_mean_pct95_z, gt_mean_pct95_z, net_canopy_height_change)
    """
    import numpy as np
    
    # Define grid parameters - grid from -5 to 5 in both x and y
    grid_min = -5
    grid_max = 5
    cell_size = 2  # 2m x 2m grid
    
    # Calculate number of grid cells
    grid_range = grid_max - grid_min
    n_cells = int(np.ceil(grid_range / cell_size))
    
    # Function to process a single point cloud
    def process_point_cloud(points):
        # Remove points with z > 50
        valid_points = points[points[:, 2] <= 50]
        
        # Create empty grid to store 95th percentile values
        grid_values = np.full((n_cells, n_cells), np.nan)
        
        # Calculate grid cell indices for each point
        x_indices = np.floor((valid_points[:, 0] - grid_min) / cell_size).astype(int)
        y_indices = np.floor((valid_points[:, 1] - grid_min) / cell_size).astype(int)
        
        # Filter out points outside the grid range
        valid_indices = (x_indices >= 0) & (x_indices < n_cells) & (y_indices >= 0) & (y_indices < n_cells)
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        z_values = valid_points[valid_indices, 2]
        
        # Calculate 95th percentile for each grid cell
        for i in range(n_cells):
            for j in range(n_cells):
                cell_z_values = z_values[(x_indices == i) & (y_indices == j)]
                if len(cell_z_values) > 0:
                    grid_values[i, j] = np.percentile(cell_z_values, 95)
        
        # Calculate mean of 95th percentile values (ignoring NaN cells)
        mean_pct95_z = np.nanmean(grid_values)
        return mean_pct95_z
    
    # Process both point clouds
    input_mean_pct95_z = process_point_cloud(input_points)
    gt_mean_pct95_z = process_point_cloud(gt_points)
    
    # Calculate net canopy height change
    net_canopy_height_change = gt_mean_pct95_z - input_mean_pct95_z
    
    return input_mean_pct95_z, gt_mean_pct95_z, net_canopy_height_change


# ---------- helper: one-sided classic Chamfer  -------------------
def _one_sided_cd(x, y, len_x=None, len_y=None):
    nn  = knn_points(x, y, K=1, lengths1=len_x, lengths2=len_y)
    d   = nn.dists.squeeze(-1)                # (B,N)
    if len_x is None:
        return d.mean(1)
    out = torch.zeros(x.shape[0], device=x.device)
    for b, L in enumerate(len_x):
        out[b] = d[b, :L].mean()
    return out                                # (B,)


# New implementation of density-aware Chamfer distance
def density_aware_chamfer_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 1.0,
    n_lambda: float = 1.0,
    eps: float = 1e-12
):
    """
    Symmetric Density-aware Chamfer Distance (DCD) using DCD-E style weighting. Equation 7
    Uses L2 norm (non-squared) distances in the exponent.
    The DCD-E style weighting ensures the factor multiplying the exp_term is <= 1,
    making (1 - factor * exp_term) naturally >= 0.

    Args:
        pred: Predicted point cloud, shape (B, N_pred, 3).
        gt: Ground truth point cloud, shape (B, N_gt, 3).
        alpha: Temperature scalar. For initial large distances (e.g., CD ~2.0),
               try alpha in [0.5, 1.5]. As CD improves, alpha can be increased.
        n_lambda: Exponent for the query count term (n_query^n_lambda).
                  Default is 1.0.
        eps: Small epsilon value for numerical stability.

    Returns:
        The mean DCD loss over the batch.
    """
    B, N_pred, _ = pred.shape
    _, N_gt, _ = gt.shape

    # --- Term 1: gt → pred (Paper: S2 -> S1, where S1=pred, S2=gt) ---
    nn_gt_to_pred = knn_points(gt, pred, K=1, return_nn=False)
    dist_gt_to_pred_sq = nn_gt_to_pred.dists[..., 0]
    dist_gt_to_pred = torch.sqrt(dist_gt_to_pred_sq.clamp_min(eps))
    idx_pred_for_gt = nn_gt_to_pred.idx[..., 0]

    exp_gt_to_pred = torch.exp(-alpha * dist_gt_to_pred)

    # --- Term 2: pred → gt (Paper: S1 -> S2, where S1=pred, S2=gt) ---
    nn_pred_to_gt = knn_points(pred, gt, K=1, return_nn=False)
    dist_pred_to_gt_sq = nn_pred_to_gt.dists[..., 0]
    dist_pred_to_gt = torch.sqrt(dist_pred_to_gt_sq.clamp_min(eps))
    idx_gt_for_pred = nn_pred_to_gt.idx[..., 0]

    exp_pred_to_gt = torch.exp(-alpha * dist_pred_to_gt)

    losses_batch = []
    for b in range(B):
        # Term 1: gt -> pred (S2 -> S1)
        # Here, effectively S_a = gt, S_b = pred for the DCD-E weight formula
        if N_gt > 0 and N_pred > 0:
            eta_gt_pred = float(N_gt) / float(N_pred) # |S_a| / |S_b|

            # n_x_hat: For each y in gt, its NN in pred is x_hat.
            # n_x_hat_counts[k] = how many times pred_point[k] is an NN for gt_points
            n_x_hat_counts = torch.bincount(idx_pred_for_gt[b], minlength=N_pred).float()
            # For each y_i in gt, get n_x_hat for its corresponding x_hat_i
            n_query_count_gt_side = n_x_hat_counts[idx_pred_for_gt[b]].pow(n_lambda) # This is n_q in DCD-E

            # DCD-E weight: 1.0 / max(eta_ab / n_q, 1.0)
            weight_factor_gt_pred = 1.0 / (torch.maximum(eta_gt_pred / (n_query_count_gt_side + eps), torch.tensor(1.0, device=pred.device)) + eps)

            loss_gt_side_points = (1.0 - exp_gt_to_pred[b] * weight_factor_gt_pred)
            # No clamp_min(0.0) needed due to DCD-E weight ensuring factor <=1
            loss_gt_side = loss_gt_side_points.mean()
        elif N_gt == 0 and N_pred > 0: # gt is empty, pred is not, high penalty for gt side
            loss_gt_side = torch.tensor(1.0, device=pred.device)
        else: # both empty or N_gt > 0, N_pred == 0 (covered by pred_side) or pred_side handles N_pred >0, N_gt==0
            loss_gt_side = torch.tensor(0.0, device=pred.device)


        # Term 2: pred -> gt (S1 -> S2)
        # Here, S_a = pred, S_b = gt for the DCD-E weight formula
        if N_pred > 0 and N_gt > 0:
            eta_pred_gt = float(N_pred) / float(N_gt) # |S_a| / |S_b|

            # n_y_hat: For each x in pred, its NN in gt is y_hat.
            # n_y_hat_counts[k] = how many times gt_point[k] is an NN for pred_points
            n_y_hat_counts = torch.bincount(idx_gt_for_pred[b], minlength=N_gt).float()
            # For each x_i in pred, get n_y_hat for its corresponding y_hat_i
            n_query_count_pred_side = n_y_hat_counts[idx_gt_for_pred[b]].pow(n_lambda) # This is n_q

            weight_factor_pred_gt = 1.0 / (torch.maximum(eta_pred_gt / (n_query_count_pred_side + eps), torch.tensor(1.0, device=pred.device)) + eps)

            loss_pred_side_points = (1.0 - exp_pred_to_gt[b] * weight_factor_pred_gt)
            loss_pred_side = loss_pred_side_points.mean()
        elif N_pred == 0 and N_gt > 0: # pred is empty, gt is not, high penalty for pred side
            loss_pred_side = torch.tensor(1.0, device=pred.device)
        else: # both empty or N_pred > 0, N_gt == 0 (covered by gt_side)
            loss_pred_side = torch.tensor(0.0, device=pred.device)

        losses_batch.append(0.5 * (loss_pred_side + loss_gt_side))

    return torch.stack(losses_batch).mean() if B > 0 and len(losses_batch) > 0 else torch.tensor(0.0, device=pred.device)


def load_geojson_tiles(geojson_path):
    """
    Load tile geometry and properties from a GeoJSON file.
    
    Args:
        geojson_path: Path to the GeoJSON file
        
    Returns:
        GeoDataFrame with tile geometries and properties
    """
    print(f"Loading tile geometries from {geojson_path}")
    try:
        # Load the GeoJSON file
        gdf = gpd.read_file(geojson_path)
        
        # Set the correct CRS explicitly
        gdf.set_crs(epsg=32611, inplace=True, allow_override=True)
        
        # Create tile_id by adding "tile_" prefix to the index
        gdf['tile_id'] = 'tile_' + gdf.index.astype(str)
        
        # Keep only necessary columns
        if 'filename' in gdf.columns:
            gdf = gdf[['tile_id', 'filename', 'geometry']]
        else:
            # Look for filename in properties
            if 'properties' in gdf.columns and isinstance(gdf.iloc[0]['properties'], dict):
                gdf['filename'] = gdf['properties'].apply(lambda p: p.get('filename', ''))
                gdf = gdf[['tile_id', 'filename', 'geometry']]
            else:
                print("Warning: No filename column found in GeoJSON")
                gdf = gdf[['tile_id', 'geometry']]
        
        print(f"Loaded {len(gdf)} tile geometries with CRS: {gdf.crs}")
        return gdf
    
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None



class RawDataset(Dataset):
    """Dataset wrapper for a list of samples."""
    def __init__(self, samples, k, max_samples=None):
        """
        Initialize the dataset.
        
        Args:
            samples: List of sample dictionaries
            k: Number of nearest neighbors for KNN
            max_samples: Maximum number of samples to include (for testing)
        """
        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples > 0:
            self.samples = samples[:max_samples]
            print(f"Using only the first {len(self.samples)} samples for testing")
        else:
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
    """Custom collate function for batches with variable-sized point clouds and imagery."""
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
    """Load a trained model from checkpoint."""
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

def process_sample_inference(model, sample_data, device):
    """
    Process a single sample through the model and return only the predictions.
    No Chamfer distance calculation is performed here.
    """
    dep_points, uav_points, e_idx, dep_attr, naip_data, uavsar_data, center, scale, bbox, tile_id = sample_data
    
    # Move data to device
    dep_points = dep_points.to(device)
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

    return pred_points


def evaluate_samples(model, dataloader, device):
    """
    Run inference on samples without calculating Chamfer distance.
    Just collect the predictions to be processed in batch later.
    """
    model.eval()  # Set model to evaluation mode
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list = batch
            
            for i in range(len(dep_list)):
                sample_data = (
                    dep_list[i], uav_list[i], edge_list[i], attr_list[i],
                    naip_list[i], uavsar_list[i], center_list[i], scale_list[i],
                    bbox_list[i], tile_id_list[i]
                )
                
                # Process sample and get predictions
                try:
                    pred_points = process_sample_inference(model, sample_data, device)
                
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
                        # Store image data
                        'naip_data': naip_list[i],
                        'uavsar_data': uavsar_list[i],
                    }
                    
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample {batch_idx * len(dep_list) + i}: {e}")
                    # Skip this sample
                    continue
    
    return results


def calculate_batch_infocd(point_sets, batch_size=50, device=None, tau=0.3):
    """
    Calculate InfoCD (Chamfer Distance with contrastive learning) for a list of point sets against ground truth.
    
    Args:
        point_sets: List of dictionaries, each containing 'pred_points' and 'gt_points'
        batch_size: Size of batch to process at once
        device: Device to run computation on
        tau: Temperature parameter for InfoCD
        
    Returns:
        List of InfoCD values
    """
    # Import the InfoCD loss function
    try:
        from src.utils.infocd import info_cd_loss
    except ImportError:
        print("ERROR: Unable to import info_cd_loss. Make sure the src module is in your Python path.")
        return [float('nan')] * len(point_sets)
    
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    total_count = len(point_sets)
    
    # Process in batches
    for i in range(0, total_count, batch_size):
        # Extract current batch
        end_idx = min(i + batch_size, total_count)
        current_batch = point_sets[i:end_idx]
        current_batch_size = len(current_batch)
        
        # Check if all point clouds have same size for true batching
        pred_sizes = [p['pred_points'].shape[0] for p in current_batch]
        gt_sizes = [p['gt_points'].shape[0] for p in current_batch]
        
        uniform_pred_size = all(size == pred_sizes[0] for size in pred_sizes)
        uniform_gt_size = all(size == gt_sizes[0] for size in gt_sizes)
        
        try:
            if uniform_pred_size and uniform_gt_size and batch_size > 1:
                # Use true batching
                pred_points_list = []
                gt_points_list = []
                
                for pair in current_batch:
                    # Convert to tensors if needed
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    pred_points_list.append(pred_points)
                    gt_points_list.append(gt_points)
                
                # Stack along batch dimension
                batch_pred_points = torch.stack(pred_points_list).to(device)
                batch_gt_points = torch.stack(gt_points_list).to(device)
                
                # Prepare lengths for InfoCD
                pred_lengths = torch.tensor([p.shape[0] for p in pred_points_list], device=device)
                gt_lengths = torch.tensor([p.shape[0] for p in gt_points_list], device=device)
                
                # Calculate lambda as per the example: λ = k / |Y|
                lam = 3.0 / gt_lengths.float()
                
                # Calculate InfoCD
                infocd_values = []
                for j in range(current_batch_size):
                    # Extract single items from batch for individual processing
                    # This is more reliable than trying to process the whole batch at once with InfoCD
                    pred_item = batch_pred_points[j].unsqueeze(0)
                    gt_item = batch_gt_points[j].unsqueeze(0)
                    pred_len = pred_lengths[j].unsqueeze(0)
                    gt_len = gt_lengths[j].unsqueeze(0)
                    l = lam[j]
                    
                    # Calculate InfoCD
                    infocd = info_cd_loss(
                        pred_item, gt_item,
                        x_lengths=pred_len,
                        y_lengths=gt_len,
                        tau=tau,
                        lam=l
                    )
                    infocd_values.append(infocd.item())
                
                # Add results
                results.extend(infocd_values)
                
            else:
                # Use individual processing for variable-sized point clouds
                for pair in current_batch:
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    # Move to device
                    pred_points = pred_points.to(device)
                    gt_points = gt_points.to(device)
                    
                    # Add batch dimension
                    pred_points = pred_points.unsqueeze(0)  # (1, N, 3)
                    gt_points = gt_points.unsqueeze(0)  # (1, M, 3)
                    
                    # Prepare lengths
                    pred_length = torch.tensor([pred_points.shape[1]], device=device)
                    gt_length = torch.tensor([gt_points.shape[1]], device=device)
                    
                    # Calculate lambda
                    lam = 3.0 / gt_length.float()
                    
                    # Calculate InfoCD
                    infocd = info_cd_loss(
                        pred_points, gt_points,
                        x_lengths=pred_length,
                        y_lengths=gt_length,
                        tau=tau,
                        lam=lam
                    )
                    
                    # Add result
                    results.append(infocd.item())
                
        except Exception as e:
            print(f"Error in batch InfoCD calculation: {e}")
            print("Falling back to individual processing for this batch")
            
            # Process each pair individually with try/except
            for pair in current_batch:
                try:
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    # Move to device
                    pred_points = pred_points.to(device)
                    gt_points = gt_points.to(device)
                    
                    # Add batch dimension
                    pred_points = pred_points.unsqueeze(0)  # (1, N, 3)
                    gt_points = gt_points.unsqueeze(0)  # (1, M, 3)
                    
                    # Prepare lengths
                    pred_length = torch.tensor([pred_points.shape[1]], device=device)
                    gt_length = torch.tensor([gt_points.shape[1]], device=device)
                    
                    # Calculate lambda
                    lam = 3.0 / gt_length.float()
                    
                    # Calculate InfoCD
                    infocd = info_cd_loss(
                        pred_points, gt_points,
                        x_lengths=pred_length,
                        y_lengths=gt_length,
                        tau=tau,
                        lam=lam
                    )
                    
                    # Add result
                    results.append(infocd.item())
                    
                except Exception as inner_e:
                    print(f"Error in individual InfoCD calculation: {inner_e}")
                    results.append(float('nan'))
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print progress
        print(f"Processed InfoCD for {end_idx}/{total_count} samples ({end_idx/total_count*100:.1f}%)")
    
    return results


# Modified function to use the new density_aware_chamfer_loss implementation
def calculate_batch_distances(point_sets, batch_size=50, device=None, dcd_alpha=1.0, dcd_n_lambda=1.0, eps=1e-12):
    """
    Calculate both standard Chamfer Distance and Density-Aware Chamfer Distance.
    
    Args:
        point_sets: List of dictionaries, each containing 'pred_points' and 'gt_points'
        batch_size: Size of batch to process at once
        device: Device to run computation on
        dcd_alpha: Temperature scalar for DCD
        dcd_n_lambda: Exponent for the query count term
        eps: Small epsilon value for numerical stability
        
    Returns:
        Tuple of (cd_results, dcd_results) - lists of distance values
    """
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cd_results, dcd_results = [], []
    total_count = len(point_sets)
    
    # Process in batches
    for i in range(0, total_count, batch_size):
        # Extract current batch
        end_idx = min(i + batch_size, total_count)
        current_batch = point_sets[i:end_idx]
        current_batch_size = len(current_batch)
        
        # Check if all point clouds have same size for true batching
        pred_sizes = [p['pred_points'].shape[0] for p in current_batch]
        gt_sizes = [p['gt_points'].shape[0] for p in current_batch]
        
        uniform_pred_size = all(size == pred_sizes[0] for size in pred_sizes)
        uniform_gt_size = all(size == gt_sizes[0] for size in gt_sizes)
        
        try:
            if uniform_pred_size and uniform_gt_size and batch_size > 1:
                # Use true batching
                pred_points_list = []
                gt_points_list = []
                
                for pair in current_batch:
                    # Convert to tensors if needed
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    pred_points_list.append(pred_points)
                    gt_points_list.append(gt_points)
                
                # Stack along batch dimension
                batch_pred_points = torch.stack(pred_points_list).to(device)
                batch_gt_points = torch.stack(gt_points_list).to(device)
                
                # Calculate standard Chamfer distance
                cd_forward = _one_sided_cd(batch_pred_points, batch_gt_points)
                cd_backward = _one_sided_cd(batch_gt_points, batch_pred_points)
                
                # Full CD is sum of both directions
                cd_values = cd_forward + cd_backward
                cd_results.extend(cd_values.tolist())
                
                # Calculate density-aware Chamfer distance using the new implementation
                dcd_values = []
                for j in range(current_batch_size):
                    # Extract single items from batch for individual processing
                    pred_item = batch_pred_points[j].unsqueeze(0)
                    gt_item = batch_gt_points[j].unsqueeze(0)
                    
                    # Calculate DCD
                    dcd = density_aware_chamfer_loss(
                        pred_item, gt_item,
                        alpha=dcd_alpha,
                        n_lambda=dcd_n_lambda,
                        eps=eps
                    )
                    dcd_values.append(dcd.item())
                
                # Add results
                dcd_results.extend(dcd_values)
                
            else:
                # Use individual processing for variable-sized point clouds
                for pair in current_batch:
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    # Move to device
                    pred_points = pred_points.to(device)
                    gt_points = gt_points.to(device)
                    
                    # Add batch dimension
                    pred_points = pred_points.unsqueeze(0)
                    gt_points = gt_points.unsqueeze(0)
                    
                    # Calculate standard Chamfer distance
                    cd_forward = _one_sided_cd(pred_points, gt_points)
                    cd_backward = _one_sided_cd(gt_points, pred_points)
                    cd_value = (cd_forward + cd_backward).item()
                    
                    # Calculate density-aware Chamfer distance using the new implementation
                    dcd_value = density_aware_chamfer_loss(
                        pred_points, gt_points,
                        alpha=dcd_alpha,
                        n_lambda=dcd_n_lambda,
                        eps=eps
                    ).item()
                    
                    # Add results
                    cd_results.append(cd_value)
                    dcd_results.append(dcd_value)
                
        except Exception as e:
            print(f"Error in batch distance calculation: {e}")
            print("Falling back to individual processing for this batch")
            
            # Process each pair individually
            for pair in current_batch:
                try:
                    pred_points = pair['pred_points']
                    gt_points = pair['gt_points']
                    
                    # Convert to tensors if needed
                    if not isinstance(pred_points, torch.Tensor):
                        pred_points = torch.tensor(pred_points, dtype=torch.float32)
                    if not isinstance(gt_points, torch.Tensor):
                        gt_points = torch.tensor(gt_points, dtype=torch.float32)
                    
                    # Move to device
                    pred_points = pred_points.to(device)
                    gt_points = gt_points.to(device)
                    
                    # Add batch dimension
                    pred_points = pred_points.unsqueeze(0)
                    gt_points = gt_points.unsqueeze(0)
                    
                    # Calculate standard CD
                    cd_forward = _one_sided_cd(pred_points, gt_points)
                    cd_backward = _one_sided_cd(gt_points, pred_points)
                    cd_value = (cd_forward + cd_backward).item()
                    
                    # Calculate DCD using the new implementation
                    dcd_value = density_aware_chamfer_loss(
                        pred_points, gt_points,
                        alpha=dcd_alpha,
                        n_lambda=dcd_n_lambda,
                        eps=eps
                    ).item()
                    
                    # Add results
                    cd_results.append(cd_value)
                    dcd_results.append(dcd_value)
                    
                except Exception as inner_e:
                    print(f"Error in individual distance calculation: {inner_e}")
                    cd_results.append(float('inf'))
                    dcd_results.append(float('inf'))
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print progress
        print(f"Processed distances for {end_idx}/{total_count} samples ({end_idx/total_count*100:.1f}%)")
    
    return cd_results, dcd_results


def create_point_cloud_dataframe(
    model_paths,
    test_data_path,
    geojson_path,
    output_path=None,
    base_config=None,
    max_samples=None,
    batch_size=50,
    inference_batch_size=1, 
    infocd_tau=0.3,
    dcd_alpha=1.0,
    dcd_n_lambda=1.0,
    eps=1e-12
):
    """
    Create a GeoDataFrame containing point cloud data, model predictions, and tile geometries.
    
    Parameters:
    -----------
    model_paths : dict
        Dictionary with paths to model checkpoints
    test_data_path : str
        Path to the test data
    geojson_path : str
        Path to the GeoJSON file with tile geometries
    output_path : str, optional
        Path to save the DataFrame as a pickle file
    base_config : MultimodalModelConfig, optional
        Base model configuration
    max_samples : int, optional
        Maximum number of samples to process (for testing)
    batch_size : int, optional
        Batch size for distance calculations
    dcd_alpha : float, optional
        Temperature scalar for DCD
    dcd_n_lambda : float, optional
        Exponent for the query count term in DCD
    eps : float, optional
        Small epsilon value for numerical stability
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing point cloud data, model predictions, and tile geometries
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set base model configuration if not provided
    if base_config is None:
        base_config = MultimodalModelConfig()
    
    # Load test data
    test_data = load_data_cached(test_data_path, device='cpu')
    if test_data is None:
        return None
    
    # Load tile geometries from GeoJSON
    tile_geometries = load_geojson_tiles(geojson_path)
    if tile_geometries is None:
        print("Warning: Proceeding without tile geometries")
    
    # Create dataset and DataLoader
    test_dataset = RawDataset(test_data, base_config.k, max_samples)
    print(f"Created test dataset with {len(test_dataset)} samples")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_batch_size,  
        shuffle=False,
        collate_fn=multimodal_variable_size_collate
    )
    
    # Load models and run inference for each model
    models_results = {}
    
    # Define model configurations for each model type
    model_configs = {
        'combined': dict(use_naip=True, use_uavsar=True),
        'naip': dict(use_naip=True, use_uavsar=False),
        'uavsar': dict(use_naip=False, use_uavsar=True),
        'baseline': dict(use_naip=False, use_uavsar=False),
        'combined_4x': dict(use_naip=True, use_uavsar=True, up_ratio=4, feature_dim = 512, img_embed_dim= 256, extractor_lcl_heads = 16),  
        'combined_6x': dict(use_naip=True, use_uavsar=True, up_ratio=6, feature_dim = 512, img_embed_dim= 256, extractor_lcl_heads = 16, expansion_lcl_heads = 16),
        'combined_8x': dict(use_naip=True, use_uavsar=True, up_ratio=8, feature_dim = 512, img_embed_dim= 192, extractor_lcl_heads = 16, expansion_lcl_heads = 16),
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
        for key, value in model_configs[model_type].items():
            setattr(model_config, key, value)
        
        # Load the model
        try:
            model = load_model(model_path, model_config)
            model.to(device)
            
            # Run inference (get predictions only, don't calculate metrics yet)
            print(f"Evaluating {model_type} model on test data...")
            results = evaluate_samples(model, test_loader, device)
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
        return None
    
    # Determine which model to use as reference
    reference_model = None
    for model_type in ['combined', 'naip', 'uavsar', 'baseline']:
        if available_models.get(model_type, False):
            reference_model = model_type
            break

    # Get set of all valid sample indices across all models
    valid_indices = set()
    for model_type, results in models_results.items():
        if available_models.get(model_type, False):
            valid_indices.update(r['sample_idx'] for r in results)

    print(f"Found {len(valid_indices)} valid unique indices")

    first_model_results = models_results[reference_model]

    # Initialize data list for DataFrame
    print("Creating DataFrame structure...")
    # Now iterate through first_model_results but avoid out-of-range indices
    data = []
    for result in first_model_results:
        sample_idx = result['sample_idx']
        
        # Skip creating a row if this index is not in all models
        if sample_idx not in valid_indices:
            continue
            
        # Rest of your row creation code using result directly
        row = {
            'sample_idx': sample_idx,
            'tile_id': result.get('tile_id', sample_idx),
            'input_points': result['dep_points'],
            'ground_truth_points': result['uav_points'],
            'input_point_count': len(result['dep_points']),
            'ground_truth_point_count': len(result['uav_points']),
            'naip_data': result.get('naip_data'),
            'uavsar_data': result.get('uavsar_data')
        }
        
        # Use the sample_idx to look up model results
        for model_type, is_available in available_models.items():
            if not is_available:
                continue
            
            model_sample = next((s for s in models_results[model_type] if s['sample_idx'] == sample_idx), None)
            if model_sample:
                row[f'{model_type}_pred_points'] = model_sample['pred_points']
                
                if 'pred_point_count' not in row:
                    row['pred_point_count'] = len(model_sample['pred_points'])
        
        data.append(row)
    
    # Now calculate all distances in batches
    print("\nCalculating distances in batches...")
    
    # Calculate input distances
    print("Calculating input distances...")
    input_cd_pairs = [
        {'pred_points': row['input_points'], 'gt_points': row['ground_truth_points']} 
        for row in data
    ]
    
    # Calculate standard Chamfer distance and DCD using the new implementation
    input_chamfer_distances, input_dcd_values = calculate_batch_distances(
        input_cd_pairs, batch_size=batch_size, device=device, 
        dcd_alpha=dcd_alpha, dcd_n_lambda=dcd_n_lambda, eps=eps
    )
    
    # Calculate input InfoCD
    print("Calculating input InfoCD...")
    input_infocd_values = calculate_batch_infocd(
        input_cd_pairs, batch_size=batch_size, device=device, tau=infocd_tau
    )
    
    # Add input distances to rows
    for i in range(len(data)):
        data[i]['input_chamfer_distance'] = input_chamfer_distances[i]
        data[i]['input_dcd'] = input_dcd_values[i]
        data[i]['input_infocd'] = input_infocd_values[i]
    
    # Calculate distances for each model
    for model_type, is_available in available_models.items():
        if not is_available:
            continue
        
        print(f"Calculating distances for {model_type} model...")
        model_cd_pairs = [
            {'pred_points': row[f'{model_type}_pred_points'], 'gt_points': row['ground_truth_points']} 
            for row in data if f'{model_type}_pred_points' in row
        ]
        
        # Calculate standard Chamfer distance and DCD using the new implementation
        model_chamfer_distances, model_dcd_values = calculate_batch_distances(
            model_cd_pairs, batch_size=batch_size, device=device,
            dcd_alpha=dcd_alpha, dcd_n_lambda=dcd_n_lambda, eps=eps
        )
        
        # Calculate InfoCD
        model_infocd_values = calculate_batch_infocd(
            model_cd_pairs, batch_size=batch_size, device=device, tau=infocd_tau
        )
        
        # Add distances to rows
        for i, (cd, dcd, infocd) in enumerate(zip(
            model_chamfer_distances, model_dcd_values, model_infocd_values
        )):
            data[i][f'{model_type}_chamfer_distance'] = cd
            data[i][f'{model_type}_dcd'] = dcd
            data[i][f'{model_type}_infocd'] = infocd
    
    # Convert point clouds to numpy arrays in the final dataframe
    print("Converting point clouds and image data to numpy arrays...")
    # Update the final data creation part
    final_data = []
    for row in data:
        final_row = {
            'sample_idx': row['sample_idx'],
            'tile_id': row['tile_id'],
            'input_point_count': row['input_point_count'],
            'ground_truth_point_count': row['ground_truth_point_count'],
            'pred_point_count': row['pred_point_count'],
            'input_chamfer_distance': row['input_chamfer_distance'],
            'input_infocd': row['input_infocd'],
            'input_chamfer_distance': row['input_chamfer_distance'],
            'input_dcd': row['input_dcd'],
            'input_infocd': row['input_infocd']
        }
        
        # Convert input and ground truth points to numpy
        input_points = row['input_points']
        gt_points = row['ground_truth_points']
        input_points_np = input_points.numpy() if isinstance(input_points, torch.Tensor) else input_points
        gt_points_np = gt_points.numpy() if isinstance(gt_points, torch.Tensor) else gt_points
        
        final_row['input_points'] = input_points_np
        final_row['ground_truth_points'] = gt_points_np
        
        # Calculate canopy height metrics
        try:
            input_mean_pct95_z, gt_mean_pct95_z, net_canopy_height_change = calculate_canopy_height_metrics(
                input_points_np, gt_points_np
            )
            final_row['input_mean_pct95_z'] = input_mean_pct95_z
            final_row['gt_mean_pct95_z'] = gt_mean_pct95_z
            final_row['net_canopy_height_change'] = net_canopy_height_change
        except Exception as e:
            print(f"Error calculating canopy height metrics for sample {row['tile_id']}: {e}")
            final_row['input_mean_pct95_z'] = np.nan
            final_row['gt_mean_pct95_z'] = np.nan
            final_row['net_canopy_height_change'] = np.nan
        
        # Add image data if available
        # Get img_bbox (same for both NAIP and UAVSAR)
        naip_data = row.get('naip_data', None)
        uavsar_data = row.get('uavsar_data', None)
        
        if naip_data and isinstance(naip_data, dict):
            # Add img_bbox
            if 'img_bbox' in naip_data and naip_data['img_bbox'] is not None:
                bbox = naip_data['img_bbox']
                # Convert to numpy if needed
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu().numpy()
                final_row['img_bbox'] = bbox
            
            # Add NAIP images
            if 'images' in naip_data and naip_data['images'] is not None:
                images = naip_data['images']
                # Convert to numpy if needed
                if isinstance(images, torch.Tensor):
                    images = images.cpu().numpy()
                final_row['naip_images'] = images
            
            # Add NAIP dates
            if 'dates' in naip_data and naip_data['dates'] is not None:
                final_row['naip_dates'] = naip_data['dates']
        
        if uavsar_data and isinstance(uavsar_data, dict):
            # Add UAVSAR images
            if 'images' in uavsar_data and uavsar_data['images'] is not None:
                images = uavsar_data['images']
                # Convert to numpy if needed
                if isinstance(images, torch.Tensor):
                    images = images.cpu().numpy()
                final_row['uavsar_images'] = images
            
            # Add UAVSAR dates
            if 'dates' in uavsar_data and uavsar_data['dates'] is not None:
                final_row['uavsar_dates'] = uavsar_data['dates']
        
        # Convert model predictions to numpy and add model Chamfer distances
        for model_type, is_available in available_models.items():
            if not is_available:
                continue
            
            if f'{model_type}_pred_points' in row:
                pred_points = row[f'{model_type}_pred_points']
                final_row[f'{model_type}_pred_points'] = pred_points.numpy() if isinstance(pred_points, torch.Tensor) else pred_points
                final_row[f'{model_type}_chamfer_distance'] = row[f'{model_type}_chamfer_distance']
                final_row[f'{model_type}_dcd'] = row[f'{model_type}_dcd']
                final_row[f'{model_type}_infocd'] = row[f'{model_type}_infocd']
        
        final_data.append(final_row)
    
    # Create DataFrame
    df = pd.DataFrame(final_data)
    
    # Join with tile geometries if available
    if tile_geometries is not None:
        print("Joining with tile geometries...")
        # Convert to GeoDataFrame by merging with tile geometries
        gdf = df.merge(tile_geometries, on='tile_id', how='left')
        
        # Check if we have missing geometries
        missing_geom_count = gdf['geometry'].isna().sum()
        if missing_geom_count > 0:
            print(f"Warning: {missing_geom_count} samples have missing geometries")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    else:
        # No geometries available, create empty GeoDataFrame
        print("Creating GeoDataFrame without geometries")
        gdf = gpd.GeoDataFrame(df)
    
    # Save GeoDataFrame if output path is provided
    if output_path:
        print(f"Saving GeoDataFrame to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_pickle(output_path)
    
    return gdf


    

if __name__ == "__main__":
    base_config = MultimodalModelConfig(
        # Core model parameters
        feature_dim=256,     # Feature dimension
        k=16,                # Number of neighbors for KNN
        up_ratio=2,          # Upsampling ratio
        pos_mlp_hdn=16,      # Hidden dimension for positional MLP
        pt_attn_dropout=0.0,
        
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
        naip_dropout=0.00,
        uavsar_dropout=0.00,
        temporal_encoder='gru',  # Type of temporal encoder
        
        # Fusion parameters
        fusion_type='cross_attention',
        max_dist_ratio=8,

        # Cross attention fusion parameters
        fusion_dropout=0.00,
        fusion_num_heads=4,
        position_encoding_dim=36,
        
        # Other parameters
        attr_dim=3,
    )

    # Define model paths
    model_paths = {
        'combined': "data/output/checkpoints/ablation_study/combined_20250512-221512/0512_ablation_study_combined_naip_uavsar_k16_f256_b15_e100.pth",  # NAIP+UAVSAR model
        'combined_8x': 'data/output/checkpoints/0529_8xUp_512ft_b1_naip_uavsar_k16_f512_b1_e30.pth',  # combined 8x model
        'naip': 'data/output/checkpoints/ablation_study/optical_only_20250512-221512/0512_ablation_study_optical_only_naip_k16_f256_b15_e100.pth',  # NAIP-only model
        'uavsar': 'data/output/checkpoints/ablation_study/sar_only_20250512-221512/0512_ablation_study_sar_only_uavsar_k16_f256_b15_e100.pth',  # UAVSAR-only model
        'baseline': "data/output/checkpoints/ablation_study/baseline_20250512-221512/0512_ablation_study_baseline_baseline_k16_f256_b15_e100.pth",  # Baseline model (no imagery)
    }
    
    # Set paths
    test_data_path = "data/processed/model_data/precomputed_test_tiles_32bit.pt"
    geojson_path = "data/processed/tiles.geojson"  # Path to the GeoJSON file
    output_path = "data/processed/model_data/point_cloud_comparison_df_0516_e100_w8x_v2.pkl"
    
    # Set to a small number for quick testing
    max_samples = None  # Set to None to process all samples 
    
    # Create the DataFrame with point cloud data and model predictions
    gdf = create_point_cloud_dataframe(
        model_paths=model_paths,
        test_data_path=test_data_path,
        geojson_path=geojson_path,
        output_path=output_path,
        base_config=base_config,
        max_samples=max_samples,
        batch_size=100,
        inference_batch_size=100,
        infocd_tau=0.3,  # Add tau parameter
        dcd_alpha=4      # Alpha for density-aware Chamfer distance
    )
    
    print(gdf.head())
    print(gdf.dtypes)


# Print GeoDataFrame info
if gdf is not None:
    print("\nGeoDataFrame created successfully!")
    print(f"Number of samples: {len(gdf)}")
    print(f"Columns: {gdf.columns.tolist()}")
    print(f"CRS: {gdf.crs}")
    
    # Print statistics for geometry availability
    if 'geometry' in gdf.columns:
        valid_geom_count = gdf.geometry.notna().sum()
        print(f"Valid geometries: {valid_geom_count}/{len(gdf)}")
    
    # Print shape/size of each object in the first row
    if len(gdf) > 0:
        print("\nFirst row object shapes:")
        first_row = gdf.iloc[0]
        for col in gdf.columns:
            value = first_row[col]
            if value is None:
                print(f"  {col}: None")
            elif isinstance(value, np.ndarray):
                print(f"  {col}: np.ndarray with shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {col}: {type(value).__name__} with length {len(value)}")
            elif hasattr(value, 'shape'):
                print(f"  {col}: {type(value).__name__} with shape {value.shape}")
            elif hasattr(value, '__len__') and not isinstance(value, (str, bytes, bytearray)):
                print(f"  {col}: {type(value).__name__} with length {len(value)}")
            else:
                print(f"  {col}: {type(value).__name__}")
    
    # Print statistics for both metrics
    cd_cols = [col for col in gdf.columns if 'chamfer_distance' in col]
    infocd_cols = [col for col in gdf.columns if 'infocd' in col]
    dcd_cols = [col for col in gdf.columns if '_dcd' in col]

       
    if cd_cols:
        print("\nChamfer Distance Statistics:")
        for col in cd_cols:
            mean_val = gdf[col].mean()
            median_val = gdf[col].median()
            print(f"  {col}:")
            print(f"    Mean: {mean_val:.6f}")
            print(f"    Median: {median_val:.6f}")
    
    if infocd_cols:
        print("\nInfoCD Statistics:")
        for col in infocd_cols:
            mean_val = gdf[col].mean()
            median_val = gdf[col].median()
            print(f"  {col}:")
            print(f"    Mean: {mean_val:.6f}")
            print(f"    Median: {median_val:.6f}")
    
    if dcd_cols:
        print("\nDensity-Aware Chamfer Distance Statistics:")
        for col in dcd_cols:
            mean_val = gdf[col].mean()
            median_val = gdf[col].median()
            print(f"  {col}:")
            print(f"    Mean: {mean_val:.6f}")
            print(f"    Median: {median_val:.6f}")