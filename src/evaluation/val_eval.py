import torch
import os
import sys
from dataclasses import asdict
from torch.utils.data import DataLoader, Dataset

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules
from src.models.multimodal_model import MultimodalModelConfig, MultimodalPointUpsampler
from src.training.multimodal_training import ShardedMultimodalPointCloudDataset, multimodal_variable_size_collate
from src.utils.chamfer_distance import chamfer_distance

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
        dep_points_attr = sample['dep_points_attr_norm']
        
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
                'relative_dates': sample['naip'].get('relative_dates', None) 
            }
        
        # UAVSAR data
        if 'uavsar' in sample:
            uavsar_data = {
                'images': sample['uavsar'].get('images', None),
                'img_bbox': sample['uavsar'].get('img_bbox', None),
                'relative_dates': sample['uavsar'].get('relative_dates', None) 
            }
        
        return (dep_points_norm, uav_points_norm, edge_index, dep_points_attr, naip_data, uavsar_data, center, scale, bbox, tile_id)

def load_model(model_path, config):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
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
    if naip_data is not None:
        if 'images' in naip_data and naip_data['images'] is not None:
            naip_data['images'] = naip_data['images'].to(device)
    
    if uavsar_data is not None:
        if 'images' in uavsar_data and uavsar_data['images'] is not None:
            uavsar_data['images'] = uavsar_data['images'].to(device)
    
    # Use automatic mixed precision to handle float16/float32 mismatches
    # This is important because the model was trained with mixed precision
    from torch.amp import autocast
    with autocast(device_type='cuda', dtype=torch.float16):
        # Run the model with all available data including spatial information
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
    
    # Calculate Chamfer distance loss (still within autocast context)
    try:
        from src.utils.chamfer_distance import chamfer_distance
        chamfer_loss, _ = chamfer_distance(
            pred_points_batch, 
            uav_points_batch,
            x_lengths=pred_length,
            y_lengths=uav_length
        )
    except ImportError:
        # Fallback to direct import if module structure is different
        print("Warning: Falling back to direct chamfer_distance import")
        from chamfer_distance import chamfer_distance
        chamfer_loss, _ = chamfer_distance(
            pred_points_batch, 
            uav_points_batch,
            x_lengths=pred_length,
            y_lengths=uav_length
        )
    
    if torch.isnan(chamfer_loss):
        print(f"WARNING: Loss for sample is NaN! {tile_id}")

    if torch.isinf(chamfer_loss):
        print(f"WARNING: Loss for sample is Inf! {tile_id}")
    
    return pred_points, chamfer_loss.item()

def evaluate_validation_samples(model, dataloader, device):
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
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
    
    return results

def main():
    """Main function to evaluate the model on validation data."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check CUDA capabilities and memory
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Using mixed precision (float16) for model inference")
    
    # Set model configuration
    # Note: Need to match the configuration of the trained model
    config = MultimodalModelConfig(
        # Core model parameters (retained from original)
        fnl_attn_hds=4,      # Final attention heads
        feature_dim=256,     # Feature dimension
        k=15,                # Number of neighbors for KNN
        up_attn_hds=4,       # Upsampling attention heads
        up_ratio=2,          # Upsampling ratio
        pos_mlp_hdn=32,      # Hidden dimension for positional MLP
        
        use_naip = False,
        use_uavsar = False,
        
        # Imagery encoder parameters
        img_embed_dim=64,    # Dimension of patch embeddings
        img_num_patches=16,  # Number of output patch embeddings
        naip_dropout=0.05,
        uavsar_dropout=0.05,
        temporal_encoder= 'transformer',
        
        # Fusion parameters
        fusion_type = 'cross_attention',
        max_dist_ratio=1.5,

        # cross attention fusion parameters
        fusion_dropout = 0.10,
        fusion_num_heads = 4,
        position_encoding_dim = 24,

        # Point Transformer parameters
        num_lcl_heads = 4,  # Local attention heads (for MultiHeadPointTransformerConv)
        num_glbl_heads = 4,  # Global attention heads (for PosAwareGlobalFlashAttention)
        pt_attn_dropout = 0.0
    )
    
    # Load the model
    model_path = "/home/jovyan/geoai_veg_map/data/output/checkpoints/0404_huber2m_3DEP_Only_baseline_k15_f256_b6_e40.pth"
    model = load_model(model_path, config)
    model.to(device)
    print("Model loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode")
    
    # Create validation dataset
    validation_data_path = "data/processed/model_data/precomputed_validation_tiles.pt"
    print(f"Loading validation data from: {validation_data_path}")
    
    # Load validation data
    try:
        # Try loading with weights_only=False first (the default)
        validation_data = torch.load(validation_data_path)
        print("Validation data loaded successfully")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        print("Trying alternative loading method...")
        try:
            # Try with map_location to CPU explicitly
            validation_data = torch.load(validation_data_path, map_location='cpu')
            print("Validation data loaded successfully with CPU mapping")
        except Exception as e2:
            print(f"Failed again: {e2}")
            raise RuntimeError("Could not load validation data")
    
    # Create appropriate dataset based on data type
    if isinstance(validation_data, list):
        print(f"Creating dataset from list of {len(validation_data)} samples")
        validation_dataset = RawDataset(validation_data, config.k)
    else:
        try:
            # Try to create ShardedMultimodalPointCloudDataset
            print("Creating ShardedMultimodalPointCloudDataset from validation data")
            validation_dataset = ShardedMultimodalPointCloudDataset(
                validation_data_path, 
                k=config.k, 
                use_naip=config.use_naip, 
                use_uavsar=config.use_uavsar
            )
        except Exception as e:
            print(f"Failed to create dataset: {e}. Attempting to use raw data.")
            # Fall back to treating the loaded data as a list of samples
            if hasattr(validation_data, '__len__'):
                validation_dataset = RawDataset(validation_data, config.k)
            else:
                raise ValueError(f"Unexpected validation data format: {type(validation_data)}")
    
    # Print first sample info for debugging
    first_sample = validation_dataset[0]
    print(f"First sample data types:")
    print(f"dep_points dtype: {first_sample[0].dtype}")
    print(f"uav_points dtype: {first_sample[1].dtype}")
    print(f"edge_index dtype: {first_sample[2].dtype}")
    if first_sample[3] is not None:
        print(f"dep_attr dtype: {first_sample[3].dtype}")
    print(f"tile_id: {first_sample[9]}")
    
    print(f"Created validation dataset with {len(validation_dataset)} samples")
    
    # Create DataLoader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,  # Evaluate one sample at a time
        shuffle=False,
        collate_fn=multimodal_variable_size_collate
    )
    
    # Evaluate the model
    results = evaluate_validation_samples(model, validation_loader, device)
    
    # Save results
    output_dir = "data/output/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "validation_results.pt")
    torch.save(results, output_path)
    print(f"Results saved to {output_path}")
    
    # Save losses separately for easier analysis
    losses_path = os.path.join(output_dir, "validation_losses.pt")
    losses_dict = {result['tile_id']: result['loss'] for result in results}
    torch.save(losses_dict, losses_path)
    print(f"Losses saved to {losses_path}")
    
    # Print summary statistics
    losses = [result['loss'] for result in results]
    mean_loss = sum(losses) / len(losses)
    print(f"Mean Chamfer Loss: {mean_loss:.6f}")
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    print(f"Min Loss: {min_loss:.6f}, Sample: {results[min_idx]['tile_id']}")
    max_loss = max(losses)
    max_idx = losses.index(max_loss)
    print(f"Max Loss: {max_loss:.6f}, Sample: {results[max_idx]['tile_id']}")

if __name__ == "__main__":
    main()
