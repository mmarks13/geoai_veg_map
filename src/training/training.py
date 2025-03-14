import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from torch.utils.data import DataLoader, Dataset  
import math
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from dataclasses import dataclass, asdict
import optuna
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Project-specific imports.
from src.evaluation.inference_eval import run_inference_and_visualize_2plots
from src.models.model import NodeShufflePointUpsampler_Relative_Attn
from src.utils.chamfer_distance import chamfer_distance




# Changes to PointCloudUpsampleDataset class to use the precomputed normalized points
class PointCloudUpsampleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # Use precomputed normalized points instead of raw points
        dep_points_norm = sample['dep_points_norm']       # [N_dep, 3] - already normalized
        uav_points_norm = sample['uav_points_norm']       # [N_uav, 3] - already normalized
        edge_index = sample['dep_edge_index']   # [2, E] - precomputed in precompute_knn_inplace
        
        # No need to call normalize_pair since points are already normalized
        return (dep_points_norm, uav_points_norm, edge_index)


# Update precompute_knn_inplace to use the precomputed KNN edge indices
def precompute_knn_inplace(model_data, k=30):
    """
    Extracts the precomputed KNN graph for each sample and places it in 'dep_edge_index'
    for compatibility with existing code.
    
    model_data: list of dicts with precomputed data
    k: Number of neighbors for KNN graph.
    """
    for sample in model_data:
        # Extract the edge index for the specified k from precomputed data
        if 'knn_edge_indices' in sample and k in sample['knn_edge_indices']:
            sample['dep_edge_index'] = sample['knn_edge_indices'][k]
        else:
            # Fallback to computing KNN if precomputed data not available
            print(f"Warning: Using fallback KNN computation for sample without precomputed k={k} edge indices.")
            dep_points = sample['dep_points_norm'].contiguous()   # Use normalized points
            edge_index = knn_graph(dep_points, k=k, loop=False)
            edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))
            sample['dep_edge_index'] = edge_index


# Update create_dataloaders function (log message changed)
def create_dataloaders(train_dataset, val_dataset, k, batch_size):
    print(f"Extracting precomputed k-NN with k={k} ...")
    precompute_knn_inplace(train_dataset, k=k)
    precompute_knn_inplace(val_dataset, k=k)
    print("Extraction complete.")
    
    torch_train_data = PointCloudUpsampleDataset(train_dataset)
    torch_val_data = PointCloudUpsampleDataset(val_dataset)
    
    train_loader = DataLoader(
        torch_train_data, batch_size=batch_size, shuffle=True, collate_fn=variable_size_collate
    )
    val_loader = DataLoader(
        torch_val_data, batch_size=batch_size, shuffle=False, collate_fn=variable_size_collate
    )
    return train_loader, val_loader


def variable_size_collate(batch):
    dep_list, uav_list, edge_list = [], [], []
    for (dep_pts, uav_pts, e_idx) in batch:
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        
    return dep_list, uav_list, edge_list



def setup_logging(model_name, log_file):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger




@dataclass
class ModelConfig:
    k: int = 30
    feature_dim: int = 64
    up_ratio: int = 2
    pos_mlp_hdn: int = 32
    up_attn_hds: int = 2
    up_concat: bool = True
    up_beta: bool = False
    up_dropout: float = 0.0
    fnl_attn_hds: int = 2


def process_batch(model, batch, device):
    """
    Process a single batch through the model and compute the average loss per sample.
    """
    dep_list, uav_list, edge_list = batch
    total_loss = 0.0
    sample_count = 0
    for dep_points, uav_points, e_idx in zip(dep_list, uav_list, edge_list):
        dep_points = dep_points.to(device)
        uav_points = uav_points.to(device)
        e_idx = e_idx.to(device)
        
        # Run the model to get predicted points
        pred_points = model(dep_points, e_idx)
        
        # Add batch dimension required by chamfer_distance function
        pred_points_batch = pred_points.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        uav_points_batch = uav_points.unsqueeze(0)    # [N, 3] -> [1, N, 3]
        
        # Get point counts if needed
        pred_length = torch.tensor([pred_points.shape[0]], device=device)
        uav_length = torch.tensor([uav_points.shape[0]], device=device)
        
        # Compute chamfer distance - function returns (distance, normals)
        # We only need the distance part
        chamfer_loss, _ = chamfer_distance(
            pred_points_batch, 
            uav_points_batch,
            x_lengths=pred_length,
            y_lengths=uav_length
        )
        
        if torch.isnan(chamfer_loss):
            print(f"WARNING: Loss for a sample is NaN!")
        total_loss += chamfer_loss
        sample_count += 1
    
    # Return the average loss for the batch
    return total_loss / sample_count if sample_count > 0 else total_loss


def train_one_epoch(model, train_loader, optimizer, device, scaler):
    """
    Trains the model for one epoch and returns average training loss along with
    batch-level performance and memory usage statistics.
    
    Returns:
        avg_train_loss (float): Average loss over the epoch.
        avg_batch_time (float): Average processing time per batch (sec).
        max_batch_time (list of float): Max processing time per batch (sec).
        avg_batch_mem (float): Average peak GPU memory (GB) per batch.
        max_batch_mem (list of float): Max peak GPU memory (GB) per batch.
    """
    model.train()
    train_loss_total = 0.0
    batch_times = []
    batch_mem = []
    
    for batch in train_loader:
        # Reset the peak memory counter for this batch.
        torch.cuda.reset_peak_memory_stats(device)
        batch_start_time = time.time()
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            batch_loss = process_batch(model, batch, device)
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()
        
        batch_end_time = time.time()
        
        train_loss_total += batch_loss.item()
        # Compute elapsed time for this batch.
        elapsed = batch_end_time - batch_start_time
        batch_times.append(elapsed)
        # Record the peak GPU memory for this batch (in GB).
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024 * 1024)
        batch_mem.append(peak_mem)
    
    avg_train_loss = train_loss_total / len(train_loader)
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_batch_mem = sum(batch_mem) / len(batch_mem)
    max_batch_time = max(batch_times)
    max_batch_mem = max(batch_mem)
    
    return avg_train_loss, avg_batch_time, max_batch_time, avg_batch_mem, max_batch_mem



def validate_one_epoch(model, val_loader, device):
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            with torch.cuda.amp.autocast():
                batch_loss = process_batch(model, batch, device)
            val_loss_total += batch_loss.item()
    avg_val_loss = val_loss_total / len(val_loader)
    return avg_val_loss

# ---------------------------
# Dataclass for Model Hyperparameters
# ---------------------------
@dataclass
class ModelConfig:
    k: int = 30
    feature_dim: int = 64
    up_ratio: int = 2
    pos_mlp_hdn: int = 32
    up_attn_hds: int = 2
    up_concat: bool = True
    up_beta: bool = False
    up_dropout: float = 0.0
    fnl_attn_hds: int = 2

# ---------------------------
# create_model() Function
# ---------------------------
def create_model(device, config: ModelConfig):
    model = NodeShufflePointUpsampler_Relative_Attn(
        feat_dim=config.feature_dim,
        up_ratio=config.up_ratio,
        pos_mlp_hidden=config.pos_mlp_hdn,
        up_attn_hds=config.up_attn_hds,
        up_concat=config.up_concat,
        up_beta=config.up_beta,
        up_dropout=config.up_dropout,
        fnl_attn_hds=config.fnl_attn_hds
    ).to(device)
    return model



def count_model_parameters(model):
    """
    Returns the total number of parameters and the number of trainable parameters in the model.
    
    Args:
        model (torch.nn.Module): The model.
    
    Returns:
        total_params (int): Total number of parameters.
        trainable_params (int): Number of parameters that require gradients.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_bytes(model):
    """
    Estimates the size (in bytes) of the model parameters.
    
    Args:
        model (torch.nn.Module): The model.
    
    Returns:
        size_bytes (int): Total size in bytes of all model parameters.
    """
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return size_bytes

def format_size(size_bytes):
    """
    Converts bytes to a human-readable format in MB.
    
    Args:
        size_bytes (int): Size in bytes.
    
    Returns:
        A formatted string representing the size in megabytes.
    """
    return f"{size_bytes / (1024**2):.2f} MB"

# Example usage:
model = NodeShufflePointUpsampler_Relative_Attn(feat_dim=118, up_ratio=2, up_attn_hds=2, fnl_attn_hds=3, up_concat=True).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def train_pointcloud_upsampler(
    train_dataset,
    val_dataset,
    model_name,
    batch_size,
    checkpoint_dir,
    model_config: ModelConfig = ModelConfig(),
    num_epochs: int = 40,
    log_file: str = "logs/training_log.txt",
    early_stopping_patience: int = 10,
    print_plots_during_training: bool = True,
    train_plots_ixs = (40,46),
    val_plots_ixs = (0,1),
    optuna_trial = None
):
    logger = setup_logging(model_name, log_file)
    start_message = "Starting training with model_config: %s", asdict(model_config)
    logger.info(start_message)
    print(start_message)
    logger.info(f"Training parameters: batch_size: {batch_size}, num_epochs: {num_epochs}, early_stopping_patience: {early_stopping_patience}")
    
    # Create DataLoaders using k from model_config
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, model_config.k, batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device, model_config)
    logger.info("Model architecture:\n%s", model)
    
    total_params, trainable_params = count_model_parameters(model)
    model_size = get_model_size_bytes(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {format_size(model_size)}")
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Modified train_one_epoch now returns additional stats:
        (avg_train_loss,
         avg_batch_time, 
         max_batch_time, 
         avg_batch_mem, 
         max_batch_mem) = train_one_epoch(model, train_loader, optimizer, device, scaler)
        
        avg_val_loss = validate_one_epoch(model, val_loader, device)
        
        scheduler.step(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Construct the log message including performance and memory stats.
        # Log the computed stats.
        log_message = (f"Epoch {epoch+1}/{num_epochs} | "
                       f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.4e} | "
                       f"Time: {epoch_time:.1f}s | "
                       f"Batch Time: Avg {avg_batch_time:.3f}s, Max {max_batch_time:.3f}s | "
                       f"Memory: Avg {avg_batch_mem:.1f}GB, Max {max_batch_mem:.1f}GB")
        
        print(log_message)
        logger.info(log_message)
        
        # Optuna reporting and early stopping logic ...
        if optuna_trial is not None:
            optuna_trial.report(avg_val_loss, epoch)
            if optuna_trial.should_prune():
                logger.info("Trial pruned at epoch {}.".format(epoch))
                raise optuna.TrialPruned()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping triggered.")
            break
        
        if print_plots_during_training and ((epoch + 1) % 5 == 0 or epoch < 5):
            run_inference_and_visualize_2plots(
                model, train_dataset, index1=train_plots_ixs[0], index2=train_plots_ixs[1],
                device='cuda', width=10, height=1.2, hide_labels=True
            )
            run_inference_and_visualize_2plots(
                model, val_dataset, index1=val_plots_ixs[0], index2=val_plots_ixs[1],
                device='cuda', width=10, height=1.2, hide_labels=True
            )
        
        if (epoch + 1) % 10 == 0:
            checkpoint_name = (
                f"{model_name}_k{model_config.k}_f{model_config.feature_dim}_b{batch_size}_"
                f"e{epoch+1}.pth"
            )
            checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    if best_model_state is not None:
        final_checkpoint_name = f"{model_name}_final_best.pth"
        final_checkpoint_path = f"{checkpoint_dir}/final_best/{final_checkpoint_name}"
        torch.save(best_model_state, final_checkpoint_path)
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    else:
        final_checkpoint_path = "N/A"
    
    logger.info("Training completed.")
    

    # Write final summary log.
    summary_log_file = "logs/summary_log.txt"
    summary_logger = logging.getLogger(f"{model_name}_summary")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        sh = logging.FileHandler(summary_log_file)
        sh.setLevel(logging.INFO)
        s_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        sh.setFormatter(s_formatter)
        summary_logger.addHandler(sh)
    
    summary_logger.info("Final Model Summary:")
    summary_logger.info(f"Timestamp: {time.ctime()}")
    summary_logger.info(f"Model: {model_name}")
    summary_logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    summary_logger.info(f"Final Checkpoint Path: {final_checkpoint_path}")
    summary_logger.info(f"Epochs Completed: {epoch+1}")
    summary_logger.info(f"Model Config: {asdict(model_config)}")
    summary_logger.info("Model architecture:\n%s", model)
    
    return best_val_loss
