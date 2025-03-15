import warnings
import os

# Filter out PyTorch Geometric warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-spline-conv'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")

# Filter out other common warnings
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")

# Environment variable to suppress NCCL warnings
os.environ["NCCL_DEBUG"] = "WARN"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import json
import tempfile
from torch.utils.data import DataLoader, Dataset
import os
import sys
import pytz
import random
import socket
import numpy as np
import shutil
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass, asdict
import gc

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Project-specific imports
from src.models.model import PointUpsampler
from src.utils.chamfer_distance import chamfer_distance


def find_free_port():
    """Find a free port to use for distributed training."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def create_shards(dataset, world_size, temp_dir, prefix, measure_key='dep_points_norm'):
    """
    Split dataset into shards and save to disk.
    
    Args:
        dataset: The full dataset
        world_size: Number of processes/GPUs
        temp_dir: Directory to save shards
        prefix: Prefix for shard filenames
        measure_key: Key to measure for balancing
        
    Returns:
        List of shard filepaths
    """
    # Get sizes for balancing
    sizes = []
    for i, sample in enumerate(dataset):
        sizes.append((i, len(sample[measure_key])))
    
    # Sort by size (largest first)
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize shards
    shards = [[] for _ in range(world_size)]
    shard_sizes = [0] * world_size
    
    # Distribute samples using greedy algorithm
    for idx, size in sizes:
        # Find the shard with the least points
        min_shard = shard_sizes.index(min(shard_sizes))
        
        # Assign this sample to that shard
        shards[min_shard].append(idx)
        shard_sizes[min_shard] += size
    
    # Log balance information
    min_size = min(shard_sizes)
    max_size = max(shard_sizes)
    avg_size = sum(shard_sizes) / world_size
    print(f"Shard balance: min={min_size}, max={max_size}, avg={avg_size}, " 
          f"ratio={max_size/min_size:.2f}")
    
    # Create each shard file
    shard_paths = []
    for rank in range(world_size):
        # Create the shard data
        shard_data = [dataset[i] for i in shards[rank]]
        
        # Save to a temporary file
        shard_path = os.path.join(temp_dir, f"{prefix}_shard_{rank}.pt")
        torch.save(shard_data, shard_path,  _use_new_zipfile_serialization=False)
        shard_paths.append(shard_path)
        gc.collect()
        print(f"Created shard {rank} with {len(shard_data)} samples, saved to {shard_path}")
    
    return shard_paths


class ShardedPointCloudDataset(Dataset):
    """Dataset that loads a specific shard file."""
    def __init__(self, shard_path, k):
        """
        Initialize dataset from a shard file.
        
        Args:
            shard_path: Path to the shard file
            k: k-value for KNN
        """
        # print(f"Loading shard from {shard_path}")
        self.data = torch.load(shard_path)
        self.k = k
        # print(f"Loaded shard with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the sample
        sample = self.data[idx]
        
        # Set edge_index from precomputed knn_edge_indices
        if 'knn_edge_indices' in sample and self.k in sample['knn_edge_indices']:
            sample['dep_edge_index'] = sample['knn_edge_indices'][self.k]
        else:
            print(f"Warning: No precomputed k={self.k} edge indices for sample")
        
        # Return the sample data
        dep_points_norm = sample['dep_points_norm'] 
        uav_points_norm = sample['uav_points_norm']
        edge_index = sample['dep_edge_index']
        dep_points_attr = sample.get('dep_points_attr', None)
        
        return (dep_points_norm, uav_points_norm, edge_index, dep_points_attr)


def variable_size_collate(batch):
    dep_list, uav_list, edge_list, attr_list = [], [], [], []
    for item in batch:
        dep_pts, uav_pts, e_idx, dep_attr = item
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        attr_list.append(dep_attr)
        
    return dep_list, uav_list, edge_list, attr_list


def setup_logging(model_name, log_file):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
    attr_dim: int = 3


def process_batch(model, batch, device):
    """
    Process a single batch through the model and compute the average loss per sample.
    """
    dep_list, uav_list, edge_list, attr_list = batch
    total_loss = 0.0
    sample_count = 0
    for i in range(len(dep_list)):
        dep_points = dep_list[i].to(device)
        uav_points = uav_list[i].to(device)
        e_idx = edge_list[i].to(device)
        dep_attr = attr_list[i].to(device) if attr_list[i] is not None else None
        
        # Run the model with attributes
        pred_points = model(dep_points, e_idx, dep_attr)
        
        # Create batch tensors for chamfer distance calculation
        pred_points_batch = pred_points.unsqueeze(0)
        uav_points_batch = uav_points.unsqueeze(0)
        
        pred_length = torch.tensor([pred_points.shape[0]], device=device)
        uav_length = torch.tensor([uav_points.shape[0]], device=device)
        
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
    
    return total_loss / sample_count if sample_count > 0 else total_loss


def create_model(device, config: ModelConfig):
    model = PointUpsampler(
        feat_dim=config.feature_dim,
        up_ratio=config.up_ratio,
        pos_mlp_hidden=config.pos_mlp_hdn,
        up_attn_hds=config.up_attn_hds,
        up_concat=config.up_concat,
        up_beta=config.up_beta,
        up_dropout=config.up_dropout,
        fnl_attn_hds=config.fnl_attn_hds,
        attr_dim=config.attr_dim
    )
    model.to(device)
    return model


def count_model_parameters(model):
    """
    Returns the total number of parameters and the number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_bytes(model):
    """
    Estimates the size (in bytes) of the model parameters.
    """
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return size_bytes


def format_size(size_bytes):
    """
    Converts bytes to a human-readable format in MB.
    """
    return f"{size_bytes / (1024**2):.2f} MB"


# DDP setup and cleanup functions
def setup_ddp(rank, world_size, port):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Add a slight delay to avoid race conditions
    time.sleep(rank * 0.1)
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        return True
    except Exception as e:
        print(f"Error initializing process group on rank {rank}: {e}")
        return False


def cleanup():
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_ddp(model, train_loader, optimizer, device, scaler):
    """
    Train the model for one epoch using DDP.
    """
    model.train()
    train_loss_total = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            batch_loss = process_batch(model, batch, device)
        
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss_total += batch_loss.item()
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    train_loss_tensor = torch.tensor(train_loss_total, device=device)
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    avg_train_loss = train_loss_tensor.item() / len(train_loader) / world_size
    
    return avg_train_loss


def validate_one_epoch_ddp(model, val_loader, device):
    """
    Validate the model for one epoch using DDP.
    """
    model.eval()
    val_loss_total = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            with autocast(device_type='cuda', dtype=torch.float16):
                batch_loss = process_batch(model, batch, device)
            val_loss_total += batch_loss.item()
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    val_loss_tensor = torch.tensor(val_loss_total, device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    avg_val_loss = val_loss_tensor.item() / len(val_loader) / world_size
    
    return avg_val_loss





def _train_worker(rank, world_size, train_shard_path, val_shard_path,
                  model_name, batch_size, checkpoint_dir, model_config,
                  port, num_epochs, log_file, early_stopping_patience):
    # Setup DDP environment (works even when world_size == 1)
    setup_successful = setup_ddp(rank, world_size, port)
    if not setup_successful:
        print(f"Rank {rank}: Failed to set up DDP")
        return None

    try:
        logger = None
        if rank == 0:
            logger = setup_logging(model_name, log_file)
            logger.info(f"Starting training with model_config: {asdict(model_config)}")
            logger.info(f"Training parameters: batch_size: {batch_size}, "
                        f"num_epochs: {num_epochs}, early_stopping_patience: {early_stopping_patience}")

        device = torch.device(f'cuda:{rank}')
        # Load the shard datasets
        train_dataset_shard = ShardedPointCloudDataset(train_shard_path, model_config.k)
        val_dataset_shard = ShardedPointCloudDataset(val_shard_path, model_config.k)

        # Create samplers (if using distributed training)
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset_shard, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset_shard, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset_shard,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=variable_size_collate
        )
        val_loader = DataLoader(
            val_dataset_shard,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=variable_size_collate
        )

        # Create the model and move it to the appropriate GPU
        model = create_model(device, model_config)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        if rank == 0:
            total_params, trainable_params = count_model_parameters(model)
            model_size = get_model_size_bytes(model)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Model size: {format_size(model_size)}")

        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=(rank==0))
        scaler = GradScaler()

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            if world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if rank == 0:
                pacific_tz = pytz.timezone('America/Los_Angeles')
                current_time = datetime.now(pacific_tz)
                print(f"Epoch {epoch+1} start time (PST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            avg_train_loss = train_one_epoch_ddp(model, train_loader, optimizer, device, scaler)
            avg_val_loss = validate_one_epoch_ddp(model, val_loader, device)
            scheduler.step(avg_val_loss)
            epoch_time = time.time() - epoch_start_time

            if rank == 0:
                log_message = (f"Epoch {epoch+1}/{num_epochs} | "
                               f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                               f"LR: {optimizer.param_groups[0]['lr']:.4e} | "
                               f"Time: {epoch_time/60:.2f}min")
                print(log_message)
                logger.info(log_message)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                    logger.info(f"New best validation loss: {best_val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                    logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

                if epochs_without_improvement >= early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

                if (epoch + 1) % 10 == 0:
                    checkpoint_name = f"{model_name}_k{model_config.k}_f{model_config.feature_dim}_b{batch_size}_e{epoch+1}.pth"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
            torch.cuda.empty_cache()

        if rank == 0 and best_model_state is not None:
            final_checkpoint_dir = os.path.join(checkpoint_dir, "final_best")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            final_checkpoint_path = os.path.join(final_checkpoint_dir, f"{model_name}_final_best.pth")
            torch.save(best_model_state, final_checkpoint_path)
            logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
            best_loss_file = os.path.join(final_checkpoint_dir, f"{model_name}_best_loss.json")
            with open(best_loss_file, 'w') as f:
                json.dump({'best_val_loss': best_val_loss}, f)
            logger.info("Training completed.")
            result = best_val_loss
        else:
            result = None
    except Exception as e:
        print(f"Rank {rank}: Exception during training: {e}")
        result = None
    finally:
        cleanup()
    return result


def train_model(train_dataset,
                val_dataset,
                model_name,
                batch_size,
                checkpoint_dir,
                model_config=ModelConfig(),
                num_epochs=40,
                log_file="logs/training_log.txt",
                early_stopping_patience=10,
                temp_dir_root="data/output/tmp_shards"):
    """
    Unified training entry point that works for both single-GPU and multi-GPU (DDP) cases.
    
    Parameters:
      - train_dataset, val_dataset: Precomputed dataset objects.
      - model_name: Name of the model (used for logging and checkpointing).
      - batch_size: Batch size per GPU.
      - checkpoint_dir: Directory in which checkpoints will be saved.
      - model_config: Instance of ModelConfig with model hyperparameters.
      - num_epochs: Number of epochs to train.
      - log_file: Path for logging.
      - early_stopping_patience: Number of epochs without improvement before early stopping.
      - temp_dir_root: Base directory to use for temporary shards.
      
    Returns:
      The best validation loss (as reported by the rank-0 process), or None if training failed.
    """
    # Create required directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(temp_dir_root, exist_ok=True)

    # Create a unique temporary directory within the specified root directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(temp_dir_root, f"pointcloud_shards_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s) for training.")

    # Pre-shard the datasets to disk
    print("Creating training data shards...")
    train_shard_paths = create_shards(train_dataset, world_size, temp_dir, "train", 'dep_points_norm')
    print("Creating validation data shards...")
    val_shard_paths = create_shards(val_dataset, world_size, temp_dir, "val", 'dep_points_norm')

    # Find a free port for DDP communication
    port = find_free_port()
    print(f"Using port {port} for DDP communication")

    best_val_loss = None
    if world_size > 1:
        # Distributed training: spawn one process per GPU.
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        processes = []
        for rank in range(world_size):
            p = torch.multiprocessing.Process(
                target=_train_worker,
                args=(rank, world_size,
                      train_shard_paths[rank],
                      val_shard_paths[rank],
                      model_name,
                      batch_size,
                      checkpoint_dir,
                      model_config,
                      port,
                      num_epochs,
                      log_file,
                      early_stopping_patience)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # After distributed training, read best validation loss from the file saved by rank 0.
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_best", f"{model_name}_final_best.pth")
        best_loss_file = os.path.join(checkpoint_dir, "final_best", f"{model_name}_best_loss.json")
        if os.path.exists(final_checkpoint_path) and os.path.exists(best_loss_file):
            with open(best_loss_file, 'r') as f:
                data = json.load(f)
                best_val_loss = data.get('best_val_loss')
            print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        else:
            print("Training did not complete successfully.")
    else:
        # Single GPU training: call the worker directly.
        best_val_loss = _train_worker(0, 1,
                                      train_shard_paths[0],
                                      val_shard_paths[0],
                                      model_name,
                                      batch_size,
                                      checkpoint_dir,
                                      model_config,
                                      port,
                                      num_epochs,
                                      log_file,
                                      early_stopping_patience)
    # Clean up temporary shards directory
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

    return best_val_loss
