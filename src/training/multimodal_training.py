import torch
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
import os
import sys
import pytz
import socket
import shutil
from datetime import datetime
import torch.distributed as dist
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import json
import time
import threading
import logging
from torch.utils.tensorboard import SummaryWriter


# Make NCCL more robust to communication issues
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800"  # in seconds

# Import from the original ddp_training.py
from src.training.ddp_training import (
    ModelConfig, 
    process_batch, 
    setup_logging, 
    setup_ddp,
    cleanup,
    find_free_port,
    count_model_parameters,
    get_model_size_bytes,
    format_size,
    monitor_gpu_stats
)

# Import our new MultimodalModelConfig and MultimodalPointUpsampler
from src.models.multimodal_model import MultimodalModelConfig, MultimodalPointUpsampler


import math
from torch.optim.lr_scheduler import _LRScheduler

class OneCycleLR(_LRScheduler):
    """
    One Cycle Learning Rate policy.
    """
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.warmup_steps = int(total_steps * pct_start)
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase from initial_lr to max_lr
            alpha = self.last_epoch / self.warmup_steps
            return [self.initial_lr + alpha * (self.max_lr - self.initial_lr) for _ in self.base_lrs]
        else:
            # Cooldown phase: cosine annealing from max_lr to min_lr
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + alpha * (self.max_lr - self.min_lr) for _ in self.base_lrs]



class ShardedMultimodalPointCloudDataset(Dataset):
    """Dataset that loads a specific shard file with support for imagery data."""
    def __init__(self, shard_path, k, use_naip=False, use_uavsar=False):
        """
        Initialize dataset from a shard file.
        
        Args:
            shard_path: Path to the shard file
            k: k-value for KNN
            use_naip: Whether to include NAIP imagery data
            use_uavsar: Whether to include UAVSAR imagery data
        """
        self.data = torch.load(shard_path)
        self.k = k
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
    
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
        
        # Return the basic sample data
        dep_points_norm = sample['dep_points_norm'] 
        uav_points_norm = sample['uav_points_norm']
        edge_index = sample['dep_edge_index']
        dep_points_attr = sample.get('dep_points_attr', None)
        
        # Extract normalization parameters
        center = sample.get('center', None)
        scale = sample.get('scale', None)
        bbox = sample.get('bbox', None)
        
        # Add imagery data if requested
        naip_data = None
        uavsar_data = None

        
        if self.use_naip and 'naip' in sample:
            naip_data = {
                'images': sample['naip'].get('images', None),
                'img_bbox': sample['naip'].get('img_bbox', None),
                'relative_dates': sample['naip'].get('relative_dates', None) 
            }
        
        if self.use_uavsar and 'uavsar' in sample:
            uavsar_data = {
                'images': sample['uavsar'].get('images', None),
                'img_bbox': sample['uavsar'].get('img_bbox', None),
                'relative_dates': sample['uavsar'].get('relative_dates', None) 
            }

        
        return (dep_points_norm, uav_points_norm, edge_index, dep_points_attr, naip_data, uavsar_data, center, scale, bbox)


def multimodal_variable_size_collate(batch):
    """
    Collate function for multimodal variable-sized point clouds.
    """
    dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list = [], [], [], [], [], []
    center_list, scale_list, bbox_list = [], [], []
    
    for item in batch:
        dep_pts, uav_pts, e_idx, dep_attr, naip_data, uavsar_data, center, scale, bbox = item
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        attr_list.append(dep_attr)
        naip_list.append(naip_data)
        uavsar_list.append(uavsar_data)
        center_list.append(center)
        scale_list.append(scale)
        bbox_list.append(bbox)
        
    return dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list


def process_multimodal_batch(model, batch, device):
    """
    Process a single multimodal batch through the model and compute the average loss per sample.
    Now includes spatial constraint information from center and scale.
    """
    dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list = batch
    total_loss = 0.0
    sample_count = 0
    
    for i in range(len(dep_list)):
        dep_points = dep_list[i].to(device)
        uav_points = uav_list[i].to(device)
        e_idx = edge_list[i].to(device)
        dep_attr = attr_list[i].to(device) if attr_list[i] is not None else None
        
        # Get normalization parameters
        center = center_list[i].to(device) if center_list[i] is not None else None
        scale = scale_list[i].to(device) if scale_list[i] is not None else None
        bbox = bbox_list[i].to(device) if bbox_list[i] is not None else None
        
        # Process imagery data if available
        naip_data = naip_list[i]
        uavsar_data = uavsar_list[i]
        
        # Move imagery data to device if available
        if naip_data is not None:
            if 'images' in naip_data and naip_data['images'] is not None:
                naip_data['images'] = naip_data['images'].to(device)
        
        if uavsar_data is not None:
            if 'images' in uavsar_data and uavsar_data['images'] is not None:
                uavsar_data['images'] = uavsar_data['images'].to(device)
        
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
        
        # Calculate Chamfer distance loss
        from src.utils.chamfer_distance import chamfer_distance
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


def create_multimodal_model(device, config: MultimodalModelConfig):
    """
    Create a multimodal point upsampling model based on configuration.
    """
    model = MultimodalPointUpsampler(config)
    model.to(device)
    return model


def create_multimodal_shards(dataset, world_size, temp_dir, prefix, 
                           use_naip=False, use_uavsar=False, measure_key='dep_points_norm'):
    """
    Split multimodal dataset into shards and save to disk.
    
    This is similar to the original create_shards function but ensures
    that imagery data is properly included in the shards.
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


def train_one_epoch_ddp(model, train_loader, optimizer, device, scaler, writer=None, epoch=0, 
                    process_batch_fn=None, accumulation_steps=4, enable_debug=False, 
                    scheduler=None):
    """
    Train the model for one epoch using DDP with TensorBoard logging and gradient accumulation.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        scaler: GradScaler for mixed precision training
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number
        process_batch_fn: Function to process a batch (defaults to process_multimodal_batch)
        accumulation_steps: Number of batches to accumulate gradients over (default: 4)
        enable_debug: Whether to enable debugging features (default: False)
        scheduler: Optional scheduler to update after each batch (for OneCycleLR)
    """
    model.train()
    train_loss_total = 0.0
    batch_count = 0
    accumulated_batch_count = 0
    
    # Use the provided batch processing function or use process_multimodal_batch by default
    if process_batch_fn is None:
        from src.training.multimodal_training import process_multimodal_batch
        process_batch_fn = process_multimodal_batch
    
    # Global step counter for tensorboard logging
    global_step = epoch * len(train_loader)
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    # Batch logging interval - only log if debugging is enabled
    batch_log_interval = 10 if enable_debug else 999999
    
    for batch_idx, batch in enumerate(train_loader):
        current_step = global_step + batch_idx
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            batch_loss = process_batch_fn(model, batch, device)
            # Scale loss by accumulation steps to maintain correct gradient magnitude
            scaled_loss = batch_loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(scaled_loss).backward()
        
        # Accumulate statistics for reporting
        train_loss_total += batch_loss.item()
        batch_count += 1
        accumulated_batch_count += 1

        # Log LR if debugging is enabled
        if writer is not None and enable_debug:
            writer.add_scalar('Metrics/learning_rate_batch', optimizer.param_groups[0]['lr'], current_step)
    
        # Log batch-level metrics if debugging is enabled
        should_log_batch = (enable_debug and 
                          batch_idx % batch_log_interval == 0 and 
                          writer is not None)
                          
        if should_log_batch:
            # Log per-batch loss
            writer.add_scalar('Loss/train_batch', batch_loss.item(), current_step)
        
        # Only update weights after accumulating gradients for specified number of steps
        # or at the end of the epoch to avoid losing the last few batches
        is_last_batch = (batch_idx == len(train_loader) - 1)
        if accumulated_batch_count == accumulation_steps or is_last_batch:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()        

            # Update learning rate scheduler if provided (for OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Reset accumulation counter
            accumulated_batch_count = 0
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    train_loss_tensor = torch.tensor(train_loss_total, device=device)
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    avg_train_loss = train_loss_tensor.item() / batch_count / world_size
    
    return avg_train_loss


def validate_one_epoch_ddp(model, val_loader, device, writer=None, epoch=0, process_batch_fn=None,
                          enable_debug=False, visualize_samples=False):
    """
    Validate the model for one epoch using DDP with TensorBoard logging.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to validate on
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number
        process_batch_fn: Function to process a batch (defaults to process_multimodal_batch)
        enable_debug: Whether to enable debugging features (default: False)
        visualize_samples: Whether to visualize sample predictions (default: False)
    """
    model.eval()
    val_loss_total = 0.0
    batch_count = 0
    
    # Use the provided batch processing function or use process_multimodal_batch by default
    if process_batch_fn is None:
        from src.training.multimodal_training import process_multimodal_batch
        process_batch_fn = process_multimodal_batch
    
    # Global step counter for tensorboard logging
    global_step = epoch * len(val_loader)
    
    # Batch logging interval - only log if debugging is enabled
    batch_log_interval = 10 if enable_debug else 999999
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            current_step = global_step + batch_idx
            
            with autocast(device_type='cuda', dtype=torch.float16):
                batch_loss = process_batch_fn(model, batch, device)
            
            # Check if we should log for this batch
            should_log_batch = (enable_debug and 
                              batch_idx % batch_log_interval == 0 and 
                              writer is not None and 
                              dist.get_rank() == 0)
            
            # Log batch-level statistics
            if should_log_batch:
                writer.add_scalar('Loss/val_batch', batch_loss.item(), current_step)
            
            val_loss_total += batch_loss.item()
            batch_count += 1
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    val_loss_tensor = torch.tensor(val_loss_total, device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    avg_val_loss = val_loss_tensor.item() / batch_count / world_size
    
    return avg_val_loss


def _train_multimodal_worker(rank, world_size, train_shard_path, val_shard_path,
                           model_name, batch_size, checkpoint_dir, model_config,
                           port, num_epochs, log_file, early_stopping_patience,
                           tensorboard_log_dir=None, enable_debug=False,
                           accumulation_steps=3, visualize_samples=False, 
                           lr_scheduler_type="plateau",
                           max_lr=5e-4, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    """
    Worker function for training the multimodal model.
    
    Args:
        enable_debug: Whether to enable debugging features (default: False)
        lr_scheduler_type: Type of learning rate scheduler to use ("plateau" or "onecycle")
        max_lr: Maximum learning rate for OneCycleLR
        pct_start: Percentage of total steps for warmup phase in OneCycleLR
        div_factor: Initial learning rate = max_lr / div_factor
        final_div_factor: Final learning rate = initial_lr / final_div_factor
    """
    # Setup DDP environment
    setup_successful = setup_ddp(rank, world_size, port)
    if not setup_successful:
        print(f"Rank {rank}: Failed to set up DDP")
        return None

    # Create TensorBoard writer for this process
    writer = None
    gpu_monitor_thread = None
    stop_monitoring = threading.Event()
    
    # Setup TensorBoard SummaryWriter for rank 0 only
    if tensorboard_log_dir is not None and rank == 0:
        tb_dir = os.path.join(tensorboard_log_dir, f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard logs will be saved to: {tb_dir}")
        
        # Log model config as text
        config_str = "\n".join([f"{k}: {v}" for k, v in asdict(model_config).items()])
        writer.add_text('Config/model_parameters', config_str, 0)
        writer.add_text('Config/training_parameters', 
                      f"batch_size: {batch_size}\nnum_epochs: {num_epochs}\n" +
                      f"early_stopping_patience: {early_stopping_patience}", 0)
        
        # Basic GPU monitoring if debugging is enabled
        if enable_debug:
            gpu_monitor_thread = threading.Thread(
                target=monitor_gpu_stats,
                args=(writer, rank, stop_monitoring, 10.0)  # 10 seconds monitoring interval
            )
            gpu_monitor_thread.daemon = True
            gpu_monitor_thread.start()
    
    logger = None
    if rank == 0:
        logger = setup_logging(model_name, log_file)
        logger.info(f"Starting training with model_config: {asdict(model_config)}")
        logger.info(f"Training parameters: batch_size: {batch_size}, "
                    f"num_epochs: {num_epochs}, early_stopping_patience: {early_stopping_patience}")
        if tensorboard_log_dir:
            logger.info(f"TensorBoard logging enabled at: {tensorboard_log_dir}")
            logger.info(f"Debug mode: {enable_debug}")

    device = torch.device(f'cuda:{rank}')
    
    # Load the shard datasets with multimodal support
    train_dataset_shard = ShardedMultimodalPointCloudDataset(
        train_shard_path, 
        model_config.k, 
        use_naip=model_config.use_naip, 
        use_uavsar=model_config.use_uavsar
    )
    
    val_dataset_shard = ShardedMultimodalPointCloudDataset(
        val_shard_path, 
        model_config.k, 
        use_naip=model_config.use_naip, 
        use_uavsar=model_config.use_uavsar
    )

    # Create samplers (if using distributed training)
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset_shard, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset_shard, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders with multimodal collate function
    train_loader = DataLoader(
        train_dataset_shard,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=multimodal_variable_size_collate
    )
    val_loader = DataLoader(
        val_dataset_shard,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=multimodal_variable_size_collate
    )

    # Create the multimodal model and move it to the appropriate GPU
    model = create_multimodal_model(device, model_config)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        total_params, trainable_params = count_model_parameters(model)
        model_size = get_model_size_bytes(model)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {format_size(model_size)}")
        
        # Log model info to TensorBoard
        if writer is not None:
            writer.add_text('Model/parameters', 
                          f"Total parameters: {total_params:,}\n" +
                          f"Trainable parameters: {trainable_params:,}\n" +
                          f"Model size: {format_size(model_size)}", 0)
            
            # Add model structure text
            model_structure = str(model.module if isinstance(model, DDP) else model)
            writer.add_text('Model/structure', f'```\n{model_structure}\n```', 0)

    # Calculate total steps for OneCycleLR
    total_steps = (len(train_loader) // accumulation_steps) * num_epochs
    
    # Initialize optimizer with appropriate initial learning rate
    initial_lr = max_lr if lr_scheduler_type == "plateau" else max_lr / div_factor
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Initialize appropriate scheduler based on type
    if lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        scheduler_batch_update = True  # Flag to indicate batch-level updates
    else:  # "plateau" - default behavior
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=5, verbose=(rank==0)
        )
        scheduler_batch_update = False  # Flag to indicate epoch-level updates

    scaler = GradScaler()

    # Log LR scheduler info
    if rank == 0 and logger is not None:
        logger.info(f"Using {lr_scheduler_type} learning rate scheduler")
        if lr_scheduler_type == "onecycle":
            logger.info(f"OneCycleLR parameters: max_lr={max_lr}, " 
                      f"pct_start={pct_start}, div_factor={div_factor}, "
                      f"final_div_factor={final_div_factor}")

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Check if rank 0 is still in the training loop
        is_running = torch.tensor(1, device=device)
        dist.broadcast(is_running, src=0)
        if is_running.item() == 0:
            print(f"Rank {rank}: Rank 0 has stopped, so stopping too.")
            break
            
        epoch_start_time = time.time()
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train_one_epoch_ddp(
            model, train_loader, optimizer, device, scaler, writer, epoch,
            process_batch_fn=process_multimodal_batch,
            accumulation_steps=accumulation_steps,
            enable_debug=enable_debug,
            scheduler=scheduler if scheduler_batch_update else None  # Pass scheduler if batch update
        )
        
        # Validate
        avg_val_loss = validate_one_epoch_ddp(
            model, val_loader, device, writer, epoch,
            process_batch_fn=process_multimodal_batch,
            enable_debug=enable_debug,
            visualize_samples=visualize_samples
        )
        
        # Update scheduler at epoch level only for ReduceLROnPlateau
        if not scheduler_batch_update:
            scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time

        # Log epoch metrics to TensorBoard
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            
            # Additional metrics if debugging is enabled
            if enable_debug:
                writer.add_scalar('Metrics/epoch_time_minutes', epoch_time/60, epoch)
                writer.add_scalar('Metrics/learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                # Track GPU memory at epoch boundaries
                allocated = torch.cuda.memory_allocated(device=device) / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved(device=device) / (1024**2)    # MB
                writer.add_scalar('Memory/allocated_MB', allocated, epoch)
                writer.add_scalar('Memory/reserved_MB', reserved, epoch)

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
                
                # Log the best model so far
                if writer is not None:
                    writer.add_scalar('Metrics/best_val_loss', best_val_loss, epoch)
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

            # Check for early stopping and inform other ranks if triggered
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                is_running = torch.tensor(0, device=device)  # We'll stop after this epoch
                if writer is not None:
                    writer.add_text('Training/early_stopping', 
                                  f"Training stopped at epoch {epoch+1} due to no improvement for {early_stopping_patience} epochs",
                                  epoch)
                break

            if (epoch + 1) % 10 == 0:
                # Use model_name directly without adding modality string again
                checkpoint_name = f"{model_name}_k{model_config.k}_f{model_config.feature_dim}_b{batch_size}_e{epoch+1}.pth"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(best_model_state, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                        
        # Clear unused GPU memory before the next epoch
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0 and best_model_state is not None:
       
        final_checkpoint_dir = os.path.join(checkpoint_dir, "final_best")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        
        # Use model_name directly without adding modality info again
        final_checkpoint_path = os.path.join(final_checkpoint_dir, f"{model_name}_final_best.pth")
        torch.save(best_model_state, final_checkpoint_path)
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
        
        # Use model_name directly for the best loss file too
        best_loss_file = os.path.join(final_checkpoint_dir, f"{model_name}_best_loss.json")
        with open(best_loss_file, 'w') as f:
            json.dump({'best_val_loss': best_val_loss}, f)
        logger.info("Training completed.")
        
        # Add final summary to TensorBoard
        if writer is not None:
            writer.add_text('Training/summary', 
                          f"Training completed with best validation loss: {best_val_loss:.6f}",
                          0)
        
        result = best_val_loss
    else:
        result = None


    # Clean up TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Stop GPU monitoring thread
    if gpu_monitor_thread is not None:
        stop_monitoring.set()
        gpu_monitor_thread.join(timeout=1.0)
        
    cleanup()
    return result


def train_multimodal_model(train_dataset,
                         val_dataset,
                         model_name,
                         batch_size,
                         checkpoint_dir,
                         model_config=MultimodalModelConfig(),  # Use MultimodalModelConfig
                         num_epochs=60,
                         log_file="logs/training_log.txt",
                         early_stopping_patience=20,
                         temp_dir_root="data/output/tmp_shards",
                         tensorboard_log_dir="data/output/tensorboard_logs",
                         enable_debug=False,
                         accumulation_steps=3,
                         visualize_samples=False,
                         lr_scheduler_type="plateau",    # New parameter
                         max_lr=5e-4,                    # New parameter
                         pct_start=0.3,                  # New parameter
                         div_factor=25.0,                # New parameter
                         final_div_factor=1e4):          # New parameter
    """
    Unified training entry point for the multimodal model.
    
    Parameters:
      - train_dataset, val_dataset: Precomputed dataset objects.
      - model_name: Name of the model (used for logging and checkpointing).
      - batch_size: Batch size per GPU.
      - checkpoint_dir: Directory in which checkpoints will be saved.
      - model_config: Instance of MultimodalModelConfig with model hyperparameters.
      - num_epochs: Number of epochs to train.
      - log_file: Path for logging.
      - early_stopping_patience: Number of epochs without improvement before early stopping.
      - temp_dir_root: Base directory to use for temporary shards.
      - tensorboard_log_dir: Directory to store TensorBoard logs
      - enable_debug: Whether to enable debugging features
      - visualize_samples: Whether to visualize sample predictions
      - lr_scheduler_type: Type of learning rate scheduler to use ("plateau" or "onecycle")
      - max_lr: Maximum learning rate for OneCycleLR
      - pct_start: Percentage of total steps for warmup phase in OneCycleLR
      - div_factor: Initial learning rate = max_lr / div_factor
      - final_div_factor: Final learning rate = initial_lr / final_div_factor
      
    Returns:
      The best validation loss (as reported by the rank-0 process), or None if training failed.
    """
    # Create required directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(temp_dir_root, exist_ok=True)
    
    if tensorboard_log_dir:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
        print(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_log_dir}")
        
        print(f"Debug mode: {enable_debug}")
    
    # Log scheduler information
    print(f"Using {lr_scheduler_type} learning rate scheduler")
    if lr_scheduler_type == "onecycle":
        print(f"OneCycleLR parameters: max_lr={max_lr}, pct_start={pct_start}, "
              f"div_factor={div_factor}, final_div_factor={final_div_factor}")

    # Create a unique temporary directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(temp_dir_root, f"multimodal_shards_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s) for training.")

    # Pre-shard the datasets to disk with multimodal support
    print("Creating training data shards...")
    train_shard_paths = create_multimodal_shards(
        train_dataset, world_size, temp_dir, "train", 
        use_naip=model_config.use_naip, 
        use_uavsar=model_config.use_uavsar
    )
    
    print("Creating validation data shards...")
    val_shard_paths = create_multimodal_shards(
        val_dataset, world_size, temp_dir, "val", 
        use_naip=model_config.use_naip, 
        use_uavsar=model_config.use_uavsar
    )

    # Find a free port for DDP communication
    port = find_free_port()
    print(f"Using port {port} for DDP communication")

    # Add modality information to model name for clarity
    modality_str = ""
    if model_config.use_naip:
        modality_str += "_naip"
    if model_config.use_uavsar:
        modality_str += "_uavsar"
    if not (model_config.use_naip or model_config.use_uavsar):
        modality_str = "_baseline"
    
    full_model_name = f"{model_name}{modality_str}"
    print(f"Training model: {full_model_name}")

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
                target=_train_multimodal_worker,
                args=(rank, world_size,
                      train_shard_paths[rank],
                      val_shard_paths[rank],
                      full_model_name,
                      batch_size,
                      checkpoint_dir,
                      model_config,
                      port,
                      num_epochs,
                      log_file,
                      early_stopping_patience,
                      tensorboard_log_dir,
                      enable_debug,
                      accumulation_steps,
                      visualize_samples,
                      lr_scheduler_type,    # New parameter
                      max_lr,               # New parameter
                      pct_start,            # New parameter
                      div_factor,           # New parameter
                      final_div_factor)     # New parameter
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # After distributed training, read best validation loss from the file saved by rank 0.
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_best", f"{full_model_name}_final_best.pth")
        best_loss_file = os.path.join(checkpoint_dir, "final_best", f"{full_model_name}_best_loss.json")
        if os.path.exists(final_checkpoint_path) and os.path.exists(best_loss_file):
            with open(best_loss_file, 'r') as f:
                data = json.load(f)
                best_val_loss = data.get('best_val_loss')
            print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
            print(f"To view training metrics, run: tensorboard --logdir={tensorboard_log_dir}")
        else:
            print("Training did not complete successfully.")
    else:
        best_val_loss = _train_multimodal_worker(
            0, 1,
            train_shard_paths[0],
            val_shard_paths[0],
            full_model_name,
            batch_size,
            checkpoint_dir,
            model_config,
            port,
            num_epochs,
            log_file,
            early_stopping_patience,
            tensorboard_log_dir,
            enable_debug,
            accumulation_steps,
            visualize_samples,
            lr_scheduler_type=lr_scheduler_type,
            max_lr=max_lr,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    
    # Clean up temporary shards directory
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

    return best_val_loss


def run_ablation_studies(train_dataset, val_dataset, base_config, checkpoint_dir, 
                        model_name="pointupsampler", batch_size=8, epochs=40, enable_debug=False,
                        lr_scheduler_type="onecycle", max_lr=5e-4, pct_start=0.3, 
                        div_factor=25.0, final_div_factor=1e4):
    """
    Run ablation studies with different modality combinations.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        base_config: Base configuration for all models
        checkpoint_dir: Base directory for checkpoints
        batch_size: Batch size per GPU
        epochs: Number of epochs to train
        enable_debug: Whether to enable debugging features
        lr_scheduler_type: Type of learning rate scheduler to use ("plateau" or "onecycle")
        max_lr: Maximum learning rate for OneCycleLR
        pct_start: Percentage of total steps for warmup phase in OneCycleLR
        div_factor: Initial learning rate = max_lr / div_factor
        final_div_factor: Final learning rate = initial_lr / final_div_factor
    """
    import os
    import time
    from dataclasses import asdict
    
    # Import here to ensure we're using the same class reference
    from src.models.multimodal_model import MultimodalModelConfig
    
    # Run sequentially instead of using multiprocessing
    # This avoids pickling issues with the config class
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Log scheduler information
    print(f"Using {lr_scheduler_type} learning rate scheduler for all runs")
    if lr_scheduler_type == "onecycle":
        print(f"OneCycleLR parameters: max_lr={max_lr}, pct_start={pct_start}, "
              f"div_factor={div_factor}, final_div_factor={final_div_factor}")
    
    # 4. Both SAR & Optical
    print("\n\n========== Running Combined (SAR & Optical) ==========\n")
    combined_config_dict = {k: v for k, v in asdict(base_config).items()}
    combined_config_dict['use_naip'] = True
    combined_config_dict['use_uavsar'] = True
    combined_config = MultimodalModelConfig(**combined_config_dict)
    
    # Make a unique directory for this run
    combined_dir = os.path.join(checkpoint_dir, f"combined_{timestamp}")
    os.makedirs(combined_dir, exist_ok=True)
    
    train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        batch_size=batch_size,
        checkpoint_dir=combined_dir,
        model_config=combined_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    # 1. Baseline (3DEP Only)
    print("\n\n========== Running Baseline (3DEP Only) ==========\n")
    baseline_config_dict = {k: v for k, v in asdict(base_config).items()}
    baseline_config_dict['use_naip'] = False
    baseline_config_dict['use_uavsar'] = False
    baseline_config = MultimodalModelConfig(**baseline_config_dict)
    
    # Make a unique directory for this run
    baseline_dir = os.path.join(checkpoint_dir, f"baseline_{timestamp}")
    os.makedirs(baseline_dir, exist_ok=True)
    
    train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        batch_size=batch_size,
        checkpoint_dir=baseline_dir,
        model_config=baseline_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    # 2. SAR-Only
    print("\n\n========== Running SAR-Only ==========\n")
    sar_config_dict = {k: v for k, v in asdict(base_config).items()}
    sar_config_dict['use_naip'] = False
    sar_config_dict['use_uavsar'] = True
    sar_config = MultimodalModelConfig(**sar_config_dict)
    
    # Make a unique directory for this run
    sar_dir = os.path.join(checkpoint_dir, f"sar_only_{timestamp}")
    os.makedirs(sar_dir, exist_ok=True)
    
    train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        batch_size=batch_size,
        checkpoint_dir=sar_dir,
        model_config=sar_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    # 3. Optical-Only
    print("\n\n========== Running Optical-Only ==========\n")
    optical_config_dict = {k: v for k, v in asdict(base_config).items()}
    optical_config_dict['use_naip'] = True
    optical_config_dict['use_uavsar'] = False
    optical_config = MultimodalModelConfig(**optical_config_dict)
    
    # Make a unique directory for this run
    optical_dir = os.path.join(checkpoint_dir, f"optical_only_{timestamp}")
    os.makedirs(optical_dir, exist_ok=True)
    
    train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        batch_size=batch_size,
        checkpoint_dir=optical_dir,
        model_config=optical_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    print("\nAll ablation studies completed.")