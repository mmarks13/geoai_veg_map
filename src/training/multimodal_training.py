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
from schedulefree import AdamWScheduleFree 

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
    One Cycle Learning Rate policy supporting multiple parameter groups.
    """
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        # Ensure max_lr is a list with one element per param group.
        if isinstance(max_lr, (list, tuple)):
            self.max_lr = list(max_lr)
        else:
            self.max_lr = [max_lr] * len(optimizer.param_groups)
        
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Compute initial and minimum LRs for each param group.
        self.initial_lr = [lr / div_factor for lr in self.max_lr]
        self.min_lr = [init_lr / final_div_factor for init_lr in self.initial_lr]
        
        self.warmup_steps = int(total_steps * pct_start)
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase from initial_lr to max_lr for each group.
            alpha = self.last_epoch / self.warmup_steps
            return [init_lr + alpha * (max_lr - init_lr)
                    for init_lr, max_lr in zip(self.initial_lr, self.max_lr)]
        else:
            # Cooldown phase: cosine annealing from max_lr to min_lr for each group.
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
            return [min_lr + alpha * (max_lr - min_lr)
                    for min_lr, max_lr in zip(self.min_lr, self.max_lr)]

                
class WarmupReduceLROnPlateau(_LRScheduler):
    """
    Adds a linear warmup phase to ReduceLROnPlateau scheduler.
    
    Args:
        optimizer: Optimizer to adjust learning rate for
        max_lr: Maximum learning rate after warmup
        total_epochs: Total number of epochs for training
        pct_start: Percentage of total epochs to use for warmup phase
        div_factor: Initial LR = max_lr / div_factor
        **kwargs: Additional arguments to pass to ReduceLROnPlateau
    """
    def __init__(self, optimizer, max_lr, total_epochs, pct_start=0.3, 
                 div_factor=25.0, mode='min', factor=0.5, patience=3, 
                 verbose=False, **kwargs):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.initial_lr = max_lr / div_factor
        self.warmup_epochs = int(total_epochs * pct_start)
        self.current_epoch = 0
        
        # Initialize optimizer with initial LR
        for group in optimizer.param_groups:
            group['lr'] = self.initial_lr
            
        # Set up plateau scheduler to start after warmup
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, 
            patience=patience, verbose=verbose, **kwargs
        )
        self.in_warmup = True
        
        # Initialize as a proper LR scheduler
        super(WarmupReduceLROnPlateau, self).__init__(optimizer)
    
    def get_lr(self):
        """Return current learning rate for each param group"""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def step(self, metrics=None, epoch=None):
        """
        Update scheduler based on metrics.
        
        Args:
            metrics: Validation metrics used by ReduceLROnPlateau
            epoch: Current epoch number (optional, uses internal counter if None)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
            
        # During warmup period, linearly increase learning rate
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            new_lr = self.initial_lr + progress * (self.max_lr - self.initial_lr)
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
            self.in_warmup = True
        # After warmup, use ReduceLROnPlateau behavior
        else:
            if self.in_warmup:
                # First epoch after warmup, set LR to max_lr
                for group in self.optimizer.param_groups:
                    group['lr'] = self.max_lr
                self.in_warmup = False
            
            # Let plateau scheduler handle LR after warmup
            if metrics is not None:
                self.plateau_scheduler.step(metrics)
            else:
                self.plateau_scheduler.step()


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
        self.data = torch.load(shard_path, weights_only=True)
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
        dep_points_attr = sample['dep_points_attr_norm']
        
        # Extract normalization parameters
        center = sample.get('center', None)
        scale = sample.get('scale', None)
        bbox = sample.get('bbox', None)
        
        # Add imagery data if requested
        naip_data = None
        uavsar_data = None

        #add tile_id for debugging
        tile_id = sample.get('tile_id', None)

        
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

        
        return (dep_points_norm, uav_points_norm, edge_index, dep_points_attr, naip_data, uavsar_data, center, scale, bbox, tile_id)


def multimodal_variable_size_collate(batch):
    """
    Collate function for multimodal variable-sized point clouds.
    """
    dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list = [], [], [], [], [], []
    center_list, scale_list, bbox_list, tile_id_list = [], [], [], []
    
    for item in batch:
        dep_pts, uav_pts, e_idx, dep_attr, naip_data, uavsar_data, center, scale, bbox, tile_id = item
        dep_list.append(dep_pts)
        uav_list.append(uav_pts)
        edge_list.append(e_idx)
        attr_list.append(dep_attr)
        naip_list.append(naip_data)
        uavsar_list.append(uavsar_data)
        center_list.append(center)
        scale_list.append(scale)
        bbox_list.append(bbox)
        tile_id_list.append(tile_id)
        
    return dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list


def process_multimodal_batch(model, batch, device):
    """
    Process a single multimodal batch through the model and compute the average loss per sample.
    Now includes spatial constraint information from center and scale.
    """
    dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list = batch
    total_loss = 0.0
    total_infocd_loss = 0.0
    total_repulsion_loss = 0.0
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

        from src.utils.infocd import info_cd_loss, repulsion_loss
        from pytorch3d.ops import knn_points

        # λ choice: k / |Y|   (e.g. k = 5 here gives a 5× stronger contrastive term)
        lam = 3.0 / uav_length.float()  


        # --- InfoCD ---  
        infocd = info_cd_loss(
            pred_points_batch, uav_points_batch,
            x_lengths=pred_length,
            y_lengths=uav_length,
            tau=0.3,        # temperature
            lam=lam        
        )

        # 2) compute dynamic repulsion radius h
        #    (a) do a 2‑NN on pred→pred
        knn = knn_points(
            pred_points_batch, pred_points_batch,
            lengths1=pred_length,
            lengths2=pred_length,
            K=2
        )
        #    (b) take the 2nd‑nearest distances, sqrt them
        nn_dists = torch.sqrt(knn.dists[..., 1].clamp(min=1e-12))

        #    (c) median per sample, then mean → scalar
        d_med = nn_dists.median(dim=1).values.mean()

        #    (d) scale by 1.8 and clamp to [0.15, 0.40] m
        h = (1.8 * d_med).clamp(0.15, 0.40)

        # --- Repulsion --- 
        repl = repulsion_loss(
            pred_points_batch,
            lengths=pred_length,
            k=8,
            h=h 
        )

        # Calculate weighted repulsion loss
        weighted_repl = 0.45 * repl
        
        # Combine with a small weight on repulsion
        sample_loss = infocd + weighted_repl

        if torch.isnan(sample_loss):
            print(f"WARNING: Loss for a sample is NaN! {tile_id_list[i]}")

        if torch.isinf(sample_loss):
            print(f"WARNING: Loss for a sample is Inf! {tile_id_list[i]}")
            
        total_loss += sample_loss
        total_infocd_loss += infocd
        total_repulsion_loss += weighted_repl
        sample_count += 1
    
    avg_loss = total_loss / sample_count if sample_count > 0 else total_loss
    avg_infocd = total_infocd_loss / sample_count if sample_count > 0 else total_infocd_loss
    avg_repl = total_repulsion_loss / sample_count if sample_count > 0 else total_repulsion_loss
    
    # Return a dictionary with all loss components
    return {
        'total': avg_loss,
        'infocd': avg_infocd,
        'repulsion': avg_repl
    }




def create_multimodal_model(device, config: MultimodalModelConfig):
    """
    Create a multimodal point upsampling model based on configuration.
    Returns the model and a set of parameter names that were loaded from checkpoint.
    """
    # Create the model with default initialization
    model = MultimodalPointUpsampler(config)
    
    # Track loaded layers for freezing functionality
    loaded_layers = set()
    
    # Selectively load weights if a checkpoint path is provided
    if config.checkpoint_path:
        # Load the checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        
        # Extract the state dictionary if needed
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
            
        # Get the current model state
        model_dict = model.state_dict()
        
        # Filter the pretrained dictionary
        if config.layers_to_load is not None:
            # Load only specific layers
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in config.layers_to_load and k in model_dict}
        else:
            # Load all matching layers (keys that exist in both dictionaries)
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            
        # Keep track of what was loaded for freezing
        loaded_layers = set(filtered_dict.keys())
        
        # Print detailed info about what's being loaded
        print(f"\nLoading {len(filtered_dict)} layers from checkpoint: {config.checkpoint_path}")
        print("Loaded layers:")
        # for idx, layer_name in enumerate(sorted(filtered_dict.keys())):
        #     print(f"  {idx+1}. {layer_name}")
        
        # Update model state with the filtered weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    
    # Apply freezing if specified in config
    if config.layers_to_freeze:
        # Verify that layers_to_freeze are in loaded_layers
        invalid_freeze_layers = [layer for layer in config.layers_to_freeze if layer not in loaded_layers]
        
        if invalid_freeze_layers:
            print("\nWARNING: The following layers were specified to freeze but were not loaded from checkpoint:")
            for layer in invalid_freeze_layers:
                print(f"  - {layer}")
            print("These layers will not be frozen.")
        
        # Filter to only freeze layers that were actually loaded
        valid_freeze_layers = [layer for layer in config.layers_to_freeze if layer in loaded_layers]
        
        # Handle freezing for the valid layers
        if valid_freeze_layers:
            print(f"\nFreezing {len(valid_freeze_layers)} loaded layers:")
            
            # Helper function to set requires_grad for specific parameters
            def set_requires_grad(model, layer_names, requires_grad=False):
                named_params = dict(model.named_parameters())
                for name in layer_names:
                    if name in named_params:
                        named_params[name].requires_grad = requires_grad
                        print(f"  - {name} (frozen)")
            
            # Freeze specific parameters in the model
            set_requires_grad(model, valid_freeze_layers, requires_grad=False)
    
    # Move model to the specified device
    model.to(device)
    
    # Print a summary of trainable vs non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"  - Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})")
    
    return model, loaded_layers  # Return both the model and loaded layers


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
                    process_batch_fn=None, accumulation_steps=1, enable_debug=False, 
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
    optimizer.train() # Set optimizer to training mode (ScheduleFree specific)
    model.train() # Set model to training mode
    train_loss_total = 0.0
    train_infocd_total = 0.0
    train_repulsion_total = 0.0
    batch_count = 0
    accumulated_batch_count = 0

    if process_batch_fn is None:
        # Ensure this import path is correct relative to where this function is defined
        from src.training.multimodal_training import process_multimodal_batch
        process_batch_fn = process_multimodal_batch

    global_step_base = epoch * (len(train_loader) // accumulation_steps) # Base step count for optimizer steps this epoch

    # Zero gradients at the beginning of the epoch
    optimizer.zero_grad()

    batch_log_interval = 10 if enable_debug else 999999

    max_grad_norm = 1.0 # Define the max_norm used for clipping here for logging consistency

    for batch_idx, batch in enumerate(train_loader):
        current_batch_step = epoch * len(train_loader) + batch_idx

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = process_batch_fn(model, batch, device)
            batch_loss = loss_dict['total']
            batch_infocd = loss_dict['infocd']
            batch_repulsion = loss_dict['repulsion']

            scaled_loss = batch_loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        train_loss_total += batch_loss.item()
        train_infocd_total += batch_infocd.item()
        train_repulsion_total += batch_repulsion.item()
        batch_count += 1
        accumulated_batch_count += 1

        # Log batch-level metrics if debugging is enabled and writer exists (rank 0 only)
        should_log_batch = (enable_debug and
                          batch_idx % batch_log_interval == 0 and
                          writer is not None)

        if should_log_batch:
            writer.add_scalar('Loss/train_batch', batch_loss.item(), current_batch_step)
            writer.add_scalar('Loss/train_infocd_batch', batch_infocd.item(), current_batch_step)
            writer.add_scalar('Loss/train_repulsion_batch', batch_repulsion.item(), current_batch_step)
            # Log LR per batch if debugging
            writer.add_scalar('Metrics/learning_rate_batch', optimizer.param_groups[0]['lr'], current_batch_step)


        is_last_batch = (batch_idx == len(train_loader) - 1)
        if accumulated_batch_count == accumulation_steps or is_last_batch:
            # --- OPTIMIZER STEP OCCURS HERE ---
            current_optimizer_step = global_step_base + (batch_idx // accumulation_steps)

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # --- GRADIENT CLIPPING & LOGGING ---
            # Clip gradients and capture the norm *before* clipping
            # Note: clip_grad_norm_ returns the total norm *before* clipping.
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

            # # Log the effective gradient norm (capped at max_norm)
            # if writer is not None:
            #      # Calculate the effective norm after clipping. If norm was <= max_norm, it's unchanged.
            #      # If norm was > max_norm, it was scaled down to max_norm.
            #      effective_grad_norm = min(grad_norm_before_clip.item(), max_grad_norm)
            #      writer.add_scalar('Gradients/norm_post_clip', effective_grad_norm, current_optimizer_step)
            #      # Optionally log the pre-clip norm for comparison/debugging
            #      writer.add_scalar('Gradients/norm_pre_clip', grad_norm_before_clip.item(), current_optimizer_step)
            # # --- END GRADIENT LOGGING ---

            # Update weights
            scaler.step(optimizer)
            scaler.update()

             # --- WEIGHT LOGGING ---
            if writer is not None:
                total_weight_norm = 0.0
                # Access model.module parameters if using DDP, otherwise model.parameters
                params_to_log = model.module.parameters() if isinstance(model, DDP) else model.parameters()
                with torch.no_grad(): # Ensure no gradients are calculated for logging
                    for p in params_to_log:
                        if p is not None and p.data is not None:
                            param_norm = p.data.norm(2)
                            total_weight_norm += param_norm.item() ** 2
                    total_weight_norm = total_weight_norm ** 0.5
                writer.add_scalar('Weights/total_norm_L2', total_weight_norm, current_optimizer_step)
            # --- END WEIGHT LOGGING ---

            # Reset gradients for the next accumulation cycle
            optimizer.zero_grad()

            # Reset accumulation counter
            accumulated_batch_count = 0
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    train_loss_tensor = torch.tensor(train_loss_total, device=device)
    train_infocd_tensor = torch.tensor(train_infocd_total, device=device)
    train_repulsion_tensor = torch.tensor(train_repulsion_total, device=device)
    
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_infocd_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_repulsion_tensor, op=dist.ReduceOp.SUM)
    
    avg_train_loss = train_loss_tensor.item() / batch_count / world_size
    avg_train_infocd = train_infocd_tensor.item() / batch_count / world_size
    avg_train_repulsion = train_repulsion_tensor.item() / batch_count / world_size
    
    # Log epoch-level losses if writer is available
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar('Loss/train_epoch_total', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_epoch_infocd', avg_train_infocd, epoch)
        writer.add_scalar('Loss/train_epoch_repulsion', avg_train_repulsion, epoch)
    
    return avg_train_loss


def validate_one_epoch_ddp(model, val_loader, device, writer=None, epoch=0, process_batch_fn=None,
                          enable_debug=False, visualize_samples=False, optimizer=None):
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
    optimizer.eval()
    model.eval()
    val_loss_total = 0.0
    val_infocd_total = 0.0
    val_repulsion_total = 0.0
    batch_count = 0
    
    # Store chamfer distances for metrics
    all_chamfer_distances = []
    
    # Use the provided batch processing function or use process_multimodal_batch by default
    if process_batch_fn is None:
        from src.training.multimodal_training import process_multimodal_batch
        process_batch_fn = process_multimodal_batch
    
    # Import chamfer distance from pytorch3d
    from pytorch3d.loss import chamfer_distance
    
    # Global step counter for tensorboard logging
    global_step = epoch * len(val_loader)
    
    # Batch logging interval - only log if debugging is enabled
    batch_log_interval = 10 if enable_debug else 999999
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            current_step = global_step + batch_idx
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Get loss components
                loss_dict = process_batch_fn(model, batch, device)
                batch_loss = loss_dict['total']
                batch_infocd = loss_dict['infocd']
                batch_repulsion = loss_dict['repulsion']
                
                # Also compute chamfer distance for each sample in the batch
                dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list = batch
                
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
                    
                    # Run the model to get predictions
                    pred_points = model(
                        dep_points, e_idx, dep_attr, 
                        naip_data, uavsar_data,
                        center, scale, bbox
                    )
                    
                    # Prepare for chamfer distance
                    pred_points_batch = pred_points.unsqueeze(0)
                    uav_points_batch = uav_points.unsqueeze(0)
                    
                    # Calculate chamfer distance using pytorch3d implementation
                    cd_loss, _ = chamfer_distance(pred_points_batch, uav_points_batch)
                    all_chamfer_distances.append(cd_loss.item())
            
            # Check if we should log for this batch
            should_log_batch = (enable_debug and 
                              batch_idx % batch_log_interval == 0 and 
                              writer is not None and 
                              dist.get_rank() == 0)
            
            # Log batch-level statistics
            if should_log_batch:
                writer.add_scalar('Loss/val_batch', batch_loss.item(), current_step)
                writer.add_scalar('Loss/val_infocd_batch', batch_infocd.item(), current_step)
                writer.add_scalar('Loss/val_repulsion_batch', batch_repulsion.item(), current_step)
            
            val_loss_total += batch_loss.item()
            val_infocd_total += batch_infocd.item()
            val_repulsion_total += batch_repulsion.item()
            batch_count += 1
    
    # Gather losses from all processes
    world_size = dist.get_world_size()
    val_loss_tensor = torch.tensor(val_loss_total, device=device)
    val_infocd_tensor = torch.tensor(val_infocd_total, device=device)
    val_repulsion_tensor = torch.tensor(val_repulsion_total, device=device)
    
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_infocd_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_repulsion_tensor, op=dist.ReduceOp.SUM)
    
    avg_val_loss = val_loss_tensor.item() / batch_count / world_size
    avg_val_infocd = val_infocd_tensor.item() / batch_count / world_size
    avg_val_repulsion = val_repulsion_tensor.item() / batch_count / world_size
    
    # Initialize chamfer distance metrics with default values
    cd_mean = cd_median = cd_min = cd_max = cd_p10 = cd_p90 = 0.0
    
    # Calculate chamfer distance metrics if we have any distances
    if len(all_chamfer_distances) > 0:
        # Convert to tensor for easy calculations
        local_cd_tensor = torch.tensor(all_chamfer_distances, device=device)
        
        # Gather chamfer distances from all processes, handling different sizes
        # First, get the local size as int64 tensor
        local_size = torch.tensor([local_cd_tensor.size(0)], dtype=torch.int64, device=device)
        
        # Create a list of tensors with matching dtype (int64)
        all_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        
        # Gather sizes from all processes
        dist.all_gather(all_sizes, local_size)
        
        # Convert to integers
        all_sizes = [int(size.item()) for size in all_sizes]
        max_size = max(all_sizes)
        
        # Pad local tensor to max size if needed
        if local_cd_tensor.size(0) < max_size:
            padding = torch.full((max_size - local_cd_tensor.size(0),), float('nan'), 
                                 dtype=local_cd_tensor.dtype, device=device)
            local_cd_tensor = torch.cat([local_cd_tensor, padding])
        
        # Create list of tensors to gather into, with matching dtype
        all_cd_tensors = [torch.zeros(max_size, dtype=local_cd_tensor.dtype, device=device) 
                          for _ in range(world_size)]
        
        # Gather all chamfer distances
        dist.all_gather(all_cd_tensors, local_cd_tensor)
        
        # Combine tensors and filter out padded values (NaNs)
        all_cd_values = []
        for i, tensor in enumerate(all_cd_tensors):
            valid_size = all_sizes[i]
            all_cd_values.extend(tensor[:valid_size].tolist())
        
        # Convert back to tensor for statistics calculation
        if len(all_cd_values) > 0:
            cd_tensor = torch.tensor(all_cd_values, device=device)
            cd_mean = cd_tensor.mean().item()
            cd_median = cd_tensor.median().item()
            cd_min = cd_tensor.min().item()
            cd_max = cd_tensor.max().item()
            cd_p10 = torch.quantile(cd_tensor, 0.1).item()  # 10th percentile
            cd_p90 = torch.quantile(cd_tensor, 0.9).item()  # 90th percentile
    
    # Log all metrics to TensorBoard if we're rank 0
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar('Loss/val_epoch_total', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_epoch_infocd', avg_val_infocd, epoch)
        writer.add_scalar('Loss/val_epoch_repulsion', avg_val_repulsion, epoch)
        
        # Log chamfer distance metrics
        writer.add_scalar('ChamferDist/mean', cd_mean, epoch)
        writer.add_scalar('ChamferDist/median', cd_median, epoch)
        writer.add_scalar('ChamferDist/min', cd_min, epoch)
        writer.add_scalar('ChamferDist/max', cd_max, epoch)
        writer.add_scalar('ChamferDist/p10', cd_p10, epoch)
        writer.add_scalar('ChamferDist/p90', cd_p90, epoch)
    
    # Return a dictionary with all metrics
    return {
        'loss': avg_val_loss,
        'infocd': avg_val_infocd,
        'repulsion': avg_val_repulsion,
        'cd_mean': cd_mean,
        'cd_median': cd_median,
        'cd_min': cd_min,
        'cd_max': cd_max,
        'cd_p10': cd_p10,
        'cd_p90': cd_p90
    }

def _train_multimodal_worker(rank, world_size, train_shard_path, val_shard_path,
                           model_name, batch_size, checkpoint_dir, model_config,
                           port, num_epochs, log_file, early_stopping_patience,
                           tensorboard_log_dir=None, enable_debug=False,
                           accumulation_steps=1, visualize_samples=False, 
                           lr_scheduler_type="plateau",
                           max_lr=5e-4, pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
                           dscrm_lr_ratio=10.0):
    """
    Worker function for training the multimodal model with discriminative learning rates.
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
                      f"early_stopping_patience: {early_stopping_patience}\n" +
                      f"max_lr: {max_lr}\ndscrm_lr_ratio: {dscrm_lr_ratio}", 0)
        
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
                    f"num_epochs: {num_epochs}, early_stopping_patience: {early_stopping_patience}, "
                    f"max_lr: {max_lr}, dscrm_lr_ratio: {dscrm_lr_ratio}")
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
    # Now also get the set of loaded layer names
    model, loaded_layers = create_multimodal_model(device, model_config)
    
    # Group parameters based on whether they were loaded from checkpoint
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include trainable parameters
            if name in loaded_layers:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    # Calculate learning rates for each group
    base_lr = max_lr  # Learning rate for new parameters
    pretrained_lr = base_lr / dscrm_lr_ratio  # Reduced learning rate for pretrained parameters
    
    if rank == 0:
        print(f"\nApplying discriminative learning rates:")
        print(f"  - New parameters: {len(new_params)} parameters with lr={base_lr}")
        print(f"  - Pretrained parameters: {len(pretrained_params)} parameters with lr={pretrained_lr}")
    
    # # Set up parameter groups for the optimizer with different starting LRs
    # param_groups = [
    #     {'params': pretrained_params, 'lr': pretrained_lr if lr_scheduler_type == "plateau" else pretrained_lr/div_factor},
    #     {'params': new_params, 'lr': base_lr if lr_scheduler_type == "plateau" else base_lr/div_factor}
    # ]
    # Set up parameter groups for the optimizer
    param_groups = [
        {'params': pretrained_params, 'lr': pretrained_lr},
        {'params': new_params, 'lr': base_lr}
    ]
    # optimizer = optim.AdamW(param_groups, weight_decay=5e-3)


    # Calculate total steps for OneCycleLR
    total_steps = (len(train_loader) // accumulation_steps) * num_epochs
    warmup_steps  = int(0.05 * total_steps)           # 5 % linear warm-up
    optimizer = AdamWScheduleFree(
        param_groups,
        lr=1e-3,                 # ignored for groups but required
        betas=(0.95, 0.999),
        weight_decay=1e-3,
        warmup_steps=warmup_steps, # replaces scheduler
        r=0.0,
        weight_lr_power=2.0,
    )

    # mark optimiser as in training mode
    optimizer.train()
    # # Initialize scheduler based on type
    # if lr_scheduler_type == "onecycle":
    #     scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=[pretrained_lr, base_lr],  # List of maximum LRs for each param group
    #         total_steps=total_steps,
    #         pct_start=pct_start,
    #         div_factor=div_factor,
    #         final_div_factor=final_div_factor
    #     )
    #     scheduler_batch_update = True
    # else:  # "plateau" with warmup
    #     scheduler = WarmupReduceLROnPlateau(
    #         optimizer, 
    #         max_lr=max_lr,
    #         total_epochs=num_epochs,
    #         pct_start=pct_start,
    #         div_factor=div_factor,
    #         mode='min', 
    #         factor=0.9, 
    #         patience=2, 
    #         verbose=(rank==0)
    #     )
    #     scheduler_batch_update = False  # Flag to indicate epoch-level updates
    # wrap the model in DDP after setting up the optimizer
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    scaler = GradScaler()

    # Log LR scheduler info
    if rank == 0 and logger is not None:
        logger.info(f"Using {lr_scheduler_type} learning rate scheduler with discriminative rates")
        logger.info(f"Pretrained params LR: {pretrained_lr}, New params LR: {base_lr}")
        if lr_scheduler_type == "onecycle":
            logger.info(f"OneCycleLR parameters: pct_start={pct_start}, div_factor={div_factor}, "
                      f"final_div_factor={final_div_factor}")

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):


        epoch_start_time = time.time()
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train_one_epoch_ddp(
            model, train_loader, optimizer, device, scaler, writer, epoch,
            process_batch_fn=process_multimodal_batch,
            accumulation_steps=accumulation_steps,
            enable_debug=enable_debug,
            # scheduler=scheduler if scheduler_batch_update else None  # Pass scheduler if batch update
        )
        
        # Validate
        val_metrics = validate_one_epoch_ddp(
            model, val_loader, device, writer, epoch,
            process_batch_fn=process_multimodal_batch,
            enable_debug=enable_debug,
            visualize_samples=visualize_samples,
            optimizer=optimizer
        )
        avg_val_loss = val_metrics['loss']
        cd_mean = val_metrics['cd_mean']
        cd_median = val_metrics['cd_median']
        cd_min = val_metrics['cd_min'] 
        cd_max = val_metrics['cd_max']
        cd_p10 = val_metrics['cd_p10']
        cd_p90 = val_metrics['cd_p90']
        
        # ───────────────────────────────────────────────────────────────
        #  update best-loss book-keeping  +  log progress   (rank-0 only)
        # ───────────────────────────────────────────────────────────────
        if rank == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = (
                    model.module.state_dict() if world_size > 1 else model.state_dict()
                )

                # ─ logging for a new best ─
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
                if writer is not None:
                    writer.add_scalar("Metrics/best_val_loss", best_val_loss, epoch)

            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement in validation loss for "
                    f"{epochs_without_improvement} epoch(s)."
                )

        # ───────────────────────────────────────────────────────────────
        #  decide whether to stop and tell every rank
        # ───────────────────────────────────────────────────────────────
        stop_tensor = torch.tensor(
            int(rank == 0 and epochs_without_improvement >= early_stopping_patience),
            device=device,
        )
        dist.broadcast(stop_tensor, src=0)

        if stop_tensor.item():
            if rank == 0:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(no improvement for {early_stopping_patience} epochs)."
                )
                if writer is not None:
                    writer.add_text(
                        "Training/early_stopping",
                        f"Stopped at epoch {epoch+1} after "
                        f"{early_stopping_patience} epochs without validation-loss improvement.",
                        epoch,
                    )
            break

        
        # # Update scheduler at epoch level only for ReduceLROnPlateau
        # if not scheduler_batch_update:
        #     scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time

        # Log epoch metrics to TensorBoard
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            writer.add_scalar('Loss/val_cd_mean_epoch', cd_mean, epoch)
            writer.add_scalar('Loss/val_cd_median_epoch', cd_median, epoch)
            writer.add_scalar('Loss/val_cd_min_epoch', cd_min, epoch)
            writer.add_scalar('Loss/val_cd_max_epoch', cd_max, epoch)
            writer.add_scalar('Loss/val_cd_p10_epoch', cd_p10, epoch)
            writer.add_scalar('Loss/val_cd_p90_epoch', cd_p90, epoch)

            
            # Additional metrics if debugging is enabled
            if enable_debug:
                writer.add_scalar('Metrics/epoch_time_minutes', epoch_time/60, epoch)
                writer.add_scalar('Metrics/learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Metrics/lr_main', optimizer.param_groups[1]['lr'], epoch)

                # Track GPU memory at epoch boundaries
                allocated = torch.cuda.memory_allocated(device=device) / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved(device=device) / (1024**2)    # MB
                writer.add_scalar('Memory/allocated_MB', allocated, epoch)
                writer.add_scalar('Memory/reserved_MB', reserved, epoch)

        if rank == 0:
            log_message = (f"Epoch {epoch+1}/{num_epochs} | "
                        f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                        f"CD Mean: {cd_mean:.6f} | CD Median: {cd_median:.6f} | "
                        f"CD Min: {cd_min:.6f} | CD Max: {cd_max:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.4e} | "
                        f"Time: {epoch_time/60:.2f}min")
            print(log_message)
            logger.info(log_message)


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


def train_multimodal_model(train_dataset=None,
                         val_dataset=None,
                         train_data_paths=None,
                         val_data_path=None,
                         model_name="model",
                         batch_size=8,
                         checkpoint_dir="checkpoints",
                         model_config=None,
                         num_epochs=60,
                         log_file="logs/training_log.txt",
                         early_stopping_patience=25,
                         temp_dir_root="data/output/tmp_shards",
                         shard_cache_dir="data/output/cached_shards",
                         use_cached_shards=False,
                         tensorboard_log_dir="data/output/tensorboard_logs",
                         enable_debug=False,
                         accumulation_steps=1,
                         visualize_samples=False,
                         lr_scheduler_type="plateau",
                         max_lr=5e-4,
                         pct_start=0.3,
                         div_factor=25.0,
                         final_div_factor=1e4,
                         dscrm_lr_ratio=10.0):

    """
    Unified training entry point for the multimodal model with shard caching.
    """
    # Create required directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(temp_dir_root, exist_ok=True)
    os.makedirs(shard_cache_dir, exist_ok=True)
    
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s) for training.")

    # Generate a cache key based on data source paths and sizes
    if train_data_paths is not None and val_data_path is not None:
        # When using file paths
        if isinstance(train_data_paths, list):
            # For multiple training data files, hash their paths
            import hashlib
            paths_str = "".join(sorted(train_data_paths)) + val_data_path
            cache_key = hashlib.md5(paths_str.encode()).hexdigest()[:10]
        else:
            # For single training data file
            import hashlib
            paths_str = train_data_paths + val_data_path
            cache_key = hashlib.md5(paths_str.encode()).hexdigest()[:10]
    elif train_dataset is not None and val_dataset is not None:
        # When using preloaded datasets, use their lengths
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        cache_key = f"train{train_len}_val{val_len}"
    else:
        raise ValueError("Either train_data_paths and val_data_path OR train_dataset and val_dataset must be provided")
    
    # Paths for cached shards
    train_cache_paths = [os.path.join(shard_cache_dir, f"{cache_key}_train_shard_{i}.pt") for i in range(world_size)]
    val_cache_paths = [os.path.join(shard_cache_dir, f"{cache_key}_val_shard_{i}.pt") for i in range(world_size)]
    
    # Check if cached shards exist and should be used
    cached_shards_exist = all(os.path.exists(p) for p in train_cache_paths + val_cache_paths)
    if use_cached_shards and cached_shards_exist:
        print(f"Using cached shards with key: {cache_key}")
        train_shard_paths = train_cache_paths
        val_shard_paths = val_cache_paths
    else:
        # Need to create shards
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(temp_dir_root, f"multimodal_shards_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)
        
         # Load data if needed and paths are provided
        if (train_dataset is None or val_dataset is None) and train_data_paths is not None and val_data_path is not None:
            print("Loading data from file paths...")
            
            # Load and combine training data if paths are provided
            if isinstance(train_data_paths, list):
                train_dataset = []
                for path in train_data_paths:
                    print(f"Loading training data from {path}...")
                    data = torch.load(path, weights_only=False)
                    train_dataset.extend(data)
            else:
                # Single path provided
                print(f"Loading training data from {train_data_paths}...")
                train_dataset = torch.load(train_data_paths, weights_only=False)
            
            # Load validation data
            print(f"Loading validation data from {val_data_path}...")
            val_dataset = torch.load(val_data_path, weights_only=False)
            
            print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        elif train_dataset is None or val_dataset is None:
            raise ValueError("Either train_data_paths and val_data_path OR train_dataset and val_dataset must be provided")

        
        # Create shards
        print("Creating training data shards...")
        train_shard_paths = create_multimodal_shards(
            train_dataset, world_size, temp_dir, "train", 
            use_naip=model_config.use_naip, 
            use_uavsar=model_config.use_uavsar
        )
        # Free memory immediately
        del train_dataset
        gc.collect()
        torch.cuda.empty_cache()

        print("Creating validation data shards...")
        val_shard_paths = create_multimodal_shards(
            val_dataset, world_size, temp_dir, "val", 
            use_naip=model_config.use_naip, 
            use_uavsar=model_config.use_uavsar
        )
        # Free memory immediately
        del val_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
        # Cache the shards for future use
        print(f"Caching shards with key: {cache_key}")
        for i, (train_shard, val_shard) in enumerate(zip(train_shard_paths, val_shard_paths)):
            cached_train_path = train_cache_paths[i]
            cached_val_path = val_cache_paths[i]
            
            shutil.copy(train_shard, cached_train_path)
            shutil.copy(val_shard, cached_val_path)
            
            # Update paths to use the cached versions
            train_shard_paths[i] = cached_train_path
            val_shard_paths[i] = cached_val_path
        
        print("Cleaning up memory...")
        gc.collect()
        torch.cuda.empty_cache()



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
                    lr_scheduler_type,    
                    max_lr,               
                    pct_start,            
                    div_factor,           
                    final_div_factor,
                    dscrm_lr_ratio)    
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
            final_div_factor=final_div_factor,
            dscrm_lr_ratio=dscrm_lr_ratio  
        )
    
    # Clean up temporary shards directory if it was created
    if 'temp_dir' in locals() and temp_dir:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    return best_val_loss

    

def run_ablation_studies(train_dataset=None, val_dataset=None,
                       train_data_paths=None, val_data_path=None,
                       base_config=None, checkpoint_dir="checkpoints", 
                       model_name="pointupsampler", batch_size=8, epochs=40, 
                       enable_debug=False, use_cached_shards=False,
                       shard_cache_dir="data/output/cached_shards",
                       lr_scheduler_type="onecycle", max_lr=5e-4, pct_start=0.3, 
                       div_factor=25.0, final_div_factor=1e4,
                       temp_dir_root="data/output/tmp_shards",
                       dscrm_lr_ratio=10.0):  # Add this parameter
    """
    Run ablation studies with different modality combinations.
    
    Args:
        train_dataset: Training dataset (optional if train_data_paths provided)
        val_dataset: Validation dataset (optional if val_data_path provided)
        train_data_paths: List of paths to training data files
        val_data_path: Path to validation data file
        base_config: Base configuration to use for all ablation studies
        checkpoint_dir: Base directory for checkpoints
        model_name: Base name for the model
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        enable_debug: Whether to enable debugging features
        use_cached_shards: Whether to use cached shards if available
        shard_cache_dir: Directory for cached shards
        lr_scheduler_type: Type of learning rate scheduler ("plateau" or "onecycle")
        max_lr: Maximum learning rate
        pct_start: Percentage of total steps for warmup phase
        div_factor: Division factor for initial learning rate
        final_div_factor: Final division factor for learning rate
        dscrm_lr_ratio: Ratio for discriminative learning rates (default: 10.0)
    """
    import os
    import time
    import shutil
    import json
    import torch
    from dataclasses import asdict
    import torch.distributed as dist
    
    # Import here to ensure we're using the same class reference
    from src.models.multimodal_model import MultimodalModelConfig
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a common directory for all final models
    all_final_models_dir = os.path.join(checkpoint_dir, f"all_ablation_results_{timestamp}")
    os.makedirs(all_final_models_dir, exist_ok=True)
    
    # Log scheduler information
    print(f"Using {lr_scheduler_type} learning rate scheduler for all runs")
    if lr_scheduler_type == "onecycle":
        print(f"OneCycleLR parameters: max_lr={max_lr}, pct_start={pct_start}, "
              f"div_factor={div_factor}, final_div_factor={final_div_factor}")
        print(f"Using discriminative learning rate ratio: {dscrm_lr_ratio}")
    
    results = {}
    
    # Define a function to copy the final model to the common directory
    def copy_final_model(src_dir, ablation_type):
        """Copy the final best model from a specific ablation run to the common directory."""
        full_model_name = f"{model_name}_{ablation_type}"
        src_path = os.path.join(src_dir, "final_best", f"{full_model_name}_final_best.pth")
        if os.path.exists(src_path):
            dst_path = os.path.join(all_final_models_dir, f"{full_model_name}_final_best.pth")
            shutil.copy(src_path, dst_path)
            print(f"Copied final model for {ablation_type} to {dst_path}")
            
            # Also copy the best loss file
            loss_src = os.path.join(src_dir, "final_best", f"{full_model_name}_best_loss.json")
            if os.path.exists(loss_src):
                loss_dst = os.path.join(all_final_models_dir, f"{full_model_name}_best_loss.json")
                shutil.copy(loss_src, loss_dst)
                print(f"Copied best loss info for {ablation_type} to {loss_dst}")
                
            return dst_path
        else:
            print(f"Warning: Final model for {ablation_type} not found at {src_path}")
            return None
    
    # Function to evaluate a pre-trained model without training
    def evaluate_pretrained_model(model_config, val_shard_path, result_dir, model_name_suffix):
        """
        Evaluate a pre-trained model on validation data without training.
        
        Args:
            model_config: Model configuration
            val_shard_path: Path to validation data shard
            result_dir: Directory to save results
            model_name_suffix: Suffix for model name
        
        Returns:
            Validation loss
        """
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        # Set up device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load the validation dataset
        val_dataset = ShardedMultimodalPointCloudDataset(
            val_shard_path,
            model_config.k,
            use_naip=model_config.use_naip,
            use_uavsar=model_config.use_uavsar
        )
        
        # Create data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=multimodal_variable_size_collate
        )
        
        # Create the model and load weights
        model = create_multimodal_model(device, model_config)
        model.eval()
        
        # Evaluate the model
        val_loss_total = 0.0
        batch_count = 0
        
        # Helper function to convert tensors in a batch to float32
        def convert_batch_to_float32(batch):
            dep_list, uav_list, edge_list, attr_list, naip_list, uavsar_list, center_list, scale_list, bbox_list, tile_id_list = batch
            
            # Convert point cloud tensors to float32
            dep_list_float32 = [dep.float() for dep in dep_list]
            uav_list_float32 = [uav.float() for uav in uav_list]
            edge_list_float32 = [edge.long() for edge in edge_list]  # Keep edge_index as long
            
            # Convert attributes if not None
            attr_list_float32 = [attr.float() if attr is not None else None for attr in attr_list]
            
            # Convert center and scale if not None
            center_list_float32 = [center.float() if center is not None else None for center in center_list]
            scale_list_float32 = [scale.float() if scale is not None else None for scale in scale_list]
            bbox_list_float32 = [bbox.float() if bbox is not None else None for bbox in bbox_list]
            
            # Convert image data if available
            naip_list_float32 = []
            for naip_data in naip_list:
                if naip_data is not None:
                    naip_float32 = {}
                    for k, v in naip_data.items():
                        if k == 'images' and v is not None:
                            naip_float32[k] = v.float()
                        else:
                            naip_float32[k] = v
                    naip_list_float32.append(naip_float32)
                else:
                    naip_list_float32.append(None)
                    
            uavsar_list_float32 = []
            for uavsar_data in uavsar_list:
                if uavsar_data is not None:
                    uavsar_float32 = {}
                    for k, v in uavsar_data.items():
                        if k == 'images' and v is not None:
                            uavsar_float32[k] = v.float()
                        else:
                            uavsar_float32[k] = v
                    uavsar_list_float32.append(uavsar_float32)
                else:
                    uavsar_list_float32.append(None)
            
            return (dep_list_float32, uav_list_float32, edge_list_float32, attr_list_float32, 
                    naip_list_float32, uavsar_list_float32, center_list_float32, 
                    scale_list_float32, bbox_list_float32, tile_id_list)
        
        with torch.no_grad():
            for batch in val_loader:
                # Convert batch to float32 before processing
                batch_float32 = convert_batch_to_float32(batch)
                batch_loss = process_multimodal_batch(model, batch_float32, device)
                val_loss_total += batch_loss.item()
                batch_count += 1
        
        avg_val_loss = val_loss_total / batch_count if batch_count > 0 else float('inf')
        
        # Save the model and results
        os.makedirs(os.path.join(result_dir, "final_best"), exist_ok=True)
        full_model_name = f"{model_name}_{model_name_suffix}"
        
        # Save the model state
        final_model_path = os.path.join(result_dir, "final_best", f"{full_model_name}_final_best.pth")
        torch.save(model.state_dict(), final_model_path)
        
        # Save the validation loss
        best_loss_file = os.path.join(result_dir, "final_best", f"{full_model_name}_best_loss.json")
        with open(best_loss_file, 'w') as f:
            json.dump({'best_val_loss': avg_val_loss}, f)
        
        print(f"Evaluated pre-trained model with validation loss: {avg_val_loss:.6f}")
        print(f"Saved model state to: {final_model_path}")
        
        return avg_val_loss
    
    # # 1. Baseline (3DEP Only)
    # print("\n\n========== Running Baseline (3DEP Only) ==========\n")
    # baseline_config_dict = {k: v for k, v in asdict(base_config).items()}
    # baseline_config_dict['use_naip'] = False
    # baseline_config_dict['use_uavsar'] = False
    # baseline_config = MultimodalModelConfig(**baseline_config_dict)
    
    # # Make a unique directory for this run
    # baseline_dir = os.path.join(checkpoint_dir, f"baseline_{timestamp}")
    # os.makedirs(baseline_dir, exist_ok=True)
    
    # # Use a specific model name for this ablation
    # baseline_model_name = "baseline"
    
    # # Check if we have a pre-trained model for baseline
    # has_pretrained_baseline = (hasattr(base_config, 'checkpoint_path') and 
    #                            base_config.checkpoint_path is not None and 
    #                            base_config.checkpoint_path != '')
    
    # if has_pretrained_baseline:
    #     print(f"Using pre-trained baseline model from: {base_config.checkpoint_path}")
        
    #     # Generate validation shard
    #     timestamp_temp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     temp_dir = os.path.join(temp_dir_root, f"multimodal_shards_{timestamp_temp}")
    #     os.makedirs(temp_dir, exist_ok=True)
        
    #     # Load val dataset if needed
    #     if val_dataset is None and val_data_path is not None:
    #         print(f"Loading validation data from {val_data_path}...")
    #         val_dataset = torch.load(val_data_path, weights_only=False)
        
    #     # Create validation shard
    #     print("Creating validation data shard for pre-trained model evaluation...")
    #     val_shard_paths = create_multimodal_shards(
    #         val_dataset, 1, temp_dir, "val", 
    #         use_naip=baseline_config.use_naip, 
    #         use_uavsar=baseline_config.use_uavsar
    #     )
        
    #     # Evaluate the pre-trained model
    #     baseline_val_loss = evaluate_pretrained_model(
    #         baseline_config, 
    #         val_shard_paths[0], 
    #         baseline_dir, 
    #         baseline_model_name
    #     )
        
    #     # Clean up temp directory
    #     shutil.rmtree(temp_dir, ignore_errors=True)
        
    # else:
    #     # Train the baseline model as usual
    #     baseline_val_loss = train_multimodal_model(
    #         train_dataset=train_dataset,
    #         val_dataset=val_dataset,
    #         train_data_paths=train_data_paths,
    #         val_data_path=val_data_path,
    #         model_name=f"{model_name}_{baseline_model_name}",
    #         batch_size=batch_size,
    #         checkpoint_dir=baseline_dir,
    #         model_config=baseline_config,
    #         num_epochs=epochs,
    #         enable_debug=enable_debug,
    #         visualize_samples=enable_debug,
    #         use_cached_shards=use_cached_shards,
    #         shard_cache_dir=shard_cache_dir,
    #         lr_scheduler_type=lr_scheduler_type,
    #         max_lr=max_lr,
    #         pct_start=pct_start,
    #         div_factor=div_factor,
    #         final_div_factor=final_div_factor,
    #         dscrm_lr_ratio=dscrm_lr_ratio
    #     )
    
    # # Print GPU memory usage
    # allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    # reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
    # print(f"GPU Memory Usage: Allocated: {allocated_memory:.2f} MB, Reserved: {reserved_memory:.2f} MB")

    # # Clear GPU memory
    # torch.cuda.empty_cache()
    # gc.collect()

    # # Copy the final model to the common directory
    # baseline_model_path = copy_final_model(baseline_dir, baseline_model_name)
    # results["baseline"] = {"val_loss": baseline_val_loss, "model_path": baseline_model_path}


    # 4. Both SAR & Optical
    print("\n\n========== Running Combined (SAR & Optical) ==========\n")
    combined_config_dict = {k: v for k, v in asdict(base_config).items()}
    combined_config_dict['use_naip'] = True
    combined_config_dict['use_uavsar'] = True
    combined_config = MultimodalModelConfig(**combined_config_dict)
    
    # Make a unique directory for this run
    combined_dir = os.path.join(checkpoint_dir, f"combined_{timestamp}")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Use a specific model name for this ablation
    combined_model_name = "combined"
    
    combined_val_loss = train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        model_name=f"{model_name}_{combined_model_name}",
        batch_size=batch_size,
        checkpoint_dir=combined_dir,
        model_config=combined_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        use_cached_shards=use_cached_shards,
        shard_cache_dir=shard_cache_dir,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        dscrm_lr_ratio=dscrm_lr_ratio
    )
    
    # Copy the final model to the common directory
    combined_model_path = copy_final_model(combined_dir, combined_model_name)
    results["combined"] = {"val_loss": combined_val_loss, "model_path": combined_model_path}

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # 2. SAR-Only
    print("\n\n========== Running SAR-Only ==========\n")
    sar_config_dict = {k: v for k, v in asdict(base_config).items()}
    sar_config_dict['use_naip'] = False
    sar_config_dict['use_uavsar'] = True
    sar_config = MultimodalModelConfig(**sar_config_dict)
    
    # Make a unique directory for this run
    sar_dir = os.path.join(checkpoint_dir, f"sar_only_{timestamp}")
    os.makedirs(sar_dir, exist_ok=True)
    
    # Use a specific model name for this ablation
    sar_model_name = "sar_only"
    
    sar_val_loss = train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        model_name=f"{model_name}_{sar_model_name}",
        batch_size=batch_size,
        checkpoint_dir=sar_dir,
        model_config=sar_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        use_cached_shards=use_cached_shards,
        shard_cache_dir=shard_cache_dir,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        dscrm_lr_ratio=dscrm_lr_ratio
    )
    
    # Copy the final model to the common directory
    sar_model_path = copy_final_model(sar_dir, sar_model_name)
    results["sar_only"] = {"val_loss": sar_val_loss, "model_path": sar_model_path}

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Optical-Only
    print("\n\n========== Running Optical-Only ==========\n")
    optical_config_dict = {k: v for k, v in asdict(base_config).items()}
    optical_config_dict['use_naip'] = True
    optical_config_dict['use_uavsar'] = False
    optical_config = MultimodalModelConfig(**optical_config_dict)
    
    # Make a unique directory for this run
    optical_dir = os.path.join(checkpoint_dir, f"optical_only_{timestamp}")
    os.makedirs(optical_dir, exist_ok=True)
    
    # Use a specific model name for this ablation
    optical_model_name = "optical_only"
    
    optical_val_loss = train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        model_name=f"{model_name}_{optical_model_name}",
        batch_size=batch_size,
        checkpoint_dir=optical_dir,
        model_config=optical_config,
        num_epochs=epochs,
        enable_debug=enable_debug,
        visualize_samples=enable_debug,
        use_cached_shards=use_cached_shards,
        shard_cache_dir=shard_cache_dir,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        dscrm_lr_ratio=dscrm_lr_ratio
    )
    
    # Copy the final model to the common directory
    optical_model_path = copy_final_model(optical_dir, optical_model_name)
    results["optical_only"] = {"val_loss": optical_val_loss, "model_path": optical_model_path}
    

    
    # Create a detailed summary file with results and configuration for publication
    summary_path = os.path.join(all_final_models_dir, "ablation_results_summary.txt")
    with open(summary_path, 'w') as f:
        # Header and basic info
        f.write(f"Point Cloud Upsampling Multimodal Ablation Study Results ({timestamp})\n")
        f.write("=" * 80 + "\n\n")
        
        # Study metadata
        f.write("STUDY METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Study ID: {timestamp}\n")
        f.write(f"Base checkpoint directory: {checkpoint_dir}\n")
        f.write(f"Results directory: {all_final_models_dir}\n\n")
        
        # Model architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Base model name: {model_name}\n")
        f.write(f"Point feature dimension: {base_config.feature_dim}\n")

        # Training parameters
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Learning rate scheduler: {lr_scheduler_type}\n")
        f.write(f"Maximum learning rate: {max_lr}\n")
        f.write(f"Warmup percentage: {pct_start}\n")
        f.write(f"Initial LR division factor: {div_factor}\n")
        f.write(f"Final LR division factor: {final_div_factor}\n")
        f.write(f"Mixed precision training: Enabled\n")
        f.write(f"Gradient clipping max norm: 3.0\n\n")
        
        # Hardware configuration
        gpu_count = torch.cuda.device_count()
        f.write("HARDWARE CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of GPUs: {gpu_count}\n")
        for i in range(gpu_count):
            if torch.cuda.is_available():
                f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
        f.write("\n")
        
        # Dataset information
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        if train_dataset is not None and val_dataset is not None:
            f.write(f"Training samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(val_dataset)}\n")
        elif isinstance(train_data_paths, list):
            f.write(f"Training data paths: {', '.join(train_data_paths)}\n")
            f.write(f"Validation data path: {val_data_path}\n")
        else:
            f.write(f"Training data path: {train_data_paths}\n")
            f.write(f"Validation data path: {val_data_path}\n")
        f.write(f"Using cached shards: {use_cached_shards}\n")
        f.write(f"Shard cache directory: {shard_cache_dir}\n\n")
        
        # Results section
        f.write("ABLATION STUDY RESULTS\n")
        f.write("-" * 80 + "\n")
        
        # Find the best model for reference
        best_modality = min(results.items(), key=lambda x: x[1]["val_loss"] if x[1]["val_loss"] is not None else float('inf'))
        best_loss = best_modality[1]["val_loss"] if best_modality[1]["val_loss"] is not None else float('inf')
        
        # Create a sorted list of modalities by performance
        sorted_modalities = sorted(results.items(), key=lambda x: x[1]["val_loss"] if x[1]["val_loss"] is not None else float('inf'))
        
        # Summary table header
        f.write("Summary Table (sorted by performance):\n")
        f.write(f"{'Rank':<5}{'Modality':<15}{'Val Loss':<15}{'Rel. Improvement':<20}{'Final Model':<40}\n")
        f.write("-" * 80 + "\n")
        
        # Add each modality to the summary table
        for i, (modality, data) in enumerate(sorted_modalities):
            val_loss = data["val_loss"]
            model_path = data["model_path"]
            
            # Calculate relative improvement compared to baseline (assuming baseline is the worst)
            baseline_loss = sorted_modalities[-1][1]["val_loss"]
            rel_improvement = ((baseline_loss - val_loss) / baseline_loss) * 100 if baseline_loss and val_loss else 0
            
            # Format model filename
            model_file = os.path.basename(model_path) if model_path else "Not found"
            
            # Write the table row
            f.write(f"{i+1:<5}{modality:<15}{val_loss:.6f}{rel_improvement:>15.2f}%{model_file:<40}\n")
        
        f.write("\n\nDetailed Results by Modality:\n")
        f.write("-" * 80 + "\n")
        
        for modality, data in sorted_modalities:
            val_loss = data["val_loss"]
            model_path = data["model_path"]
            best_marker = " (BEST)" if modality == best_modality[0] else ""
            
            # Calculate improvement against baseline
            baseline_loss = sorted_modalities[-1][1]["val_loss"]
            improvement = ((baseline_loss - val_loss) / baseline_loss) * 100 if baseline_loss and val_loss else 0
            
            f.write(f"Modality: {modality}{best_marker}\n")
            f.write(f"  Validation Loss: {val_loss:.6f}\n")
            if modality != sorted_modalities[-1][0]:  # Not the baseline
                f.write(f"  Improvement over baseline: {improvement:.2f}%\n")
            if best_modality[0] != modality and modality != sorted_modalities[-1][0]:
                best_vs_current = ((val_loss - best_loss) / best_loss) * 100
                f.write(f"  Gap to best model: {best_vs_current:.2f}%\n")
            
            # Add configuration details specific to this modality
            f.write("  Configuration:\n")
            
            if modality == "baseline":
                f.write("    - 3DEP point cloud data only\n")
                f.write("    - No additional modalities\n")
                if has_pretrained_baseline:
                    f.write(f"    - Pre-trained model: {base_config.checkpoint_path}\n")
            elif modality == "sar_only":
                f.write("    - 3DEP point cloud data\n")
                f.write("    - UAVSAR imagery\n")
                f.write("    - No optical imagery\n")
            elif modality == "optical_only":
                f.write("    - 3DEP point cloud data\n")
                f.write("    - NAIP optical imagery\n")
                f.write("    - No SAR imagery\n")
            elif modality == "combined":
                f.write("    - 3DEP point cloud data\n")
                f.write("    - NAIP optical imagery\n")
                f.write("    - UAVSAR imagery\n")
            
            f.write(f"  Model path: {model_path if model_path else 'Not found'}\n\n")
        
        # Conclusions section
        f.write("CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best performing model: {best_modality[0]} (Validation Loss: {best_modality[1]['val_loss']:.6f})\n")
        
        # Rank modalities by performance
        f.write("Performance ranking (best to worst):\n")
        for i, (modality, data) in enumerate(sorted_modalities):
            f.write(f"{i+1}. {modality}: {data['val_loss']:.6f}\n")
        
        # Add recommended model for inference
        f.write(f"\nRecommended model for inference: {best_modality[0]}\n")
        f.write(f"Model file: {os.path.basename(best_modality[1]['model_path']) if best_modality[1]['model_path'] else 'Not found'}\n")

    # Save results as JSON for easier programmatic access and analysis
    json_summary_path = os.path.join(all_final_models_dir, "ablation_results.json")
    with open(json_summary_path, 'w') as f:
        # Create a detailed JSON structure with all configurations
        json_results = {
            "metadata": {
                "timestamp": timestamp,
                "date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "checkpoint_dir": checkpoint_dir,
                "results_dir": all_final_models_dir,
                "baseline_pretrained": has_pretrained_baseline
            },
            "model_config": {
                "base_model_name": model_name,
                "feature_dim": base_config.feature_dim,
                "k_value": base_config.k,
                "checkpoint_path": base_config.checkpoint_path if hasattr(base_config, 'checkpoint_path') else None
            },
            "training_config": {
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "Adam",
                "lr_scheduler_type": lr_scheduler_type,
                "max_lr": max_lr,
                "pct_start": pct_start,
                "div_factor": div_factor,
                "final_div_factor": final_div_factor,
                "gradient_clipping": 3.0
            },
            "hardware": {
                "num_gpus": gpu_count,
                "gpu_types": [torch.cuda.get_device_name(i) for i in range(gpu_count)] if torch.cuda.is_available() else ["Unknown"]
            },
            "results": {
                m: {
                    "val_loss": d["val_loss"],
                    "model_file": os.path.basename(d["model_path"]) if d["model_path"] else None,
                    "full_path": d["model_path"],
                    "rank": i + 1,
                    "relative_improvement": ((sorted_modalities[-1][1]["val_loss"] - d["val_loss"]) / sorted_modalities[-1][1]["val_loss"]) * 100 
                        if sorted_modalities[-1][1]["val_loss"] and d["val_loss"] else 0
                } for i, (m, d) in enumerate(sorted_modalities)
            },
            "best_model": best_modality[0]
        }
        
        # Add modality-specific configurations
        json_results["modality_configs"] = {
            "baseline": {"use_naip": False, "use_uavsar": False, "pretrained": has_pretrained_baseline},
            "sar_only": {"use_naip": False, "use_uavsar": True},
            "optical_only": {"use_naip": True, "use_uavsar": False},
            "combined": {"use_naip": True, "use_uavsar": True}
        }
        
        json.dump(json_results, f, indent=2)

    # Create a publication-ready LaTeX table
    latex_summary_path = os.path.join(all_final_models_dir, "ablation_results_table.tex")
    with open(latex_summary_path, 'w') as f:
        f.write("% LaTeX table for publication\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Point Cloud Upsampling Ablation Study Results}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Modality & Validation Loss & Rel. Improvement (\\%) & Rank \\\\\n")
        f.write("\\midrule\n")
        
        for i, (modality, data) in enumerate(sorted_modalities):
            val_loss = data["val_loss"]
            # Calculate relative improvement compared to baseline
            baseline_loss = sorted_modalities[-1][1]["val_loss"]
            rel_improvement = ((baseline_loss - val_loss) / baseline_loss) * 100 if baseline_loss and val_loss else 0
            
            # Format modality name for LaTeX
            mod_name = modality.replace("_", "\\_")
            if modality == "baseline" and has_pretrained_baseline:
                mod_name = mod_name + "$^*$"  # Add asterisk for pre-trained model
            if modality == best_modality[0]:
                mod_name = "\\textbf{" + mod_name + "}"
            
            f.write(f"{mod_name} & {val_loss:.6f} & {rel_improvement:.2f} & {i+1} \\\\\n")
        
        f.write("\\bottomrule\n")
        if has_pretrained_baseline:
            f.write("\\multicolumn{4}{l}{$^*$Pre-trained model evaluation only (no training)} \\\\\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nAll ablation studies completed.")
    print(f"Final models saved in: {all_final_models_dir}")
    print(f"Results summary saved to: {summary_path}")
    print(f"JSON results saved to: {json_summary_path}")
    print(f"LaTeX table saved to: {latex_summary_path}")

    return all_final_models_dir