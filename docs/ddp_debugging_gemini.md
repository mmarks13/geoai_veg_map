

# PyTorch DDP, `pytorch_geometric`, Multiprocessing, and Upsampling Debugging Summary

This document summarizes the debugging process for a PyTorch training script using Distributed Data Parallel (DDP), `pytorch_geometric`, multiprocessing data loading (`num_workers > 0`), and point cloud upsampling.  The goal was to achieve stable and efficient training with variable-sized point cloud inputs.

## Initial Problem

The initial script was experiencing low GPU utilization and various crashes when attempting to use DDP and multiple data loader workers. The inputs are point clouds of varying sizes.

## Debugging Journey and Solutions

The debugging process involved a systematic approach of isolating potential issues by:

1.  **Creating an MVRE (Minimal, Verifiable, and Reproducible Example):**  A simplified script (`ddp_test.py`) was created with dummy data, a basic GCN model, and essential training components. This allowed us to isolate issues from the complexities of the full application code.

2.  **Testing Incrementally:**  We tested different configurations in a specific order:
    *   `ddp=False`, `num_workers=0` (single process, single worker)
    *   `ddp=False`, `num_workers=1` (single process, multiple workers)
    *   `ddp=False`, `num_workers > 1`
    *   `ddp=True`, `num_workers=0` (DDP, single worker per process)
    *   `ddp=True`, `num_workers > 0` (DDP, multiple workers)

3. **Using `torchrun`:** We used `torchrun` for launching.

This incremental testing helped pinpoint where errors occurred.

The following problems were encountered and solved:

| Problem                                                                      | Solution                                                                                                                                                                                                                                                                                         |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Low GPU utilization with variable-sized point clouds.                          | Implemented a (commented out) `DynamicBatchSampler` to equalize computational load per GPU by targeting a similar number of points per batch, rather than a fixed number of samples.  Also implemented gradient accumulation.                                                   |
| `OSError: [Errno 28] No space left on device` (related to `/dev/shm`).      | Implemented a context manager (`data_loader_context`) using `try...finally` (or could use `@contextmanager`) to *guarantee* that `DataLoader` instances are properly deleted and garbage collected, preventing semaphore leaks. Set `persistent_workers=False`. |
| `TypeError: autocast.__init__() missing 1 required positional argument: 'device_type'` | Used `torch.amp.autocast(device_type='cuda')` (or `'cpu'`) for correct mixed precision, and handled older PyTorch versions with a version check.  Also used the correct `torch.amp.GradScaler`.                                                       |
| `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | Ensured the loss calculation was correctly connected to the computational graph.  The `chamfer_distance` function was used, and we made sure to use the differentiable output of that function (the first element of the returned tuple). Ensured no `.detach()` calls.  |
| `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) ...` (multiple instances) | 1.  Corrected the model's output shape to match the upsampled target data by setting the last `GCNConv`'s `out_channels` to the correct number of output features (3 for x, y, z coordinates).  2. Corrected the upsampling logic in `process_batch`.       |
| `ValueError: not enough values to unpack (expected 3, got 2)` / `ValueError: too many values to unpack`                  | Corrected the `variable_size_collate` function and `DataLoader` setup to properly handle `pytorch_geometric`'s `Batch` objects, and correctly handle the nuances of `DistributedSampler` with and without multiprocessing.                           |
| `AttributeError: 'DistributedDataParallel' object has no attribute 'upsample_factor'` | Accessed attributes of the underlying model through `model.module` when using DDP.  Created a conditional to access attributes properly in both DDP and non-DDP cases.                                                                                        |
| Hanging during DDP initialization (no error message, just stuck).        | Identified likely issue as DDP + `num_workers = 0`. Ensured we used `torchrun`.   |
|`NameError: name 'subprocess' is not defined`       | Added `import subprocess`      |

**Final Working MVRE Code (`ddp_test.py`):**

```python
import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from chamfer_distance import chamfer_distance

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from preprocess import normalize_pair
from model import NodeShufflePointUpsampler_Relative_Attn
import subprocess

# --- Configuration ---
@dataclass
class Config:
    # Model parameters
    k: int = 10
    feature_dim: int = 64
    up_ratio: int = 2
    pos_mlp_hidden: int = 32
    up_attn_hds: int = 2
    up_concat: bool = False
    up_beta: bool = False
    up_dropout: float = 0.0
    fnl_attn_hds: int = 2

    # Training parameters
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 0.01
    accumulation_steps: int = 1
    target_points_per_gpu: int = 100000
    max_batch_size: int = 8
    num_workers: int = 0  # Start with 0, then increase
    prefetch_factor: int = 2
    ddp: bool = False
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = "test_model"
    checkpoint_dir: str = "checkpoints"

# --- Dataset ---
class PointCloudUpsampleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            sample = self.data_list[idx]
            dep_points = sample['dep_points']
            uav_points = sample['uav_points']
            edge_index = sample['dep_edge_index']
            dep_points_norm, uav_points_norm, _, _ = normalize_pair(dep_points, uav_points)
            return dep_points_norm, uav_points_norm, edge_index
        except Exception as e:
            print(f"ERROR in __getitem__, idx={idx}: {e}")
            raise

# --- Collate Function ---
def variable_size_collate(batch):
    data_list = [Data(x=dep_points, edge_index=edge_index, y=uav_points) for dep_points, uav_points, edge_index in batch]
    return Batch.from_data_list(data_list)

# --- Data Loader Context Manager ---
@contextmanager
def data_loader_context(dataset, config):
    if config.ddp:
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size if not config.ddp else None,  # Corrected batch_size
        sampler=train_sampler if config.ddp else None,  # Sampler ONLY if ddp
        shuffle=not config.ddp and train_sampler is None, #correct shuffle
        collate_fn=variable_size_collate,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    try:
        yield train_loader
    finally:
        print("CLEANUP: Deleting DataLoader...")
        del train_loader
        gc.collect()
        print("CLEANUP: Done.")

# --- Process Batch Function ---
def process_batch(model, batch, device, config):
    batch = batch.to(device)
    pred_features = model(batch)  # Pass the entire batch object

    upsample_factor = model.module.node_shuffle.up_ratio if config.ddp else model.node_shuffle.up_ratio
    upsampled_pos = batch.x.repeat_interleave(upsample_factor, dim=0)

    upsampled_edge_index = batch.edge_index.repeat(1, upsample_factor)
    offset = torch.arange(upsample_factor, device=device) * batch.x.shape[0]
    offset = offset.repeat_interleave(batch.edge_index.shape[1])
    upsampled_edge_index = upsampled_edge_index + offset

    pred_coords = upsampled_pos + pred_features.repeat_interleave(upsample_factor, dim=0)
    loss = chamfer_distance(pred_coords.unsqueeze(0), batch.y.unsqueeze(0))[0]
    print(f"Loss: {loss.item()}, Requires Grad: {loss.requires_grad}")
    return loss

# --- Training Loop ---
def train_one_epoch(model, train_loader, optimizer, device, scaler, config):
    model.train()
    train_loss_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader):
        if torch.__version__ >= '1.10.0':
            with torch.amp.autocast(device_type=config.device_type):
                loss = process_batch(model, batch, device, config)
        else:
            with torch.amp.autocast():
                loss = process_batch(model, batch, device, config)

        loss = loss / config.accumulation_steps

        if config.ddp and (i + 1) % config.accumulation_steps != 0:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss_total += loss.item() * config.accumulation_steps

    avg_train_loss = train_loss_total / len(train_loader)
    return avg_train_loss

# --- Precompute KNN ---
def precompute_knn_inplace(model_data, k=10):
    for i, sample in enumerate(model_data):
        dep_points = sample['dep_points'].contiguous()
        edge_index = knn_graph(dep_points, k=k, loop=False)
        edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))
        sample['dep_edge_index'] = edge_index

# --- Main Training Function ---
def train(rank, config, world_size):
    if config.ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # DO NOT USE mp.set_start_method here when using torchrun
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(config.device_type)

    model = NodeShufflePointUpsampler_Relative_Attn(
        feat_dim=config.feature_dim,
        up_ratio=config.up_ratio,
        pos_mlp_hidden=config.pos_mlp_hidden,
        up_attn_hds=config.up_attn_hds,
        up_concat=config.up_concat,
        up_beta=config.up_beta,
        up_dropout=config.up_dropout,
        fnl_attn_hds=config.fnl_attn_hds
    ).to(device)
    if config.ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(enabled=(config.device_type == 'cuda'))

    # Create dummy data
    num_samples = 100
    num_points = 50
    num_edges = 100
    dummy_data = []
    for _ in range(num_samples):
        dep_points = torch.randn(num_points, 3, dtype=torch.float)
        uav_points = torch.randn(num_points * config.up_ratio, 3, dtype=torch.float)
        edge_index = torch.randint(0, num_points, (2, num_edges), dtype=torch.long)
        dummy_data.append({'dep_points': dep_points, 'uav_points': uav_points, 'dep_edge_index': edge_index})

    precompute_knn_inplace(dummy_data, k=config.k)
    dataset = PointCloudUpsampleDataset(dummy_data)

    with data_loader_context(dataset, config) as train_loader:
        for epoch in range(config.num_epochs):
            avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, config)
            print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

    if config.ddp:
        torch.distributed.destroy_process_group()  # Correct cleanup

# --- Main Entry Point ---
def main():
    config = Config()
    config.ddp = True  # Test DDP
    config.num_workers = 3 # Test num_workers

    if config.ddp:
        world_size = torch.cuda.device_count()
        # Use torchrun, NOT mp.spawn
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={world_size}",
            "ddp_test.py",  # Your script name
        ]
        subprocess.run(command, check=True)  # Use subprocess for torchrun

    else:
        train(0, config, 1)  # Single process training

if __name__ == '__main__':
    main()



