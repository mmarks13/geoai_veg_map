
### Summary of Multi‑GPU Training with PyTorch DDP (MVP)

**1. Initial Setup for Multi‑GPU Training with DDP**
- **Objective:** Build an MVP using Distributed Data Parallel (DDP) in PyTorch with a DataLoader that uses multiple workers.
- **Key Components:**
  - **DDP Initialization:** Using `torch.distributed.init_process_group` and wrapping the model with `DistributedDataParallel`.
  - **Multiprocessing:** Using `mp.spawn` with the "spawn" start method.
  - **Gradient Accumulation & AMP:** Employing gradient accumulation and mixed precision training with `GradScaler` and `autocast`.

---

**2. Data Loading & Shared Memory Challenges**
- **Issues Encountered:**
  - SIGBUS errors with `num_workers > 0` due to limited shared memory (e.g., `/dev/shm` is only 64 MB).
  - Leaked semaphore warnings from the resource tracker.
- **Mitigation Strategies Explored:**
  - Setting `mp.set_sharing_strategy('file_system')`.
  - Lowering `num_workers` (and even using `num_workers=0`) to reduce shared memory usage.
  - Disabling resource tracker warnings with a hack.
  - Overriding temporary directories via `TMPDIR` and `TORCH_SHM_DIR` to point to a directory with more space.
  - Having each worker load its own dataset from disk rather than passing large data objects between processes.

---

**3. Model & Loss Issues**
- **Loss Function Problem:**
  - A dummy `chamfer_distance` function initially returned a constant tensor (0.5) that did not require gradients.
  - **Solution:** Redefine the loss to be differentiable (e.g., using mean squared error between tensors) so that gradients propagate.
- **Conditional Branches & Unused Parameters:**
  - Explored refactoring model code to eliminate conditional branches that might lead to unused parameters, thus avoiding the need for `find_unused_parameters=True`.

---

**4. Improving GPU Utilization**
- **Techniques Attempted:**
  - Enabling pinned memory (`pin_memory=True`) and using non-blocking data transfers.
  - Implementing a `DataPrefetcher` to overlap data loading with computation.
  - Enabling cuDNN benchmarking (`torch.backends.cudnn.benchmark = True`) for consistent kernel selection.
  - Increasing effective batch size via gradient accumulation.
- **Remaining Issue:** GPU utilization is still lower than desired (20–60%), prompting further investigation into data loading and processing bottlenecks.

---

**5. Checkpointing and KNN Precomputation Integration**
- **Precomputing KNN:**
  - Integrated a provided function `precompute_knn_inplace` that computes a k‑NN graph for each sample’s `'dep_points'` and stores it in `'dep_edge_index'`.
  - Saved the precomputed dataset to disk for reuse by each worker.
- **Model Checkpointing:**
  - Implemented basic checkpointing: saving the model’s state dictionary at the end of each epoch (on rank 0) to allow for recovery/resumption.

---

**6. Final MVP Code and Next Steps**
- **MVP Components:**
  - A simplified dataset, collate function, and model (SimpleGCN) are used.
  - The training loop supports gradient accumulation, DDP, multi-worker DataLoader, KNN precomputation, and checkpointing.
  - Warnings regarding deprecated APIs are suppressed.
- **Key Issues Addressed:**
  - Adjusted `autocast` usage to remove the `device_type` argument.
  - Resolved SIGBUS errors by modifying DataLoader settings and environment variables.
  - Managed shared memory challenges by having workers load data independently.

---

**7. What to Revisit Next**
- **DataLoader & Shared Memory:**  
  Experiment further with strategies for increasing `num_workers` without hitting shared memory limits (e.g., using alternative temporary directories or optimizing the collate function).
- **Checkpointing & Precomputation:**  
  Refine the integration of KNN precomputation and model checkpointing for robust, resumable training.
- **Model Collation:**  
  Ensure that the collate function correctly converts precomputed data (possibly stored as dictionaries) into tensors.
- **GPU Utilization Profiling:**  
  Add detailed timing and profiling to pinpoint bottlenecks in data loading, transfer, and computation.
- **Scaling:**  
  Investigate increasing the effective batch size and reducing gradient accumulation steps if GPU utilization remains low.

This roadmap summarizes all the key experiments, challenges, and solutions we discussed, and will serve as a solid foundation when you revisit multi-GPU training with PyTorch DDP.



# DataParallel Attampt Summary

## Model Wrapping  
- Started with `DataParallel`, then switched to PyG’s `GeoDataParallel` for proper graph handling.  
- Verified available devices:  
  `print("Device IDs:", model.device_ids)`

## Input & Forward Signature Adjustments  
- Changed the forward method to accept a single input (a `Data` object or list of `Data` objects) instead of multiple tensors.  
- Updated calls from:  
  `pred_points = model(dep_points, e_idx)`  
  to either wrapping inputs in a `Data` object:  
  `data = Data(pos=dep_points, edge_index=e_idx)`  
  `pred_points = model([data])`  
  or using batched graphs:  
  `batched_data = Batch.from_data_list(data_list)`  
  `pred_points = model(batched_data)`

## Batching vs. Per-Sample Processing  
- Initially processing one sample at a time prevented multi-GPU utilization.  
- Suggested two approaches:  
  1. **List of Data Objects:** Pass a list of individual `Data` objects so `GeoDataParallel` can scatter them:  
     `pred_points = model([data1, data2, data3])`  
  2. **Batched Graph:** Use `Batch.from_data_list()` to combine samples:  
     `batched_data = Batch.from_data_list(data_list)`  
     `pred_points = model(batched_data)`  
     Then split outputs using:  
     `losses = [chamfer_distance_pytorch3d(pred_points[batched_data.batch == i], uav_list[i]) for i in range(len(uav_list))]`

## Fixing Layer Input Format  
- Encountered a `NoneType` error in the point transformer layer.  
- Resolved by explicitly passing node features as a tuple:  
  `x_feat = self.pt_conv1((pos, pos), pos, edge_index)`

## GPU Data Movement & Utilization  
- Data objects were prematurely moved to GPU (`cuda:0`), so scatter never split the workload.  
- Revised code to create `Data` objects on CPU and let `GeoDataParallel` handle moving them:  
  `data = Data(pos=dep_points, edge_index=e_idx)`  
  `pred_points = model([data])`  
  `uav_points = uav_points.to(pred_points.device)`  (Move only when needed)

## Memory Cleanup  
- When jobs fail, release GPU memory before retrying:  
  `import torch, gc`  
  `torch.cuda.empty_cache()`  
  `gc.collect()`  
- Wrap training in try/except/finally to ensure cleanup:  
  `try:`  
  `    train()`  
  `except Exception as e:`  
  `    print(f"Error: {e}")`  
  `finally:`  
  `    torch.cuda.empty_cache()`  
  `    gc.collect()`

This summary encapsulates the key changes and issues we addressed, so you can pick up where you left off later.
