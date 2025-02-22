#!/usr/bin/env python
"""
Standalone script for computing Chamfer Distance (and optional normal distance)
between two batches of point clouds on GPU during PyTorch training.

The expected input for points is a tensor of shape (N, P, D), where N is the batch size,
P is the maximum number of points (padded if necessary), and D is the point dimension.
Optional arguments include per-cloud lengths and normals.
"""

from collections import namedtuple
import torch
import torch.nn.functional as F


def _validate_chamfer_reduction_inputs(batch_reduction, point_reduction):
    """Check that reduction parameters are valid."""
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(points, lengths, normals):
    """
    Ensure input point clouds (and normals) are in tensor format.
    If lengths is None, assume each cloud has P points.
    """
    if isinstance(points, torch.Tensor):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points.contiguous()
        N, P, D = X.shape
        if lengths is None:
            lengths = torch.full((N,), P, dtype=torch.int64, device=X.device)
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, D)")
        return X, lengths, normals
    else:
        raise ValueError("points must be a torch.Tensor of shape (N, P, D)")


def knn_points(x, y, lengths1=None, lengths2=None, K=1):
    """
    For each point in x, find the K nearest neighbors in y.
    Args:
        x: Tensor of shape (N, P1, D)
        y: Tensor of shape (N, P2, D)
        lengths1: Optional LongTensor of shape (N,) giving valid counts in x.
        lengths2: Optional LongTensor of shape (N,) giving valid counts in y.
        K: number of neighbors (default=1)
    Returns:
        A namedtuple with fields:
            - dists: (N, P1, K) tensor of squared distances.
            - idx:   (N, P1, K) tensor of indices (into y).
    """
    N, P1, D = x.shape
    _, P2, _ = y.shape
    # Compute pairwise squared Euclidean distances.
    dists = torch.cdist(x, y, p=2) ** 2  # shape: (N, P1, P2)

    # If provided, mask out padded points in y.
    if lengths2 is not None:
        # Create a mask of shape (N, P2): True where the index is invalid.
        mask = torch.arange(P2, device=y.device)[None, :].expand(N, P2) >= lengths2[:, None]
        dists.masked_fill_(mask.unsqueeze(1), float("inf"))

    # For each point in x, get the K smallest distances (and their indices)
    dists_k, idx = torch.topk(dists, k=K, dim=2, largest=False, sorted=True)
    KNN = namedtuple("KNN", ["dists", "idx"])
    return KNN(dists=dists_k, idx=idx)


def knn_gather(x, idx, lengths=None):
    """
    Gathers neighbors from tensor x based on indices.
    Args:
        x: Tensor of shape (N, M, D) to be indexed.
        idx: LongTensor of shape (N, L, K) containing indices along dimension 1.
        lengths: Optional LongTensor of shape (N,) indicating valid number of points in x.
    Returns:
        Tensor of shape (N, L, K, D) where gathered invalid entries (if any) are set to 0.
    """
    N, M, D = x.shape
    _, L, K = idx.shape
    # Expand idx for gathering features.
    idx_expanded = idx.unsqueeze(-1).expand(N, L, K, D)
    # Expand x along a new dimension to match idx.
    gathered = x.unsqueeze(1).expand(N, L, M, D).gather(2, idx_expanded)
    if lengths is not None:
        # For each batch, any index greater or equal to lengths is invalid.
        valid_mask = idx < lengths.view(N, 1, 1)
        gathered[~valid_mask] = 0.0
    return gathered


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction="mean",
    point_reduction="mean",
):
    """
    Compute the Chamfer distance between two batches of point clouds.
    
    Args:
        x: Tensor of shape (N, P1, D) representing the first point cloud batch.
        y: Tensor of shape (N, P2, D) representing the second point cloud batch.
        x_lengths: Optional LongTensor of shape (N,) with valid point counts for x.
        y_lengths: Optional LongTensor of shape (N,) with valid point counts for y.
        x_normals: Optional Tensor of shape (N, P1, D) with normals for x.
        y_normals: Optional Tensor of shape (N, P2, D) with normals for y.
        weights: Optional Tensor of shape (N,) with per-cloud weights.
        batch_reduction: Reduction over the batch ("mean" or "sum") or None.
        point_reduction: Reduction over the points ("mean" or "sum").
        
    Returns:
        A tuple (chamfer_loss, chamfer_normals) where:
            - chamfer_loss is a scalar tensor giving the distance loss.
            - chamfer_normals is a scalar tensor giving the normals loss (or None if normals not provided).
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    # Process inputs (assumes x and y are tensors; extend this helper if needed)
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    return_normals = (x_normals is not None) and (y_normals is not None)

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Create masks for padded points (if clouds are heterogeneous)
    x_mask = torch.arange(P1, device=x.device)[None, :] >= x_lengths[:, None]
    y_mask = torch.arange(P2, device=y.device)[None, :] >= y_lengths[:, None]

    # Find nearest neighbors: for each point in x, find the closest in y and vice versa.
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    # Squared distances from each point to its nearest neighbor.
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # Zero out contributions from padded (invalid) points.
    cham_x = cham_x.masked_fill(x_mask, 0.0)
    cham_y = cham_y.masked_fill(y_mask, 0.0)

    if weights is not None:
        cham_x = cham_x * weights.view(N, 1)
        cham_y = cham_y * weights.view(N, 1)

    # Compute normal consistency loss if normals are provided.
    if return_normals:
        # For each point in x, gather the nearest neighbor normal from y.
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_x = cham_norm_x.masked_fill(x_mask, 0.0)
        cham_norm_y = cham_norm_y.masked_fill(y_mask, 0.0)
        if weights is not None:
            cham_norm_x = cham_norm_x * weights.view(N, 1)
            cham_norm_y = cham_norm_y * weights.view(N, 1)
    else:
        cham_norm_x = torch.zeros_like(cham_x)
        cham_norm_y = torch.zeros_like(cham_y)

    # Reduce over points.
    cham_x = cham_x.sum(dim=1)  # (N,)
    cham_y = cham_y.sum(dim=1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(dim=1)
        cham_norm_y = cham_norm_y.sum(dim=1)

    if point_reduction == "mean":
        cham_x = cham_x / x_lengths.to(cham_x.dtype)
        cham_y = cham_y / y_lengths.to(cham_x.dtype)
        if return_normals:
            cham_norm_x = cham_norm_x / x_lengths.to(cham_norm_x.dtype)
            cham_norm_y = cham_norm_y / x_lengths.to(cham_norm_y.dtype)

    # Reduce over batch.
    if batch_reduction is not None:
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else torch.tensor(N, device=x.device, dtype=cham_x.dtype)
            cham_x = cham_x / div
            cham_y = cham_y / div
            if return_normals:
                cham_norm_x = cham_norm_x / div
                cham_norm_y = cham_norm_y / div

    cham_dist = cham_x + cham_y
    cham_normals = (cham_norm_x + cham_norm_y) if return_normals else None
    return cham_dist, cham_normals


# Example usage:
if __name__ == "__main__":
    # Create two random point cloud batches (on GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, P1, P2, D = 4, 1000, 1200, 3
    # Random point clouds (assume fully valid, so lengths = P1, P2)
    x = torch.rand(N, P1, D, device=device)
    y = torch.rand(N, P2, D, device=device)
    # Optionally, random normals (normalized)
    x_normals = F.normalize(torch.rand(N, P1, D, device=device), dim=2)
    y_normals = F.normalize(torch.rand(N, P2, D, device=device), dim=2)

    # Compute chamfer distance (and normals loss)
    loss, loss_normals = chamfer_distance(x, y, x_normals=x_normals, y_normals=y_normals)
    print("Chamfer Distance:", loss.item())
    if loss_normals is not None:
        print("Normals Loss:", loss_normals.item())
