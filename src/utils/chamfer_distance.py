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
    norm_type="L2_squared",
    huber_delta=2.0, 
    trim_percentage_x_to_y=0.0,
    trim_percentage_y_to_x=0.01,
    eps=1e-8
):
    """
    Compute the Chamfer distance between two batches of point clouds
    with selectable norm types (L1, L2, L2-squared, Huber)
    and optional trimming of largest distances.

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
        norm_type (str): Specifies the distance metric/loss:
                         - 'L1': Manhattan distance.
                         - 'L2': Euclidean distance (non-squared).
                         - 'L2_squared': Squared Euclidean distance (standard Chamfer).
                         - 'Huber': Huber loss applied to L2 distance.
        huber_delta (float): The threshold delta for Huber loss (used if norm_type='Huber').
                             Must be positive. Defaults to 1.0.
        trim_percentage_x_to_y (float): Percentage (0.0 to <1.0) of largest
                                        x->y distances to discard before reduction.
        trim_percentage_y_to_x (float): Percentage (0.0 to <1.0) of largest
                                        y->x distances to discard before reduction.
        eps (float): Small epsilon added for numerical stability.

    Returns:
        A tuple (chamfer_loss, chamfer_normals) where:
            - chamfer_loss is a scalar tensor giving the distance loss.
            - chamfer_normals is a scalar tensor giving the normals loss (or None).
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    # Updated norm_type validation
    if norm_type not in ["L1", "L2", "L2_squared", "Huber"]:
         raise ValueError("norm_type must be one of 'L1', 'L2', 'L2_squared', 'Huber'")
    if norm_type == "Huber" and not (huber_delta > 0):
         raise ValueError("huber_delta must be positive when norm_type is 'Huber'")
    if not (0.0 <= trim_percentage_x_to_y < 1.0):
        raise ValueError("trim_percentage_x_to_y must be in [0.0, 1.0)")
    if not (0.0 <= trim_percentage_y_to_x < 1.0):
        raise ValueError("trim_percentage_y_to_x must be in [0.0, 1.0)")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    return_normals = (x_normals is not None) and (y_normals is not None)

    N, P1, D = x.shape
    P2 = y.shape[1]

    x_lengths_long = x_lengths.long()
    y_lengths_long = y_lengths.long()
    x_mask = torch.arange(P1, device=x.device)[None, :] >= x_lengths_long[:, None]
    y_mask = torch.arange(P2, device=y.device)[None, :] >= y_lengths_long[:, None]

    if P1 == 0 or P2 == 0:
        cham_x_per_point = torch.zeros(N, P1, device=x.device, dtype=x.dtype)
        cham_y_per_point = torch.zeros(N, P2, device=y.device, dtype=y.dtype)
        cham_norm_x_sum = torch.zeros(N, device=x.device, dtype=x.dtype)
        cham_norm_y_sum = torch.zeros(N, device=y.device, dtype=y.dtype)
        k_vec_x = torch.zeros_like(x_lengths_long)
        k_vec_y = torch.zeros_like(y_lengths_long)
        cham_norm_x_per_point = None
        cham_norm_y_per_point = None
    else:
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

        # Calculate per-point losses based on the chosen norm type
        if norm_type == "L1":
            # Note: knn_gather might be slow/approximate in dummy version
            x_coords_near_y = knn_gather(y, x_nn.idx, y_lengths)[..., 0, :]
            y_coords_near_x = knn_gather(x, y_nn.idx, x_lengths)[..., 0, :]
            cham_x_per_point = torch.sum(torch.abs(x - x_coords_near_y), dim=2)
            cham_y_per_point = torch.sum(torch.abs(y - y_coords_near_x), dim=2)
        elif norm_type == "L2":
            cham_x_per_point = torch.sqrt(x_nn.dists[..., 0].clamp(min=0) + eps)
            cham_y_per_point = torch.sqrt(y_nn.dists[..., 0].clamp(min=0) + eps)
        elif norm_type == "Huber":
            # Huber loss is applied to the L2 distance (non-squared)
            l2_dist_x = torch.sqrt(x_nn.dists[..., 0].clamp(min=0) + eps)
            l2_dist_y = torch.sqrt(y_nn.dists[..., 0].clamp(min=0) + eps)
            # Use F.huber_loss comparing the distance to zero
            cham_x_per_point = F.huber_loss(l2_dist_x, torch.zeros_like(l2_dist_x),
                                            delta=huber_delta, reduction='none')
            cham_y_per_point = F.huber_loss(l2_dist_y, torch.zeros_like(l2_dist_y),
                                            delta=huber_delta, reduction='none')
        else: # norm_type == "L2_squared"
            cham_x_per_point = x_nn.dists[..., 0]
            cham_y_per_point = y_nn.dists[..., 0]

        # --- Apply Trimming ---
        k_vec_x = torch.ceil((1.0 - trim_percentage_x_to_y) * x_lengths.float()).long()
        k_vec_y = torch.ceil((1.0 - trim_percentage_y_to_x) * y_lengths.float()).long()
        k_vec_x = torch.where(x_lengths > 0, k_vec_x.clamp(min=1), torch.tensor(0, device=x.device, dtype=torch.long))
        k_vec_y = torch.where(y_lengths > 0, k_vec_y.clamp(min=1), torch.tensor(0, device=y.device, dtype=torch.long))

        if trim_percentage_x_to_y > 0.0:
            cham_x_masked = cham_x_per_point.masked_fill(x_mask, float('inf'))
            if P1 > 0:
                 sorted_cham_x = torch.topk(cham_x_masked, P1, dim=1, largest=False, sorted=True)[0]
            else:
                 sorted_cham_x = cham_x_masked
            keep_mask_x = torch.arange(P1, device=x.device)[None, :] < k_vec_x[:, None]
            cham_x_per_point = sorted_cham_x.masked_fill(~keep_mask_x, 0.0)
        else:
             cham_x_per_point = cham_x_per_point.masked_fill(x_mask, 0.0)

        if trim_percentage_y_to_x > 0.0:
            cham_y_masked = cham_y_per_point.masked_fill(y_mask, float('inf'))
            if P2 > 0:
                sorted_cham_y = torch.topk(cham_y_masked, P2, dim=1, largest=False, sorted=True)[0]
            else:
                sorted_cham_y = cham_y_masked
            keep_mask_y = torch.arange(P2, device=y.device)[None, :] < k_vec_y[:, None]
            cham_y_per_point = sorted_cham_y.masked_fill(~keep_mask_y, 0.0)
        else:
             cham_y_per_point = cham_y_per_point.masked_fill(y_mask, 0.0)

        # --- Compute Normal Consistency Loss ---
        if return_normals:
            # Ensure knn_gather returns valid shapes even if knn results were odd
            idx_x_nn = x_nn.idx if hasattr(x_nn, 'idx') else torch.full((N, P1, 1), -1, dtype=torch.long, device=x.device)
            idx_y_nn = y_nn.idx if hasattr(y_nn, 'idx') else torch.full((N, P2, 1), -1, dtype=torch.long, device=y.device)

            x_normals_near = knn_gather(y_normals, idx_x_nn, y_lengths)[..., 0, :]
            y_normals_near = knn_gather(x_normals, idx_y_nn, x_lengths)[..., 0, :]

            cham_norm_x_per_point = 1 - torch.abs(F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=eps))
            cham_norm_y_per_point = 1 - torch.abs(F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=eps))
            cham_norm_x_per_point = cham_norm_x_per_point.masked_fill(x_mask, 0.0)
            cham_norm_y_per_point = cham_norm_y_per_point.masked_fill(y_mask, 0.0)
            cham_norm_x_sum = cham_norm_x_per_point.sum(dim=1)
            cham_norm_y_sum = cham_norm_y_per_point.sum(dim=1)
        else:
            cham_norm_x_per_point = None
            cham_norm_y_per_point = None
            cham_norm_x_sum = torch.zeros(N, device=x.device, dtype=x.dtype)
            cham_norm_y_sum = torch.zeros(N, device=y.device, dtype=y.dtype)


    # --- Reduce over points ---
    cham_x_sum = cham_x_per_point.sum(dim=1)
    cham_y_sum = cham_y_per_point.sum(dim=1)

    if point_reduction == "mean":
        k_vec_x_clamped = k_vec_x.to(cham_x_sum.dtype).clamp(min=1.0)
        k_vec_y_clamped = k_vec_y.to(cham_y_sum.dtype).clamp(min=1.0)
        cham_x = cham_x_sum / k_vec_x_clamped
        cham_y = cham_y_sum / k_vec_y_clamped

        x_lengths_clamped = x_lengths.to(cham_norm_x_sum.dtype).clamp(min=1.0)
        y_lengths_clamped = y_lengths.to(cham_norm_y_sum.dtype).clamp(min=1.0)
        cham_norm_x = cham_norm_x_sum / x_lengths_clamped if return_normals else cham_norm_x_sum
        cham_norm_y = cham_norm_y_sum / y_lengths_clamped if return_normals else cham_norm_y_sum

    elif point_reduction == "sum":
        cham_x = cham_x_sum
        cham_y = cham_y_sum
        cham_norm_x = cham_norm_x_sum
        cham_norm_y = cham_norm_y_sum
    else:
        raise ValueError("Invalid point_reduction type")


    # --- Apply weights per batch element ---
    if weights is not None:
         weights_squeezed = weights.to(cham_x.dtype)
         cham_x = cham_x * weights_squeezed
         cham_y = cham_y * weights_squeezed
         cham_norm_x = cham_norm_x * weights_squeezed
         cham_norm_y = cham_norm_y * weights_squeezed


    # --- Reduce over batch ---
    if batch_reduction is not None:
        cham_x_batchsum = cham_x.sum()
        cham_y_batchsum = cham_y.sum()
        cham_norm_x_batchsum = cham_norm_x.sum()
        cham_norm_y_batchsum = cham_norm_y.sum()

        if batch_reduction == "mean":
            if point_reduction == "sum":
                if weights is None:
                     divisor_x = k_vec_x.sum().to(cham_x.dtype).clamp(min=1.0)
                     divisor_y = k_vec_y.sum().to(cham_y.dtype).clamp(min=1.0)
                     divisor_norm_x = x_lengths.sum().to(cham_norm_x.dtype).clamp(min=1.0)
                     divisor_norm_y = y_lengths.sum().to(cham_norm_y.dtype).clamp(min=1.0)
                else:
                     divisor_x = (weights * k_vec_x.to(weights.dtype)).sum().clamp(min=eps)
                     divisor_y = (weights * k_vec_y.to(weights.dtype)).sum().clamp(min=eps)
                     divisor_norm_x = (weights * x_lengths.to(weights.dtype)).sum().clamp(min=eps)
                     divisor_norm_y = (weights * y_lengths.to(weights.dtype)).sum().clamp(min=eps)
            elif point_reduction == "mean":
                 if weights is None:
                      divisor = torch.tensor(N, device=x.device, dtype=cham_x.dtype).clamp(min=1.0)
                 else:
                      divisor = weights.sum().clamp(min=eps)
                 divisor_x = divisor_y = divisor_norm_x = divisor_norm_y = divisor

            cham_x = cham_x_batchsum / divisor_x
            cham_y = cham_y_batchsum / divisor_y
            # Handle potentially zero divisor for normals if not computed
            cham_norm_x = cham_norm_x_batchsum / divisor_norm_x if return_normals and divisor_norm_x > 0 else cham_norm_x_batchsum
            cham_norm_y = cham_norm_y_batchsum / divisor_norm_y if return_normals and divisor_norm_y > 0 else cham_norm_y_batchsum

        elif batch_reduction == "sum":
            cham_x = cham_x_batchsum
            cham_y = cham_y_batchsum
            cham_norm_x = cham_norm_x_batchsum
            cham_norm_y = cham_norm_y_batchsum
        # else batch_reduction is None, use per-batch-element values

    # --- Final Combination ---
    cham_dist = cham_x + cham_y
    cham_normals = (cham_norm_x + cham_norm_y) if return_normals else None

    # Ensure output is scalar if batch reduction is done
    if batch_reduction is not None:
        # Check if already scalar before squeezing
        if cham_dist.dim() > 0:
             cham_dist = cham_dist.squeeze()
        if cham_normals is not None and cham_normals.dim() > 0:
            cham_normals = cham_normals.squeeze()

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
