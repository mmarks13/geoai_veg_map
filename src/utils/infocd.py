# src/utils/infocd.py
#
# InfoCD + Repulsion loss that plugs into your existing knn_points / knn_gather
# utilities (imported from the file you posted).

import torch
import torch.nn.functional as F
from .chamfer_distance import knn_gather   # reuse what you already have


from collections import namedtuple
from pytorch3d.ops import knn_points as _p3d_knn          
KNN = namedtuple("KNN", ["dists", "idx"])

def knn_points(x, y, lengths1=None, lengths2=None, K=1):
    """
    Fast K‑NN using PyTorch3D.
      x : (N, P1, D)  query points
      y : (N, P2, D)  database points
    Returns:
      KNN(dists, idx) with the SAME shapes & semantics
      as your original function:
        dists : (N, P1, K)  squared L2 distances
        idx   : (N, P1, K)  indices into y
    """
    out = _p3d_knn(
        x.float(), y.float(),
        lengths1=lengths1,    # optional padding masks
        lengths2=lengths2,
        K=K,
        return_nn=False,      # we only need distances + indices
    )
    return KNN(dists=out.dists, idx=out.idx)

    
# --------------------------------------------------------------------------- #
# InfoCD    (F. Lin et al., NeurIPS‑23)                                      #
# --------------------------------------------------------------------------- #
def info_cd_loss(
    x, y,
    x_lengths=None, y_lengths=None,
    tau=0.8,             # temperature  (≈ 0.5–1 × median Chamfer works well)
    lam=None,            # if None, use λ = 1/|y|   (paper choice τ′ = τ)
    eps=1e-8
):
    """
    InfoCD as defined in Eq.(5) of the paper.
      L = 1/τ·CD  +  λ · log Σ_k exp(-d_k/τ)        (symmetric x↔y)
    with L1 distances (more stable than L2 for InfoCD).

    Args:
        x, y : (N,P,D) and (N,Q,D) tensors
        *_lengths : optional valid counts (N,)
        tau      : Softmax temperature ( > 0 )
        lam      : λ trade‑off; if None we set λ = 1 / |other cloud|
    Returns:
        scalar loss  (averaged over batch)
    """

    # ----- nearest‑neighbor dists ------------ #
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    # gather the actual neighbor coordinates so we can take L1 distance
    y_near_x = knn_gather(y, x_nn.idx, y_lengths)[..., 0, :]          # (N,P,D)
    x_near_y = knn_gather(x, y_nn.idx, x_lengths)[..., 0, :]          # (N,Q,D)

    d_x = torch.sum(torch.abs(x - y_near_x), dim=-1)                  # (N,P)
    d_y = torch.sum(torch.abs(y - x_near_y), dim=-1)                  # (N,Q)

    # valid masks (deal with padding)
    if x_lengths is not None:
        P = x.size(1)
        x_mask = torch.arange(P, device=x.device)[None, :] >= x_lengths[:, None]
        d_x = d_x.masked_fill(x_mask, 0.0)          # zeros for padded rows
    if y_lengths is not None:
        Q = y.size(1)
        y_mask = torch.arange(Q, device=y.device)[None, :] >= y_lengths[:, None]
        d_y = d_y.masked_fill(y_mask, 0.0)

    # per‑cloud sizes (avoid div‑by‑0)
    len_x = x_lengths.to(d_x.dtype) if x_lengths is not None else torch.full(
        (x.size(0),), x.size(1), dtype=d_x.dtype, device=x.device)
    len_y = y_lengths.to(d_y.dtype) if y_lengths is not None else torch.full(
        (y.size(0),), y.size(1), dtype=d_y.dtype, device=y.device)

    # λ choice: λ = 1/|other cloud| unless user overrides
    lam_x = 1.0 / (len_y + eps) if lam is None else lam
    lam_y = 1.0 / (len_x + eps) if lam is None else lam

    # main terms --------------------------------
    # (1) 1/τ · CD  (mean over valid points)
    cd_x = d_x.sum(dim=1) / (tau * len_x)          # (N,)
    cd_y = d_y.sum(dim=1) / (tau * len_y)

    # (2) contrastive regulariser  λ log Σ exp(‒d/τ)
    reg_x = lam_x * torch.logsumexp(-d_x / tau, dim=1)   # (N,)
    reg_y = lam_y * torch.logsumexp(-d_y / tau, dim=1)

    loss_batch = cd_x + cd_y + reg_x + reg_y             # (N,)
    return loss_batch.mean()


# --------------------------------------------------------------------------- #
# Repulsion / uniformity loss (PU‑Net)                                        #
# --------------------------------------------------------------------------- #
def repulsion_loss(points, lengths=None, k=5, h=0.03, eps=1e-8):
    """
    Push neighboring generated points away from each other to
    encourage a uniform distribution.

    Args:
        points   : (N,P,D) predicted point cloud
        lengths  : optional valid counts (N,)
        k        : use k nearest neighbours  (k >= 1)
        h        : influence radius (same units as coords)
    Returns:
        scalar repulsion loss  (mean over batch)
    """
    # k+1 because knn includes the query point itself at distance zero
    knn = knn_points(points, points, lengths1=lengths, lengths2=lengths, K=k+1)
    d = torch.sqrt(knn.dists[..., 1:] + eps)             # discard self‑dist
    # weight = exp(-d^2 / h^2)   as in PU‑Net
    w = torch.exp(-d ** 2 / h ** 2)
    rep = F.relu(h - d) * w                              # (N,P,k)
    # mask out padded rows if needed
    if lengths is not None:
        P = points.size(1)
        mask = torch.arange(P, device=points.device)[None, :] >= lengths[:, None]
        rep = rep.masked_fill(mask.unsqueeze(-1), 0.0)
        denom = lengths.clamp(min=1).to(rep.dtype)
        loss = rep.sum(dim=(1, 2)) / denom               # per cloud
    else:
        loss = rep.mean(dim=(1, 2))
    return loss.mean()
