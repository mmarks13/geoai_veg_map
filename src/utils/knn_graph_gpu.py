# utils/knn_graph_gpu.py
from pytorch3d.ops import knn_points
import torch
from pytorch3d.ops import knn_points
import torch

def knn_edge_index(pos, k: int, batch=None, lengths=None):
    """
    Build a PyG‑style edge_index on the GPU with PyTorch3D's knn_points.

    Parameters
    ----------
    pos   : (N, 3) or (B, N, 3) fp16/fp32 tensor of XYZ coords on CUDA
    k     : number of neighbours you want ( *including* the self‑point )
    batch : (N,) long tensor that maps every point to a cloud ID.
            If None, treat the whole tensor as one cloud.
    lengths : optional (B,) int tensor of valid‑lengths; useful if pos is padded.

    Returns
    -------
    edge_index : (2, E) long tensor on GPU, undirected + self‑loops removed
    """
    KNN = []

    if batch is not None:
        # ---- multiple clouds packed in a single tensor ----
        for b in batch.unique(sorted=True):
            mask     = (batch == b)
            sub_idx  = torch.nonzero(mask, as_tuple=False).view(-1)   # (Nb,)
            sub_pos  = pos[mask]                                      # (Nb,3)

            # add fake batch‑dim so knn_points sees shape (1,Nb,3)
            knn_out  = knn_points(sub_pos.float()[None],              # query
                                  sub_pos.float()[None],              # database
                                  K=k, return_nn=False)

            tgt = sub_idx.repeat_interleave(k)                        # (Nb*k,)
            src = sub_idx[ knn_out.idx.view(-1) ]                     # map back
            KNN.append(torch.stack([src, tgt], dim=0))                # (2,E_b)

        edge_index = torch.cat(KNN, dim=1)

    else:
        # ---- single cloud ----
        pos_b  = pos.float().unsqueeze(0)                             # (1,N,3)
        knn_out = knn_points(pos_b, pos_b, K=k, return_nn=False)

        N   = pos.shape[0]
        tgt = torch.arange(N, device=pos.device).repeat_interleave(k) # (N*k,)
        src = knn_out.idx.view(-1)                                    # (N*k,)
        edge_index = torch.stack([src, tgt], dim=0)                   # (2,E)

    # ---- drop self‑edges (the first neighbour is the point itself) ----
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep]
