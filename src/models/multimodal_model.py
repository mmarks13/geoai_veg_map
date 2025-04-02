import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointTransformerConv
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph, PointTransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys
from src.utils.chamfer_distance import chamfer_distance


# Import the new components we just defined
from .encoders import NAIPEncoder, UAVSAREncoder
from .fusion import SpatialFusion
from .cross_attn_fusion import CrossAttentionFusion
import math

@dataclass
class MultimodalModelConfig:
    # Core model parameters
    k: int = 15
    feature_dim: int = 96
    up_ratio: int = 2
    pos_mlp_hdn: int = 32
    up_attn_hds: int = 2
    up_concat: bool = True
    up_beta: bool = False
    up_dropout: float = 0.0
    fnl_attn_hds: int = 2
    attr_dim: int = 3
    
    # Modality flags for ablation
    use_naip: bool = False
    use_uavsar: bool = False
    
    # Imagery encoder parameters
    img_embed_dim: int = 32
    img_num_patches: int = 16
    
    # Fusion module selection
    fusion_type: str = "spatial"  # Either "spatial" or "cross_attention"
    
    # Common fusion parameters
    temperature: float = 0.1
    max_dist_ratio: float = 1.5
    
    # CrossAttentionFusion specific parameters
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1
    position_encoding_dim: int = 32
    
    # parameters for encoder dropouts
    naip_dropout: float = 0.1
    uavsar_dropout: float = 0.1

    temporal_encoder: str = "gru"  # Type of temporal encoder: 'gru' or 'transformer'


    def __reduce__(self):
        """
        Custom reduce method to make the class picklable for multiprocessing.
        """
        return (
            self.__class__,
            (
                self.k,
                self.feature_dim,
                self.up_ratio,
                self.pos_mlp_hdn,
                self.up_attn_hds,
                self.up_concat,
                self.up_beta,
                self.up_dropout,
                self.fnl_attn_hds,
                self.attr_dim,
                self.use_naip,
                self.use_uavsar,
                self.img_embed_dim,
                self.img_num_patches,
                self.fusion_type,
                self.temperature,
                self.max_dist_ratio,
                self.fusion_num_heads,
                self.fusion_dropout,
                self.position_encoding_dim,
                self.naip_dropout,
                self.uavsar_dropout,
                self.temporal_encoder
            )
        )

def create_multimodal_model(device, config: MultimodalModelConfig):
    """
    Create a multimodal point upsampling model based on configuration.
    """
    model = MultimodalPointUpsampler(config)
    model.to(device)
    return model



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointTransformerConv
from torch.nn.functional import scaled_dot_product_attention


class PosAwareGlobalFlashAttention(nn.Module):
    """
    Position-Aware Global Flash Attention module that incorporates
    3D point positions into a flash attention mechanism.
   
    Features:
    - Processes 3D positions to create positional encodings
    - Integrates position information with feature vectors
    - Applies multi-head self-attention using scaled dot product attention
    - Global attention across all points
    """
    def __init__(self, dim, pos_encoding_dim=32, num_heads=4, dropout=0.1):
        """
        Initialize Position-Aware Global Flash Attention module.
       
        Args:
            dim: Feature dimension
            pos_encoding_dim: Dimension for positional encoding (default: 32)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention
        """
        super().__init__()
        self.dim = dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_heads = num_heads
        self.dropout = dropout
       
        # Position encoding MLP
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, pos_encoding_dim),
            nn.ReLU(),
            nn.Linear(pos_encoding_dim, pos_encoding_dim)
        )
       
        # Projection for combining position encoding with features
        self.pos_feature_combiner = nn.Linear(dim + pos_encoding_dim, dim)
       
        # Ensure dimension is divisible by number of heads
        assert dim % num_heads == 0, f"Dimension {dim} must be divisible by number of heads {num_heads}"
        self.head_dim = dim // num_heads
       
        # Linear projections for attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
       
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
   
    def forward(self, x, pos):
        """
        Apply position-aware flash attention to input tensor.
       
        Args:
            x: Input feature tensor of shape [N, dim]
            pos: Point position tensor of shape [N, 3]
           
        Returns:
            Output tensor of shape [N, dim]
        """
        # Apply residual connection pattern with normalization
        residual = x
        x = self.norm1(x)
       
        # Get original shape
        orig_shape = x.shape
        seq_len = orig_shape[0]
       
        # Add batch dimension if not present
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)  # [1, N, dim]
            pos = pos.unsqueeze(0) if len(pos.shape) == 2 else pos  # [1, N, 3]
            batch_size = 1
        else:
            batch_size = x.size(0)
       
        # Process positional information
        pos_flat = pos.view(-1, 3)
        pos_encoding = self.pos_encoder(pos_flat)
        pos_encoding = pos_encoding.view(*pos.shape[:-1], self.pos_encoding_dim)
       
        # Combine position encodings with feature vectors
        x_with_pos = torch.cat([x, pos_encoding], dim=-1)
        x_combined = self.pos_feature_combiner(x_with_pos)
       
        # Project to queries, keys, values
        q = self.q_proj(x_combined).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_combined).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_combined).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
       
        # Apply scaled dot product attention (Flash Attention)
        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
       
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_proj(attn_output)
       
        # Remove batch dimension if it was added
        if len(orig_shape) == 2:
            output = output.squeeze(0)
       
        # Apply second normalization and residual connection
        output = self.norm2(output + residual)
           
        return output


class MultiHeadPointTransformer(nn.Module):
    """
    A multi-head implementation using multiple PointTransformerConv layers.
    """
    def __init__(self, in_channels, out_channels, heads=4, concat=True):
        super().__init__()
        self.heads = heads
        self.concat = concat
       
        # If concat=True, each head contributes out_channels/heads features
        self.head_out_channels = out_channels // heads if concat else out_channels
       
        # Create multiple PointTransformerConv instances (one per head)
        self.convs = nn.ModuleList([
            PointTransformerConv(in_channels, self.head_out_channels)
            for _ in range(heads)
        ])
       
        # Projection layers when concat=True
        if concat and heads > 1:
            self.projection = nn.Sequential(
                nn.Linear(heads * self.head_out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        elif not concat and heads > 1:
            self.projection = nn.Linear(heads * out_channels, out_channels)
        else:
            self.projection = nn.Identity()
       
    def forward(self, x, pos, edge_index):
        # Apply each head
        head_outputs = [conv(x, pos, edge_index) for conv in self.convs]
       
        if self.concat:
            # Concatenate the outputs along the feature dimension
            concat_output = torch.cat(head_outputs, dim=-1)
            if self.heads > 1:
                return self.projection(concat_output)
            else:
                return concat_output
        else:
            # If not concatenating, either average or use projection
            if self.heads > 1:
                concat_output = torch.cat(head_outputs, dim=-1)
                return self.projection(concat_output)
            else:
                return head_outputs[0]


class FeatureExtractor(nn.Module):
    """
    Takes positions, attributes, and a precomputed edge_index and returns features.
    Uses PointTransformerConv layers to capture local geometric detail and Flash Attention.
    """
    def __init__(self, feature_dim=64, attr_dim=3, pos_encoding_dim=32):
        super().__init__()
        # First PointTransformerConv layer with half the output dimension
        self.pt_conv1 = MultiHeadPointTransformer(in_channels=attr_dim, out_channels=feature_dim//2)
       
        # Position-aware flash attention layer
        self.pos_flash_attention = PosAwareGlobalFlashAttention(
            dim=feature_dim//2,
            pos_encoding_dim=pos_encoding_dim,
            num_heads=4
        )
       
        # Second layer with original feature dimension
        self.pt_conv2 = MultiHeadPointTransformer(in_channels=feature_dim//2, out_channels=feature_dim)

    def forward(self, pos, attr, edge_index):
        """
        Args:
            pos: Tensor of shape [N, 3] containing point coordinates.
            attr: Tensor of shape [N, attr_dim] containing point attributes.
            edge_index: Tensor of shape [2, E] containing the edge indices.
        Returns:
            x_feat: Tensor of shape [N, feature_dim] representing per-point features.
        """
        # Use first PointTransformerConv layer
        x_feat = self.pt_conv1(attr, pos, edge_index)
       
        # Apply position-aware flash attention
        x_feat = self.pos_flash_attention(x_feat, pos)
       
        # Use second PointTransformerConv layer
        x_feat = self.pt_conv2(x_feat, pos, edge_index)
       
        return x_feat


class Feature_Expansion(nn.Module):
    def __init__(self, feat_dim=64, up_ratio=2, pos_encoding_dim=32,
                 up_attn_hds=2, up_concat=True, fnl_attn_hds=2):
        """  
        This module:
          (a) Uses a MultiHeadPointTransformer to expand features.
          (b) Performs periodic shuffle to upsample features.
          (c) Refines features using a position-aware flash attention module.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.up_ratio = up_ratio

        # MultiHeadPointTransformer for feature expansion
        self.point_transformer = MultiHeadPointTransformer(
            in_channels=feat_dim,
            out_channels=up_ratio * feat_dim,
            heads=up_attn_hds,
            concat=up_concat
        )
       
        # Position-aware global flash attention
        self.pos_flash_attention = PosAwareGlobalFlashAttention(
            dim=feat_dim,
            pos_encoding_dim=pos_encoding_dim,
            num_heads=fnl_attn_hds
        )
   
    def forward(self, x_feat, original_pos, edge_index):
        """
        Args:
            x_feat: [N, feat_dim] node features.
            edge_index: [2, E] graph connectivity.
            original_pos: [N, 3] original coordinates corresponding to x_feat.
        Returns:
            x_out: [r*N, feat_dim] refined features with position-aware attention.
        """
        N, C = x_feat.shape
        r = self.up_ratio
       
        # 1) Feature Expansion using MultiHeadPointTransformer
        x_expanded = self.point_transformer(x_feat, original_pos, edge_index)
       
        # 2) Group features: reshape to [N, r, feat_dim]
        # Each original point now has r associated feature vectors
        x_grouped = x_expanded.view(N, r, C)
       
        # 3) Convert to upsampled format: transform from [N, r, feat_dim] to [r*N, feat_dim]
        # This reorganizes data so we have r*N total points with feat_dim features each
        x_upsampled = x_grouped.permute(1, 0, 2).reshape(r * N, C)

        # 4) Duplicate the original positions r times to match upsampled points
        pos_rN = original_pos.unsqueeze(0).expand(r, N, 3).reshape(r * N, 3)

        # 5) Apply position-aware flash attention with duplicatetd original positions
        x_final = self.pos_flash_attention(x_upsampled, pos_rN)
       
        return x_final


class MultimodalPointUpsampler(nn.Module):
    """
    Enhanced point cloud upsampler that can incorporate NAIP and UAVSAR imagery
    with configurable spatial fusion methods
    """
    def __init__(self, config: MultimodalModelConfig):
        """
        Initialize the multimodal point cloud upsampler
       
        Args:
            config: Configuration for the model
        """
        super().__init__()
        self.config = config
       
        # Track which modalities are being used
        self.use_naip = config.use_naip
        self.use_uavsar = config.use_uavsar
       
        # ====== 1) Feature Extractor  ======
        self.feature_extractor = FeatureExtractor(
            feature_dim=config.feature_dim,
            attr_dim=config.attr_dim,
            pos_encoding_dim=config.position_encoding_dim
        )
       
        # ====== 2) Imagery Encoders ======
        if self.use_naip:
            self.naip_encoder = NAIPEncoder(
                in_channels=4,  # RGB + NIR
                image_size=40,  # 40x40 pixels
                patch_size=10,  # 10x10 pixel patches
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.naip_dropout,
                temporal_encoder_type=config.temporal_encoder  
            )
       
        if self.use_uavsar:
            self.uavsar_encoder = UAVSAREncoder(
                in_channels=6,  # 6 polarization bands
                image_size=4,   # 4x4 pixels
                patch_size=1,   # 1x1 pixel patches
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.uavsar_dropout,
                temporal_encoder_type=config.temporal_encoder  
            )
       
        # ====== 3) Configurable Fusion Module ======
        # Select and initialize the appropriate fusion module based on config
        if config.fusion_type.lower() == "cross_attention":
            self.fusion = CrossAttentionFusion(
                point_dim=config.feature_dim,
                patch_dim=config.img_embed_dim,
                use_naip=self.use_naip,
                use_uavsar=self.use_uavsar,
                num_patches=config.img_num_patches,
                max_dist_ratio=config.max_dist_ratio,
                num_heads=config.fusion_num_heads,
                attention_dropout=config.fusion_dropout,
                position_encoding_dim=config.position_encoding_dim
            )
        else:  # Default to spatial fusion
            self.fusion = SpatialFusion(
                point_dim=config.feature_dim,
                patch_dim=config.img_embed_dim,
                use_naip=self.use_naip,
                use_uavsar=self.use_uavsar,
                num_patches=config.img_num_patches,
                temperature=config.temperature,
                max_dist_ratio=config.max_dist_ratio
            )
       
        # ====== 4) Feature Expansion ======
        self.feature_expansion = Feature_Expansion(
            feat_dim=config.feature_dim,
            up_ratio=config.up_ratio,
            up_attn_hds=config.up_attn_hds,
            up_concat=True,
            pos_encoding_dim=config.position_encoding_dim,
            fnl_attn_hds=config.fnl_attn_hds
        )
       
        # ====== 5) Point Decoder ======
        self.point_decoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 2, config.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 4, config.feature_dim // 8),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 8, 3)
        )
   
    def forward(self, dep_points, edge_index, dep_attr=None, naip=None, uavsar=None, center=None, scale=None, bbox=None):      
       
        """
        Forward pass of the multimodal point cloud upsampler with configurable fusion
       
        Args:
            dep_points: 3DEP point cloud coordinates [N_dep, 3]
            edge_index: Edge indices for graph connectivity [2, E]
            dep_attr: 3DEP point attributes [N_dep, attr_dim]
            naip: Dictionary containing NAIP imagery data or None
                - 'images': NAIP images [n_images, 4, H, W]
                - 'img_bbox': Bounding box for spatial alignment
            uavsar: Dictionary containing UAVSAR imagery data or None
                - 'images': UAVSAR images [n_images, 6, H, W]
                - 'img_bbox': Bounding box for spatial alignment
            center: Center used for point cloud normalization [1, 3]
            scale: Scale used for point cloud normalization
            bbox: Point cloud bounding box [xmin, ymin, xmax, ymax]
       
        Returns:
            pred_points: Predicted point cloud coordinates [r*N_dep, 3]
        """
        # Get device from points tensor
        device = dep_points.device
       
        # Remove extreme values likely from bird returns
        dep_points[:, 2] = torch.clamp(dep_points[:, 2], 0, 150)  # >150m is certainly a bird in any natural landscape

        # ====== 1) Point Cloud Feature Extraction ======
        x_feat = self.feature_extractor(dep_points, dep_attr, edge_index)  # [N_dep, feat_dim]
       
        # ====== 2) Imagery Feature Extraction (if applicable) ======
        naip_embeddings = None
        uavsar_embeddings = None
       
        if self.use_naip and naip is not None and 'images' in naip:
            # Make sure NAIP images are on the correct device
            if naip['images'] is not None:
                naip['images'] = naip['images'].to(device)
                # Also move relative_dates to the same device if present
                if 'relative_dates' in naip and naip['relative_dates'] is not None:
                    naip['relative_dates'] = naip['relative_dates'].to(device)
                   
            naip_embeddings = self.naip_encoder(
                naip['images'],
                naip.get('img_bbox', None),
                naip.get('relative_dates', None)
            )  # [num_patches, embed_dim]
       
        if self.use_uavsar and uavsar is not None and 'images' in uavsar:
            # Make sure UAVSAR images are on the correct device
            if uavsar['images'] is not None:
                uavsar['images'] = uavsar['images'].to(device)
                # Also move relative_dates to the same device if present
                if 'relative_dates' in uavsar and uavsar['relative_dates'] is not None:
                    uavsar['relative_dates'] = uavsar['relative_dates'].to(device)
                   
            uavsar_embeddings = self.uavsar_encoder(
                uavsar['images'],
                uavsar.get('img_bbox', None),
                uavsar.get('relative_dates', None)
            )  # [num_patches, embed_dim]
       
        # ====== 3) Apply Selected Fusion Module ======
        x_fused = self.fusion(
            point_features=x_feat,                                # [N_dep, feat_dim]
            edge_index=edge_index,                                # [2, E]
            point_positions=dep_points,                           # [N_dep, 3]
            naip_embeddings=naip_embeddings,                      # [num_patches, embed_dim] or None
            uavsar_embeddings=uavsar_embeddings,                  # [num_patches, embed_dim] or None
            main_bbox=bbox,                                       # [xmin, ymin, xmax, ymax]
            naip_bbox=naip.get('img_bbox', None) if naip is not None else None,
            uavsar_bbox=uavsar.get('img_bbox', None) if uavsar is not None else None,
            center=center,                                        # [1, 3]
            scale=scale                                           # scalar                                        
        )  # [N_dep, feat_dim]
       
        # ====== 4) Feature Expansion ======
        x_up = self.feature_expansion(x_fused, dep_points, edge_index)  # [r*N_dep, feat_dim]
       
        # ====== 5) Decode to 3D Coordinates ======
        pred_points = self.point_decoder(x_up)  # [r*N_dep, 3]
       
        return pred_points