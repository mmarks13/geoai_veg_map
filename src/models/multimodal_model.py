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
    feature_dim: int = 256
    up_ratio: int = 2
    pos_mlp_hdn: int = 16

    # Point Transformer parameters
    pt_attn_dropout: float = 0.05

    # Feature extractor attention heads
    extractor_lcl_heads: int = 4
    extractor_glbl_heads: int = 4
    
    # Feature expansion attention heads
    expansion_lcl_heads: int = 4
    expansion_glbl_heads: int = 4
    
    # Feature refinement attention heads
    refinement_lcl_heads: int = 4
    refinement_glbl_heads: int = 4

    # Legacy parameters (kept for backward compatibility)
    num_lcl_heads: int = 4  # Local attention heads (for backward compatibility)
    num_glbl_heads: int = 4  # Global attention heads (for backward compatibility)
    up_attn_hds: int = 4     # Legacy parameter
    up_concat: bool = True   # No longer used but kept for backward compatibility
    up_beta: bool = False    # Legacy parameter
    fnl_attn_hds: int = 4  # Final attention heads

    attr_dim: int = 3

    # Modality flags for ablation
    use_naip: bool = False
    use_uavsar: bool = False

    # Imagery encoder parameters
    img_embed_dim: int = 64
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

    # Parameters for encoder dropouts
    naip_dropout: float = 0.1
    uavsar_dropout: float = 0.1

    temporal_encoder: str = "gru"  # Type of temporal encoder: 'gru' or 'transformer'

    # Checkpoint loading parameters
    checkpoint_path: str = None  # Path to checkpoint file for weight initialization
    layers_to_load: list = None  # Specific layers to load from checkpoint
    layers_to_freeze: list = None # Specific layers to freeze (must be in layers_to_load or loaded from checkpoint)


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
                self.pt_attn_dropout,
                self.extractor_lcl_heads,
                self.extractor_glbl_heads,
                self.expansion_lcl_heads,
                self.expansion_glbl_heads,
                self.refinement_lcl_heads,
                self.refinement_glbl_heads,
                self.num_lcl_heads,
                self.num_glbl_heads,
                self.up_attn_hds,
                self.up_concat,
                self.up_beta,
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
                self.temporal_encoder,
                self.checkpoint_path,
                self.layers_to_load,
                self.layers_to_freeze,
            )
        )





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch import Tensor
from typing import Optional, Union, Tuple
from torch_geometric.nn.conv import PointTransformerConv

# PyTorch Geometric imports
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, SparseTensor
import torch_sparse  # For set_diag function
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import to_undirected


class PosAwareGlobalFlashAttention(nn.Module):
    """
    Position-Aware Global Flash Attention module that incorporates
    3D point positions into a flash attention mechanism.
   
    Features:
    - Processes 3D positions to create positional encodings
    - Integrates position information with feature vectors
    - Applies multi-head self-attention using scaled dot product attention
    - Global attention across all points
    - Includes feedforward network similar to standard transformer
    """
    def __init__(self, dim, pos_encoding_dim=32, num_heads=4, dropout=0, ffn_expansion_factor=2):
        """
        Initialize Position-Aware Global Flash Attention module.
       
        Args:
            dim: Feature dimension
            pos_encoding_dim: Dimension for positional encoding (default: 32)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention
            ffn_expansion_factor: Expansion factor for feedforward network (default: 4)
        """
        super().__init__()
        self.dim = dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_heads = num_heads
        self.dropout = dropout
       
        # Position encoding MLP
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, pos_encoding_dim),
            nn.GELU(),
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
        self.norm3 = nn.LayerNorm(dim)  # New normalization for feedforward network
        
        # Feedforward Network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion_factor),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion_factor, dim)
        )
   
    def forward(self, x, pos):
        """
        Apply position-aware flash attention followed by feedforward network to input tensor.
       
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
       
        # Apply normalization and residual connection after attention
        output = self.norm2(output + residual)
        
        # Apply feedforward network with residual connection
        residual = output
        output = self.norm3(output)
        output = self.ffn(output)
        output = output + residual
           
        return output


class MultiHeadPointTransformer(nn.Module):
    """
    A multi-head implementation using multiple PointTransformerConv layers.
    
    This class implements a multi-head attention mechanism for point cloud processing
    by using multiple parallel PointTransformerConv layers and combining their outputs
    through concatenation.
    """
    def __init__(self, in_channels, out_channels, heads=4, ffn_expansion_factor=2):
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate output channels per head
        # Each head contributes out_channels/heads features
        self.head_out_channels = out_channels // heads
                
        # Create multiple PointTransformerConv instances (one per head)
        self.convs = nn.ModuleList([
            PointTransformerConv(in_channels, self.head_out_channels) 
            for _ in range(heads)
        ])
        
        # Define projection layer for multiple heads (without normalization)
        if heads > 1:
            self.projection = nn.Sequential(
                nn.Linear(heads * self.head_out_channels, out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            # For single head, no projection needed
            self.projection = nn.Identity()
        
        # Output normalization layer to apply after residual connection
        self.post_norm = nn.LayerNorm(out_channels)
        
        # Feedforward Network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * ffn_expansion_factor),
            nn.GELU(),
            nn.Linear(out_channels * ffn_expansion_factor, out_channels)
        )
        
        # FFN normalization layer
        self.ffn_norm = nn.LayerNorm(out_channels)

    def forward(self, x, pos, edge_index):
        """
        Forward pass through the multi-head transformer.
        
        Args:
            x: Node features
            pos: Node positions
            edge_index: Graph connectivity
            
        Returns:
            Transformed node features
        """
        # Store input for residual connection if in_channels equals out_channels
        residual = x if self.in_channels == self.out_channels else None
        
        # Apply each transformer head in parallel
        head_outputs = [conv(x, pos, edge_index) for conv in self.convs]
        
        # Concatenate the outputs along the feature dimension
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # Apply projection to get output
        output = self.projection(concat_output)
        
        # Add residual connection if dimensions match
        if residual is not None:
            output = output + residual
            
        # Apply normalization after adding residual
        output = self.post_norm(output)
        
        # Store output for FFN residual connection
        ffn_residual = output
        
        # Apply FFN
        output = self.ffn(output)
        
        # Add FFN residual connection
        output = output + ffn_residual
        
        # Apply FFN normalization
        output = self.ffn_norm(output)
        
        return output





class LocalGlobalPointAttentionBlock(nn.Module):
    """
    Enhanced local-global attention block with dual MLP design:
    1. Local attention + FFN →  Upsampling (optional) → Global attention + FFN
    
    Supports feature-guided position generation for upsampling.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_lcl_heads=4,
                 num_glbl_heads=4,
                 pos_encoding_dim=32, 
                 dropout=0.0, 
                 up_ratio=None,
                 k_neighbors=15,
                 pos_gen_hidden_dim=64):
        """
        Initialize the LocalGlobalPointAttentionBlock
        
        Args:
            in_channels: Input feature dimensions
            out_channels: Output feature dimensions
            num_lcl_heads: Number of attention heads for local attention (0 to skip)
            num_glbl_heads: Number of attention heads for global attention (0 to skip)
            pos_encoding_dim: Dimension for positional encoding
            dropout: Dropout probability
            up_ratio: If provided, performs point upsampling by this ratio
            k_neighbors: Number of neighbors for KNN graph construction when edge_index is None
            pos_gen_hidden_dim: Hidden dimension for position generator network
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_lcl_heads = num_lcl_heads
        self.num_glbl_heads = num_glbl_heads
        self.pos_encoding_dim = pos_encoding_dim
        self.dropout = dropout
        self.up_ratio = up_ratio
        self.k_neighbors = k_neighbors
        
        # Flag to determine whether to use local/global attention
        self.use_local_attention = num_lcl_heads > 0
        self.use_global_attention = num_glbl_heads > 0
        
        # Determine internal dimensions based on whether upsampling or not
        if up_ratio is not None:
            # When upsampling, the PointTransformer expands features by up_ratio
            self.pt_out_channels = up_ratio * out_channels
            # After reshaping, features return to original dimension
            self.flash_attn_dim = out_channels
        else:
            # Without upsampling, both modules work with the same dimensions
            self.pt_out_channels = out_channels
            self.flash_attn_dim = out_channels
        
        # 1. Local structure module (MultiHeadPointTransformer) or simple projection
        if self.use_local_attention:
            self.point_transformer = MultiHeadPointTransformer(
                in_channels=in_channels,
                out_channels=self.pt_out_channels,
                heads=num_lcl_heads
            )
        else:
            # Simple linear projection when local attention is skipped
            self.local_projection = nn.Linear(in_channels, self.pt_out_channels)
    

        # 2. Global context module (PosAwareGlobalFlashAttention) if enabled
        if self.use_global_attention:
            self.pos_flash_attention = PosAwareGlobalFlashAttention(
                dim=self.flash_attn_dim,
                pos_encoding_dim=pos_encoding_dim,
                num_heads=num_glbl_heads,
                dropout=dropout
            )
        
        
        # Feature-guided position generator (only used when upsampling)
        if up_ratio is not None:
            self.position_generator = nn.Sequential(
                nn.Linear(out_channels, pos_gen_hidden_dim),
                nn.GELU(),
                nn.Linear(pos_gen_hidden_dim, pos_gen_hidden_dim),
                nn.GELU(),
                nn.Linear(pos_gen_hidden_dim, 3)
            )
            
            # Initialize the last layer with small weights for stable training
            nn.init.normal_(self.position_generator[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.position_generator[-1].bias)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in={self.in_channels}, '
                f'out={self.out_channels}, '
                f'lcl_heads={self.num_lcl_heads}, '
                f'glbl_heads={self.num_glbl_heads}, '
                f'up_ratio={self.up_ratio})')
        
    def forward(self, x_feat, pos, edge_index=None):
        """
        Forward pass through the LocalGlobalPointAttentionBlock
        
        Args:
            x_feat: [N, in_channels] Input features
            pos: [N, 3] Input 3D positions
            edge_index: [2, E] Graph connectivity (optional, will build KNN if None)
            
        Returns:
            Tuple containing:
              - Output features: [N, out_channels] or [up_ratio*N, out_channels]
              - Output positions: [N, 3] or [up_ratio*N, 3]
        """
        # Build KNN graph if edge_index is not provided and using local attention
        if edge_index is None and self.use_local_attention:
            edge_index = knn_graph(
                pos, 
                k=self.k_neighbors, 
                batch=None,
                loop=False, 
                flow='source_to_target'
            )
            edge_index = to_undirected(edge_index)  # Make edges bidirectional
            
        # Store input for potential residual connection
        identity = x_feat if self.in_channels == self.out_channels else None
        
        # 1. Apply local attention via MultiHeadPointTransformer or simple projection
        if self.use_local_attention:
            x_local = self.point_transformer(x_feat, pos, edge_index)
        else:
            x_local = self.local_projection(x_feat)
        

        # 3. Apply upsampling if up_ratio is specified
        if self.up_ratio is not None:
            # Upsampling logic
            r = self.up_ratio
            N = x_feat.shape[0]
            C = self.flash_attn_dim
            
            # Group features: reshape to [N, r, out_channels]
            x_grouped = x_local.view(N, r, C)
            
            # Convert to upsampled format: from [N, r, out_channels] to [r*N, out_channels]
            x_upsampled = x_grouped.permute(1, 0, 2).reshape(r * N, C)
            
            # Feature-guided position generation
            base_positions = pos.repeat_interleave(r, dim=0)  # [r*N, 3]
            position_offsets = self.position_generator(x_upsampled)  # [r*N, 3]
            pos_upsampled = base_positions + position_offsets  # [r*N, 3]
            
            # Apply global attention on upsampled features or skip if disabled
            if self.use_global_attention:
                x_global = self.pos_flash_attention(x_upsampled, pos_upsampled)
            else:
                x_global = x_upsampled
            
            
            return x_global, pos_upsampled
        else:
            # Without upsampling - apply global attention or skip
            if self.use_global_attention:
                x_global = self.pos_flash_attention(x_local, pos)
            else:
                x_global = x_local
            
            
            return x_global, pos






class MultimodalPointUpsampler(nn.Module):
    """
    Enhanced point cloud upsampler that incorporates LocalGlobalPointAttentionBlocks
    for better feature extraction and upsampling, with support for multimodal fusion.
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
        
        # Ensure we have the necessary dropout parameters with defaults
        extractor_dropout = getattr(config, 'extractor_dropout', config.pt_attn_dropout)
        expansion_dropout = getattr(config, 'expansion_dropout', config.pt_attn_dropout)
        refinement_dropout = getattr(config, 'refinement_dropout', config.pt_attn_dropout)
        
        # Get granular attention head counts - use specific values if available,
        # otherwise fall back to the global defaults
        extractor_lcl_heads = getattr(config, 'extractor_lcl_heads', config.num_lcl_heads)
        extractor_glbl_heads = getattr(config, 'extractor_glbl_heads', config.num_glbl_heads)
        expansion_lcl_heads = getattr(config, 'expansion_lcl_heads', config.num_lcl_heads)
        expansion_glbl_heads = getattr(config, 'expansion_glbl_heads', config.num_glbl_heads)
        refinement_lcl_heads = getattr(config, 'refinement_lcl_heads', config.num_lcl_heads)
        refinement_glbl_heads = getattr(config, 'refinement_glbl_heads', config.num_glbl_heads)
        
        # Get position generation hidden dimension with default
        pos_gen_hidden_dim = getattr(config, 'pos_gen_hidden_dim', 64)
        
        # ====== 1) Initial Feature Extractor (LocalGlobalPointAttentionBlock) ======
        self.feature_extractor = LocalGlobalPointAttentionBlock(
            in_channels=6,
            out_channels=config.feature_dim,
            num_lcl_heads=extractor_lcl_heads,
            num_glbl_heads=extractor_glbl_heads, 
            pos_encoding_dim=config.position_encoding_dim,
            dropout=extractor_dropout,
            k_neighbors=config.k
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
        
        # ====== 3) Fusion Module ======
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
        
        # ====== 4) Feature Expansion with Upsampling (LocalGlobalPointAttentionBlock) ======
        self.feature_expansion = LocalGlobalPointAttentionBlock(
            in_channels=config.feature_dim,
            out_channels=config.feature_dim,
            num_lcl_heads=expansion_lcl_heads,
            num_glbl_heads=expansion_glbl_heads,
            pos_encoding_dim=config.position_encoding_dim,
            dropout=expansion_dropout,
            up_ratio=config.up_ratio,
            pos_gen_hidden_dim=pos_gen_hidden_dim,
            k_neighbors=config.k
        )
        
        # ====== 5) Additional Feature Refinement (LocalGlobalPointAttentionBlock) ======
        self.feature_refinement = LocalGlobalPointAttentionBlock(
            in_channels=config.feature_dim,
            out_channels=config.feature_dim,
            num_lcl_heads=refinement_lcl_heads,
            num_glbl_heads=refinement_glbl_heads,
            pos_encoding_dim=config.position_encoding_dim,
            dropout=refinement_dropout,
            k_neighbors=config.k
        )
        
        # ====== 6) Point Decoder ======
        self.point_decoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.GELU(),
            nn.Linear(config.feature_dim // 2, config.feature_dim // 4),
            nn.GELU(),
            nn.Linear(config.feature_dim // 4, config.feature_dim // 8),
            nn.GELU(),
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

        dep_points_and_attr = torch.cat([dep_attr, dep_points], dim=1)  # [N_dep, 6]

        # ====== 1) Point Cloud Feature Extraction ======
        x_feat, _ = self.feature_extractor(dep_points_and_attr, dep_points, edge_index)
        
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
        
        # ====== 4) Feature Expansion with Upsampling ======
        x_up, pos_up = self.feature_expansion(x_fused, dep_points, edge_index)  # [r*N_dep, feat_dim], [r*N_dep, 3]
        
        # ====== 5) Feature Refinement with new edge indices ======
        x_refined, _ = self.feature_refinement(x_up, pos_up)  # [r*N_dep, feat_dim]
        
        # ====== 6) Decode to 3D Coordinates ======
        pred_offset = self.point_decoder(x_refined)  # [r*N_dep, 3]
        
        # Add predicted offsets to the upsampled positions to get final coordinates
        pred_points = pos_up + pred_offset
        
        return pred_points