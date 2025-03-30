import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Import original model components
from src.models.model import FeatureExtractor, Feature_Expansion

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
        
        # ====== 1) Feature Extractor (from original model) ======
        self.feature_extractor = FeatureExtractor(
            feature_dim=config.feature_dim,
            attr_dim=config.attr_dim
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
        
        # ====== 4) Node Shuffle (from original model) ======
        self.node_shuffle = Feature_Expansion(
            feat_dim=config.feature_dim,
            up_ratio=config.up_ratio,
            pos_mlp_hidden=config.pos_mlp_hdn,
            up_attn_hds=config.up_attn_hds,
            up_concat=config.up_concat,
            up_beta=config.up_beta,
            up_dropout=config.up_dropout,
            fnl_attn_hds=config.fnl_attn_hds
        )
        
        # Add LayerNorm after the node shuffle
        self.norm_after_nodeshuffle = nn.LayerNorm(self.node_shuffle.adjusted_dim)
        
        # ====== 5) Point Decoder (from original model) ======
        total_dim = self.node_shuffle.adjusted_dim
        self.point_decoder = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
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
        
        #remove extreme values likely from bird returns 
        dep_points[:, 2] = torch.clamp(dep_points[:, 2], 0, 150) # >150m is certainly a bird in any natural landscape

        # Default to zero attributes if None is provided
        if dep_attr is None:
            dep_attr = torch.zeros(dep_points.size(0), self.config.attr_dim, device=device)
        
        # ====== 1) Point Cloud Feature Extraction ======
        x_feat = self.feature_extractor(dep_attr, dep_points, edge_index)  # [N_dep, feat_dim]
        
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
                naip.get('relative_dates', None)  # Add this parameter
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
                uavsar.get('relative_dates', None)  # Add this parameter
            )  # [num_patches, embed_dim]
        
        # ====== 3) Apply Selected Fusion Module ======
        x_fused = self.fusion(
            point_features=x_feat,                                # [N_dep, feat_dim]
            edge_index=edge_index,                               # [2, E]
            point_positions=dep_points,                          # [N_dep, 3]
            naip_embeddings=naip_embeddings,                     # [num_patches, embed_dim] or None
            uavsar_embeddings=uavsar_embeddings,                 # [num_patches, embed_dim] or None
            main_bbox=bbox,                                      # [xmin, ymin, xmax, ymax]
            naip_bbox=naip.get('img_bbox', None) if naip is not None else None,
            uavsar_bbox=uavsar.get('img_bbox', None) if uavsar is not None else None,
            center=center,                                       # [1, 3]
            scale=scale                                          # scalar                                        
        )  # [N_dep, feat_dim]
        
        # ====== 4) Feature Expansion ======
        x_up = self.node_shuffle(x_fused, edge_index, dep_points)  # [r*N_dep, adjusted_dim]
        # x_up = self.norm_after_nodeshuffle(x_up)
        # x_up = F.relu(x_up)
        
        # ====== 5) Decode to 3D Coordinates ======
        pred_points = self.point_decoder(x_up)  # [r*N_dep, 3]
        
        return pred_points