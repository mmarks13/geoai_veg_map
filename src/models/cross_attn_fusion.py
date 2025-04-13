import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import PointTransformerConv

class CrossAttentionFusion(nn.Module):
    """
    A fusion module that leverages cross-attention with implicit positional encodings 
    to fuse point cloud features with patch embeddings, and uses PointTransformerConv
    for final feature extraction.
    """
    def __init__(
        self,
        point_dim,              # Dimension of point features
        patch_dim=32,           # Dimension of patch embeddings
        use_naip=False,         # Whether to use NAIP features
        use_uavsar=False,       # Whether to use UAVSAR features
        num_patches=16,         # Number of patch embeddings per modality
        max_dist_ratio=1.5,     # Maximum distance ratio for masking attention
        num_heads=4,            # Number of attention heads
        attention_dropout=0.1,  # Dropout probability for attention
        position_encoding_dim=24, # Dimension for positional encodings, must be divisible by both 4 and 6 
        use_distance_mask=False  # Whether to use distance-based attention masking
    ):
        super().__init__()
        self.point_dim = point_dim
        self.patch_dim = patch_dim
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
        self.num_patches = num_patches
        self.max_dist_ratio = max_dist_ratio
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.position_encoding_dim = position_encoding_dim
        self.use_distance_mask = use_distance_mask
        
        # If neither modality is used, this module becomes a pass-through
        if not (use_naip or use_uavsar):
            self.identity = True
            return
        else:
            self.identity = False

        # Projects for point features to create queries
        self.point_query_proj = nn.Linear(point_dim, point_dim)

        # Projections for patch features to create keys and values
        if use_naip:
            self.naip_key_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            self.naip_value_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            
        if use_uavsar:
            self.uavsar_key_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            self.uavsar_value_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)

        # Layer normalization for pre-processing
        self.norm1 = nn.LayerNorm(point_dim)

        # Calculate output dimension after concatenation
        concat_dim = point_dim  # Start with point features
        if use_naip:
            concat_dim += point_dim  # Add NAIP attention output dimension
        if use_uavsar:
            concat_dim += point_dim  # Add UAVSAR attention output dimension

        # Linear layers for feature extraction and projection
        self.linear1 = nn.Linear(concat_dim, concat_dim)
        self.linear2 = nn.Linear(concat_dim, point_dim)
        self.act = nn.ReLU()  # ReLU activation between linear layers
       
        # Add layer normalization for post-processing
        self.norm2 = nn.LayerNorm(point_dim)
        
    def positional_encoding(self, positions, dim):
        """
        Generate sinusoidal positional encodings for multi-dimensional positions
    
        Args:
            positions: Position tensor [N, D_pos]
            dim: Dimension of the positional encoding (must be divisible by 2*D_pos)
    
        Returns:
            encodings: Positional encodings [N, dim]
        """
        device = positions.device
        N, D_pos = positions.shape
        assert dim % (2 * D_pos) == 0, "dim must be divisible by 2 * number of position dimensions"
    
        encodings = []
        dim_per_pos = dim // D_pos
    
        freq_seq = torch.arange(dim_per_pos // 2, device=device).float()
        inv_freq = 1.0 / (10000 ** (freq_seq / (dim_per_pos // 2)))
    
        for d in range(D_pos):
            pos_vals = positions[:, d].unsqueeze(1)  # [N, 1]
            args = pos_vals * inv_freq.unsqueeze(0)  # [N, dim_per_pos//2]
    
            encodings.append(torch.sin(args))
            encodings.append(torch.cos(args))
    
        encodings = torch.cat(encodings, dim=1)  # [N, dim]
    
        return encodings

    
    def get_patch_positions(self, img_bbox, patches_per_side):
        """
        Compute positions of patches within an image bbox where (0,0) is the center
        and corners are defined by the bounding box dimensions
        
        Args:
            img_bbox: [minx, miny, maxx, maxy] of the image
            patches_per_side: Number of patches per side (e.g., 4 for 4x4 grid)
                
        Returns:
            positions: [num_patches, 2] tensor with x,y coordinates
        """
        device = img_bbox.device if isinstance(img_bbox, torch.Tensor) else torch.device('cpu')
        
        # Convert to tensor if needed
        if not isinstance(img_bbox, torch.Tensor):
            img_bbox = torch.tensor(img_bbox, device=device, dtype=torch.float32)  # [4]
        
        minx, miny, maxx, maxy = img_bbox
        
        # Calculate bbox dimensions
        width = maxx - minx
        height = maxy - miny
        
        # Calculate patch size
        patch_size_x = width / patches_per_side
        patch_size_y = height / patches_per_side
        
        # Create grid of patch centers with (0,0) at image center
        half_width = width / 2
        half_height = height / 2
        
        x_centers = torch.linspace(
            -half_width + patch_size_x/2, 
            half_width - patch_size_x/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        y_centers = torch.linspace(
            -half_height + patch_size_y/2, 
            half_height - patch_size_y/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        # Create all combinations
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')  # Both [patches_per_side, patches_per_side]
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [patches_per_side^2, 2] = [num_patches, 2]
        
        return positions
    
    def cross_attention(self, queries, keys, values, mask=None):
        """
        Multi-head cross-attention mechanism with attention dropout
        
        Args:
            queries: Query tensor [N, dim]
            keys: Key tensor [M, dim]
            values: Value tensor [M, dim]
            mask: Optional attention mask [N, M]
            
        Returns:
            output: Attention output [N, dim]
        """
        # Reshape for multi-head attention
        batch_size = 1  # We're processing a single point cloud
        n_queries = queries.size(0)
        n_keys = keys.size(0)
        
        # Split channels into multiple heads
        head_dim = queries.size(1) // self.num_heads
        q = queries.view(batch_size, n_queries, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, N, dim/num_heads]
        k = keys.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, M, dim/num_heads]
        v = values.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, M, dim/num_heads]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, num_heads, N, M]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), -1000.0)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [1, num_heads, N, M]
        
        # Apply attention dropout
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [1, num_heads, N, dim/num_heads]
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * n_queries, -1)  # [N, dim]
        
        return attn_output
    
    def forward(self, point_features, edge_index, point_positions, naip_embeddings=None, uavsar_embeddings=None,
                main_bbox=None, naip_bbox=None, uavsar_bbox=None, center=None, scale=None):
        """
        Fuse point features with patch embeddings using cross-attention
        with implicit positional encodings and PointTransformerConv for final feature extraction
        
        Args:
            point_features: Point features [N, D_p]
            edge_index: Edge indices for graph connectivity [2, E]
            point_positions: Point positions in 3D space [N, 3]
            naip_embeddings: NAIP patch embeddings [P, D_patch] or None
            uavsar_embeddings: UAVSAR patch embeddings [P, D_patch] or None
            main_bbox: Bounding box of the point cloud [xmin, ymin, xmax, ymax]
            naip_bbox: Bounding box of NAIP imagery [minx, miny, maxx, maxy]
            uavsar_bbox: Bounding box of UAVSAR imagery [minx, miny, maxx, maxy]
            
        Returns:
            fused_features: Point features enhanced with patch information [N, D_p]
        """
        # Identity case - no imagery modalities used
        if self.identity:
            return point_features  # [N, D_p]
        
        # If we don't have position information, we can't do spatial fusion
        if point_positions is None:
            print("No point positions provided, returning original point features.")
            return point_features  # [N, D_p]
        
        # Get device
        device = point_features.device
        point_positions = point_positions.to(device)  # [N, 3]
        
        # Normalize and prepare inputs
        N = point_features.size(0)  # Number of points
        point_features = self.norm1(point_features)  # [N, D_p]
        
        # Encode point positions using sinusoidal encoding
        point_pos_encoded = self.positional_encoding(point_positions, self.position_encoding_dim)  # [N, pos_dim]
        
        # Create point queries with implicit position information
        queries = self.point_query_proj(point_features)  # [N, D_p]
        
        # Prepare to store attention outputs for concatenation
        to_concat = [point_features]  # Start with original point features [N, D_p]
        
        # Process NAIP modality if available
        if self.use_naip and naip_embeddings is not None and naip_bbox is not None:
            # Get patch positions
            patches_per_side = int(math.sqrt(self.num_patches))
            naip_patch_positions = self.get_patch_positions(
                naip_bbox, 
                patches_per_side
            ).to(device)  # [P, 2]
            
            # Create mask only if distance masking is enabled
            mask = None
            if self.use_distance_mask:
                # Calculate distances between points and patches - only when masking is used
                squared_diffs = (
                    point_positions[:, :2].unsqueeze(1) -  # [N, 1, 2]
                    naip_patch_positions.unsqueeze(0)      # [1, P, 2]
                ).pow(2)  # [N, P, 2]
                
                distances = torch.sqrt(squared_diffs.sum(dim=-1))  # [N, P]
                mask = distances > self.max_dist_ratio  # [N, P], True where attention should be masked
            
            # Encode patch positions using sinusoidal encoding
            naip_pos_encoded = self.positional_encoding(naip_patch_positions, self.position_encoding_dim)  # [P, pos_dim]
            
            # Combine patch features with positional encodings
            naip_features_with_pos = torch.cat([naip_embeddings.to(device), naip_pos_encoded], dim=1)  # [P, D_patch + pos_dim]
            
            # Create keys and values for cross-attention
            naip_keys = self.naip_key_proj(naip_features_with_pos)  # [P, D_p]
            naip_values = self.naip_value_proj(naip_features_with_pos)  # [P, D_p]
            
            # Compute attention output for NAIP with mask
            naip_attn_output = self.cross_attention(queries, naip_keys, naip_values, mask)  # [N, D_p]
            to_concat.append(naip_attn_output)
        
        # Process UAVSAR modality if available
        if self.use_uavsar and uavsar_embeddings is not None and uavsar_bbox is not None:
            # Get patch positions
            patches_per_side = int(math.sqrt(self.num_patches))
            uavsar_patch_positions = self.get_patch_positions(
                uavsar_bbox, 
                patches_per_side
            ).to(device)  # [P, 2]
            
            # Create mask only if distance masking is enabled
            mask = None
            if self.use_distance_mask:
                # Calculate distances between points and patches - only when masking is used
                squared_diffs = (
                    point_positions[:, :2].unsqueeze(1) -  # [N, 1, 2]
                    uavsar_patch_positions.unsqueeze(0)    # [1, P, 2]
                ).pow(2)  # [N, P, 2]
                
                distances = torch.sqrt(squared_diffs.sum(dim=-1))  # [N, P]
                mask = distances > self.max_dist_ratio  # [N, P], True where attention should be masked
            
            # Encode patch positions using sinusoidal encoding
            uavsar_pos_encoded = self.positional_encoding(uavsar_patch_positions, self.position_encoding_dim)  # [P, pos_dim]
            
            # Combine patch features with positional encodings
            uavsar_features_with_pos = torch.cat([uavsar_embeddings.to(device), uavsar_pos_encoded], dim=1)  # [P, D_patch + pos_dim]
            
            # Create keys and values for cross-attention
            uavsar_keys = self.uavsar_key_proj(uavsar_features_with_pos)  # [P, D_p]
            uavsar_values = self.uavsar_value_proj(uavsar_features_with_pos)  # [P, D_p]
            
            # Compute attention output for UAVSAR with mask
            uavsar_attn_output = self.cross_attention(queries, uavsar_keys, uavsar_values, mask)  # [N, D_p]
            to_concat.append(uavsar_attn_output)
        
        # Combine outputs using concatenation and linear layers
        if len(to_concat) > 1:  # If we have at least one modality in addition to point features
            # Concatenate features
            concatenated = torch.cat(to_concat, dim=1)  # [N, D_p + ...] 
        else:
            # If no modalities used, just use the point features
            concatenated = point_features  # [N, D_p]

        # Apply linear layers for feature extraction and projection 
        concatenated = self.act(self.linear1(concatenated))  # [N, concat_dim or D_p]
        fused_features = self.linear2(concatenated)  # [N, D_p]
        
        # Apply normalization to the output features
        fused_features = self.norm2(fused_features)  # [N, D_p]
        
        return fused_features