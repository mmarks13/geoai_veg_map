import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    2D positional encoding for image patches
    """
    def __init__(self, embed_dim, max_h=10, max_w=10):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create a grid of normalized coordinates (between 0 and 1)
        h_pos = torch.linspace(0, 1, max_h).unsqueeze(1).expand(-1, max_w)  # [max_h, max_w]
        w_pos = torch.linspace(0, 1, max_w).unsqueeze(0).expand(max_h, -1)  # [max_h, max_w]
        
        # Reshape to [max_h*max_w, 2]
        pos = torch.stack([h_pos, w_pos], dim=-1).view(-1, 2)  # [max_h*max_w, 2]
        
        # Project 2D positions to same dimension as embeddings
        self.projection = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        self.register_buffer('pos', pos)  # [max_h*max_w, 2]
        
    def forward(self, x, n_patches=None):
        """
        Args:
            x: Patch embeddings [n_patches, embed_dim] or None
            n_patches: Number of patches if x is None
        Returns:
            Embeddings with positional encoding added [n_patches, embed_dim]
        """
        if x is None:
            if n_patches is None:
                raise ValueError("Either x or n_patches must be provided")
            # Just return positional encodings
            positions = self.pos[:n_patches]  # [n_patches, 2]
            return self.projection(positions)  # [n_patches, embed_dim]
        else:
            # Add positional encodings to input embeddings
            n_patches = x.size(0)
            positions = self.pos[:n_patches]  # [n_patches, 2]
            pos_enc = self.projection(positions)  # [n_patches, embed_dim]
            return x + pos_enc  # [n_patches, embed_dim]


import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings
    (Shifted-Patch Tokenisation variant)

    Args:
        in_channels (int):  Number of input image channels
        patch_size  (int):  Down-sampling factor (e.g. 10 for 40×40 → 4×4)
        embed_dim   (int):  Output feature dimension per patch
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size

        # --- Local “stem” with stride-1 convolutions -----------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim,       kernel_size=3, stride=1, padding=1),
        )

        # Down-sample to a 4×4 grid (stride = patch_size)
        self.pool = nn.AvgPool2d(patch_size)
        # Layer normalization over channel dimension
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, C, H, W] input image batch
        Returns:
            patch_embed (Tensor): [B, N, embed_dim] where
                                  N = (H // patch_size) × (W // patch_size)
        """
        # Local feature extraction
        x = self.stem(x)              # [B, embed_dim, H, W]
        # Spatial down-sampling
        x = self.pool(x)              # [B, embed_dim, H', W']  (H' = W' = 4)
        # Flatten patches
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim] (N = H'×W' = 16)
        # Per-patch normalization
        x = self.norm(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import StochasticDepth


# ----------------------------------------------------------------------
# Helper: MLP with depth-wise conv (ConViT trick)
# ----------------------------------------------------------------------
class MLPWithDWConv(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1   = nn.Linear(dim, hidden)
        self.act   = nn.GELU()
        self.dwconv = nn.Conv1d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2   = nn.Linear(hidden, dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):                  # x: [B, N, D]
        x = self.act(self.fc1(x))          # [B, N, hidden]
        # depth-wise conv over the sequence
        x = x.transpose(1, 2)              # [B, hidden, N]
        x = self.dwconv(x)
        x = x.transpose(1, 2)              # back to [B, N, hidden]
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)                # [B, N, D]


# ----------------------------------------------------------------------
# TransformerEncoderBlock
# ----------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()

        # 1) Self-attention branch
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, num_heads=4,
                                            dropout=dropout, batch_first=True)
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(dim))
        self.drop1  = StochasticDepth(drop_path, mode="row")

        # 2) Feed-forward branch
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = MLPWithDWConv(dim, mlp_ratio=4, dropout=dropout)
        self.gamma2 = nn.Parameter(1e-5 * torch.ones(dim))
        self.drop2  = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 2:                   # [N, D]  → add batch dim
            x, single = x.unsqueeze(0), True

        # self-attention + residual
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         need_weights=False)
        x = x + self.drop1(self.gamma1 * a)

        # MLP + residual
        y = self.ffn(self.norm2(x))
        x = x + self.drop2(self.gamma2 * y)

        return x.squeeze(0) if single else x




class TemporalGRUEncoder(nn.Module):
    """
    GRU + attention-pool encoder for aggregating temporal information

    The GRU models order; a lightweight attention layer learns
    which acquisition dates are most informative.
    """
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim

        # ------------------------------------------------------------------
        # 1) Bidirectional GRU over the temporal dimension
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )                                               # output → [N, T, 2*hidden_dim]

        # ------------------------------------------------------------------
        # 2) Attention weights α_t  (one scalar per timestep)
        # ------------------------------------------------------------------
        self.attn_fc = nn.Linear(self.hidden_dim * 2, 1)  # [N, T, 1]

        # ------------------------------------------------------------------
        # 3) Final projection back to embed_dim
        # ------------------------------------------------------------------
        self.projection = nn.Linear(self.hidden_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Patch embeddings across time  [T, N, D]
                T : number of temporal acquisitions
                N : number of patches
                D : embedding dimension
        Returns:
            aggregated (Tensor): Temporally aggregated embeddings [N, D]
        """
        T, N, D = x.shape                                 # [T, N, D]

        # --------------------------------------------------------------
        # Re-arrange so each patch’s time series is processed together
        # --------------------------------------------------------------
        x = x.permute(1, 0, 2)                            # [N, T, D]

        # --------------------------------------------------------------
        # Pass through bi-GRU
        # --------------------------------------------------------------
        h, _ = self.gru(x)                                # [N, T, 2*hidden_dim]

        # --------------------------------------------------------------
        # Compute attention weights and apply softmax over the T dimension
        # --------------------------------------------------------------
        alpha = F.softmax(self.attn_fc(h), dim=1)         # [N, T, 1]

        # --------------------------------------------------------------
        # Weighted temporal average
        # --------------------------------------------------------------
        context = (alpha * h).sum(dim=1)                  # [N, 2*hidden_dim]

        # --------------------------------------------------------------
        # Project back to embed_dim
        # --------------------------------------------------------------
        aggregated = self.projection(context)             # [N, D]

        return aggregated



class TemporalDifferenceModule(nn.Module):
    """
    Module that computes and processes temporal differences between frames
    with learned non-linear aggregation and consideration of time intervals
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        
        # MLP for processing frame differences with time interval information
        self.diff_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),  # +1 for time interval
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Attention-based aggregation for differences
        self.diff_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,  
            dropout=dropout,
            batch_first=True
        )
        
        # Query vector for attention-based aggregation
        self.query_vector = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query_vector, mean=0, std=0.02)
        
        # Layer norm for pre-attention normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, relative_dates=None):
        """
        Args:
            x: Temporal sequence of embeddings [N, T, D]
                N: Number of patches
                T: Number of temporal frames
                D: Embedding dimension
            relative_dates: Relative days from reference date [T, 1] or [T]
        Returns:
            diff_features: Aggregated difference features [N, D]
        """
        N, T, D = x.shape
        
        # Skip if we have only one frame
        if T <= 1:
            return torch.zeros(N, D, device=x.device)
        
        # Compute temporal differences (embeddings)
        diffs = x[:, 1:, :] - x[:, :-1, :]  # [N, T-1, D]
        
        # Process time intervals if available
        if relative_dates is not None:
            # Ensure relative_dates has the right shape
            if relative_dates.dim() == 1:
                relative_dates = relative_dates.unsqueeze(1)  # [T, 1]
            
            # Compute time intervals between consecutive frames
            time_diffs = relative_dates[1:] - relative_dates[:-1]  # [T-1, 1]
            
            # Normalize time differences
            # This helps with numerical stability
            time_diffs = time_diffs / (time_diffs.abs().max() + 1e-6)
            
            # Expand time differences to match batch dimension
            time_diffs = time_diffs.expand(N, T-1, 1)  # [N, T-1, 1]
            
            # Concatenate embedding differences with time differences
            diffs_with_time = torch.cat([diffs, time_diffs], dim=2)  # [N, T-1, D+1]
            
            # Process differences through MLP
            processed_diffs = self.diff_mlp(diffs_with_time)  # [N, T-1, D]
        else:
            # If no time information, just process the embedding differences
            processed_diffs = self.diff_mlp(diffs)  # [N, T-1, D]
        
        # Apply layer normalization before attention
        processed_diffs = self.norm(processed_diffs)  # [N, T-1, D]
        
        # Expand query vector to batch size
        query = self.query_vector.expand(N, -1, -1)  # [N, 1, D]
        
        # Attention-based aggregation of differences
        diff_agg, _ = self.diff_attention(
            query=query,                 # [N, 1, D]
            key=processed_diffs,         # [N, T-1, D]
            value=processed_diffs        # [N, T-1, D]
        )  # [N, 1, D]
        
        # Remove singleton dimension
        diff_agg = diff_agg.squeeze(1)  # [N, D]
        
        return diff_agg



class TemporalTransformerEncoder(nn.Module):
    """
    Lightweight transformer-based encoder for aggregating temporal information
    with support for relative date-based positional encoding and temporal difference module for better change detection
    """
    def __init__(
        self, 
        embed_dim, 
        num_heads=2, 
        dropout=0.05, 
        num_layers=2,
        ff_multiplier=2,
        max_temporal_len=20
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Temporal positional encoding (fixed fallback)
        self.temporal_pos_encoding = nn.Parameter(torch.zeros(1, max_temporal_len, embed_dim))
        nn.init.normal_(self.temporal_pos_encoding, mean=0, std=0.02)
        
        # Date-based positional encoding projection
        self.date_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * ff_multiplier),
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Global temporal token for aggregation
        self.temporal_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.temporal_token, mean=0, std=0.02)
        
        # Temporal difference module
        self.diff_module = TemporalDifferenceModule(
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Fusion layer to combine transformer and difference features
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x, relative_dates=None):
        """
        Args:
            x: Patch embeddings across temporal dimension [T, N, D]
                T: Number of temporal acquisitions
                N: Number of patches per acquisition
                D: Embedding dimension
            relative_dates: Relative days from UAV acquisition date [T, 1]
        Returns:
            aggregated: Temporally aggregated patch embeddings [N, D]
        """
        T, N, D = x.shape  # [T, N, D]
        
        # Reshape to process each patch separately through time
        x = x.permute(1, 0, 2)  # [N, T, D]
        
        # Add positional encodings based on relative dates if available
        if relative_dates is not None:
            # Ensure relative_dates has the right shape
            if relative_dates.dim() == 1:
                relative_dates = relative_dates.unsqueeze(1)  # [T, 1]
                
            # Generate date-based positional encodings
            date_pos_enc = self.date_embedding(relative_dates)  # [T, D]
            
            # Expand to match batch dimension
            date_pos_enc = date_pos_enc.unsqueeze(0).expand(N, -1, -1)  # [N, T, D]
            
            # Apply date-based positional encoding
            x_with_pos = x + date_pos_enc  # [N, T, D]
        else:
            # Fallback to standard positional encoding
            pos_enc = self.temporal_pos_encoding[:, :T, :]
            x_with_pos = x + pos_enc  # [N, T, D]
        
        # Process through temporal difference module
        diff_features = self.diff_module(x_with_pos, relative_dates)  # [N, D]
        
        # Prepend temporal token for transformer processing
        temp_tokens = self.temporal_token.expand(N, -1, -1)  # [N, 1, D]
        x_with_token = torch.cat([temp_tokens, x_with_pos], dim=1)  # [N, T+1, D]
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x_with_token)  # [N, T+1, D]
        
        # Extract the temporal token which has aggregated information
        transformer_features = transformer_output[:, 0, :]  # [N, D]
        
        # Combine transformer features with difference features
        combined = torch.cat([transformer_features, diff_features], dim=1)  # [N, 2*D]
        
        # Fuse the features
        fused = self.fusion(combined)  # [N, D]
        
        # Normalize the final features
        aggregated = F.normalize(fused, p=2, dim=1)
        
        return aggregated


class NAIPEncoder(nn.Module):
    """
    Encoder for NAIP optical imagery with transformer-based processing
    """
    def __init__(
        self,
        in_channels=4,          # RGB + NIR
        image_size=40,          # 40x40 pixels
        patch_size=10,          # 10x10 pixel patches
        embed_dim=32,           # Dimension of patch embeddings
        num_patches=16,         # Number of output patch embeddings
        dropout=0.1,            # Dropout rate
        temporal_encoder_type='gru',  # Type of temporal encoder: 'gru' or 'transformer'
        # Additional parameters for transformer encoder
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_ff_multiplier=2,
        max_temporal_len=32
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Calculate number of patches along each dimension
        n_patches_h = image_size // patch_size
        n_patches_w = image_size // patch_size
        self.n_patches = n_patches_h * n_patches_w
        
        # Verify that the requested number of patches matches what the image dimensions will produce
        if self.n_patches != num_patches:
            raise ValueError(
                f"Number of patches mismatch: {num_patches} requested but image dimensions "
                f"({image_size}x{image_size}) with patch size {patch_size} produces {self.n_patches} patches. "
                f"Please adjust image size or patch size to get {num_patches} patches."
            )
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Positional encoding (applied before transformer)
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_h=n_patches_h,
            max_w=n_patches_w
        )
        
        # Transformer encoder block
        self.transformer_block = TransformerEncoderBlock(
            dim=embed_dim,
            dropout=dropout
        )
        
        # Temporal encoder for aggregating across time
        if temporal_encoder_type == 'gru':
            self.temporal_encoder = TemporalGRUEncoder(embed_dim=embed_dim)
        elif temporal_encoder_type == 'transformer':
            self.temporal_encoder = TemporalTransformerEncoder(
                embed_dim=embed_dim,
                num_heads=transformer_num_heads,
                dropout=dropout,
                num_layers=transformer_num_layers,
                ff_multiplier=transformer_ff_multiplier,
                max_temporal_len=max_temporal_len
            )
        else:
            raise ValueError(f"Unsupported temporal encoder type: {temporal_encoder_type}. Use 'gru' or 'transformer'.")
    
    
    def forward(self, x, img_bbox=None, relative_dates=None):
        """
        Args:
            x: NAIP images [T, C, H, W] where T is number of temporal acquisitions
            img_bbox: Bounding box information for spatial alignment
            relative_dates: Relative days from UAV acquisition date [T, 1]
        Returns:
            patch_embed: Patch embeddings with positional information [num_patches, embed_dim]
        """
        device = next(self.parameters()).device
        
        if x is None or x.shape[0] == 0:
            # Return zero embeddings if no NAIP data is provided
            return torch.zeros(self.num_patches, self.embed_dim, device=device)  # [num_patches, embed_dim]
            
        T, C, H, W = x.shape  # [T, C, H, W]
        
        # Process each temporal acquisition
        patch_embeds = []
        for t in range(T):
            # Get patch embeddings for this acquisition
            patches = self.patch_embed(x[t].unsqueeze(0))  # [1, n_patches, embed_dim]
            patches = patches.squeeze(0)  # [n_patches, embed_dim]
            
            # Add positional encoding
            patches = self.pos_encoding(patches)  # [n_patches, embed_dim]
            
            # Apply transformer
            patches = self.transformer_block(patches)  # [n_patches, embed_dim]
            
            patch_embeds.append(patches)
        
        # Stack along temporal dimension
        patch_embeds = torch.stack(patch_embeds)  # [T, n_patches, embed_dim]
        
        
        # Aggregate across temporal dimension, using dates if available
        if isinstance(self.temporal_encoder, TemporalTransformerEncoder) and relative_dates is not None:
            aggregated = self.temporal_encoder(patch_embeds, relative_dates)
        else:
            aggregated = self.temporal_encoder(patch_embeds)
        
        # Normalize features
        aggregated = F.normalize(aggregated, p=2, dim=1)
        
        return aggregated



class UAVSAREncoder(nn.Module):
    """
    Encoder for UAVSAR imagery with transformer-based processing
    """
    def __init__(
        self,
        in_channels=6,          # 6 polarization bands
        image_size=4,           # 4x4 pixels
        patch_size=1,           # 1x1 pixel patches (treat each pixel as a patch)
        embed_dim=32,           # Dimension of patch embeddings
        num_patches=16,         # Number of output patch embeddings
        dropout=0.1,            # Dropout rate
        temporal_encoder_type='gru',  # Type of temporal encoder: 'gru' or 'transformer'
        # Additional parameters for transformer encoder
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_ff_multiplier=2,
        max_temporal_len=32
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # For UAVSAR, we have a 4x4 grid, so 16 patches total when patch_size=1
        self.n_patches = (image_size // patch_size) ** 2
        
        # Verify that the requested number of patches matches what the image dimensions will produce
        if self.n_patches != num_patches:
            raise ValueError(
                f"Number of patches mismatch: {num_patches} requested but image dimensions "
                f"({image_size}x{image_size}) with patch size {patch_size} produces {self.n_patches} patches. "
                f"Please adjust image size or patch size to get {num_patches} patches."
            )
        
        # Initial embedding of each pixel/patch
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.GELU()
        )
        
        # Positional encoding (applied before transformer)
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_h=image_size,
            max_w=image_size
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Transformer encoder block
        self.transformer_block = TransformerEncoderBlock(
            dim=embed_dim,
            dropout=dropout
        )
        
        # Temporal encoder for aggregating across time
        if temporal_encoder_type == 'gru':
            self.temporal_encoder = TemporalGRUEncoder(embed_dim=embed_dim)
        elif temporal_encoder_type == 'transformer':
            self.temporal_encoder = TemporalTransformerEncoder(
                embed_dim=embed_dim,
                num_heads=transformer_num_heads,
                dropout=dropout,
                num_layers=transformer_num_layers,
                ff_multiplier=transformer_ff_multiplier,
                max_temporal_len=max_temporal_len
            )
        else:
            raise ValueError(f"Unsupported temporal encoder type: {temporal_encoder_type}. Use 'gru' or 'transformer'.")
    
    def forward(self, x, attention_mask=None, img_bbox=None, relative_dates=None):
        """
        Args:
            x: UAVSAR images [T, G_max, C, H, W] where:
            - T is the number of UAVSAR acquisition events
            - G_max is the maximum number of images per event (includes padding)
            - C is number of channels (polarization bands)
            - H, W are spatial dimensions
            attention_mask: Boolean mask [T, G_max] indicating valid (non-padded) images
            img_bbox: Bounding box information for spatial alignment
            relative_dates: Relative days from UAV acquisition date [T, 1]
        Returns:
            patch_embed: Patch embeddings with positional information [num_patches, embed_dim]
        """
        device = next(self.parameters()).device
        
        if x is None or x.shape[0] == 0:
            # Return zero embeddings if no UAVSAR data is provided
            return torch.zeros(self.num_patches, self.embed_dim, device=device)  # [num_patches, embed_dim]
        
        T, G_max, C, H, W = x.shape  # [T, G_max, C, H, W]
        
        # If no attention mask provided, assume all images are valid
        if attention_mask is None:
            attention_mask = torch.ones((T, G_max), dtype=torch.bool, device=x.device)
        
        # Reshape for batch processing through spatial encoder
        x_reshaped = x.view(T * G_max, C, H, W)  # [T*G_max, C, H, W]
        
        # Apply initial embedding
        patches = self.patch_embed(x_reshaped)  # [T*G_max, embed_dim, H, W]
        
        # Reshape to [T*G_max, H*W, embed_dim]
        patches = patches.flatten(2).transpose(1, 2)  # [T*G_max, H*W, embed_dim]
        
        # Apply normalization
        patches = self.norm(patches)  # [T*G_max, H*W, embed_dim]
        
        # Apply positional encoding efficiently using broadcasting
        n_patches = H * W
        # Get positional encoding once (doesn't depend on the batch content)
        pos_encoding = self.pos_encoding(None, n_patches)  # [n_patches, embed_dim]
        
        # Apply to all elements in batch via broadcasting
        # Reshape patches to [T*G_max, n_patches, embed_dim]
        patches = patches + pos_encoding.unsqueeze(0)  # Broadcasting adds to each item in batch
        
        # Apply transformer block
        patches = self.transformer_block(patches)  # [T*G_max, H*W, embed_dim]
        
        # Reshape back to [T, G_max, n_patches, embed_dim]
        spatial_embeddings = patches.view(T, G_max, n_patches, self.embed_dim)
        
        # Perform masked averaging across G_max dimension
        # First, create a float mask for arithmetic operations
        float_mask = attention_mask.float().unsqueeze(-1).unsqueeze(-1)  # [T, G_max, 1, 1]
        
        # Zero out embeddings where mask is False
        masked_embeddings = spatial_embeddings * float_mask  # [T, G_max, n_patches, embed_dim]
        
        # Sum across G_max dimension
        summed_embeddings = masked_embeddings.sum(dim=1)  # [T, n_patches, embed_dim]
        
        # Count number of True values in mask per timestep
        # Add small epsilon to avoid division by zero
        counts = attention_mask.sum(dim=1).float().clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)  # [T, 1, 1]
        
        # Divide sum by count to get average
        avg_embeddings = summed_embeddings / counts  # [T, n_patches, embed_dim]
        
        # Aggregate across temporal dimension using temporal encoder
        if isinstance(self.temporal_encoder, TemporalTransformerEncoder) and relative_dates is not None:
            aggregated = self.temporal_encoder(avg_embeddings, relative_dates)
        else:
            aggregated = self.temporal_encoder(avg_embeddings)
        
        # Normalize features
        aggregated = F.normalize(aggregated, p=2, dim=1)
        
        return aggregated 