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


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings
    """
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Image tensor [B, C, H, W]
        Returns:
            patch_embed: Patch embeddings [B, N, embed_dim]
                where N = (H // patch_size) * (W // patch_size)
        """
        # Project patches
        x = self.proj(x)  # [B, embed_dim, H', W'] where H' = H//patch_size, W' = W//patch_size
        
        # Rearrange to [B, H'*W', embed_dim]
        B, C, H, W = x.shape  # C is now embed_dim
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        
        # Apply normalization
        x = self.norm(x)  # [B, H'*W', embed_dim]
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with single-headed self-attention
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(dim)
        # Self-attention layer 
        self.attn = nn.MultiheadAttention(dim, num_heads=2, dropout=dropout, batch_first=True)
        # Layer normalization for feed-forward
        self.norm2 = nn.LayerNorm(dim)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        """
        Args:
            x: Input tensor [N, D] or [B, N, D]
                N: Number of patches
                D: Embedding dimension
        Returns:
            out: Transformed tensor with same shape as input
        """
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]
            single_batch = True
        else:
            single_batch = False
            
        # Self-attention with residual connection
        norm_x = self.norm1(x)  # [B, N, D]
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)  # [B, N, D]
        x = x + self.dropout(attn_output)  # [B, N, D]
        
        # Feed-forward with residual connection
        norm_x = self.norm2(x)  # [B, N, D]
        ffn_output = self.ffn(norm_x)  # [B, N, D]
        x = x + self.dropout(ffn_output)  # [B, N, D]
        
        # Remove batch dimension if it was added
        if single_batch:
            x = x.squeeze(0)  # [N, D]
            
        return x


class TemporalGRUEncoder(nn.Module):
    """
    Simple GRU-based encoder for aggregating temporal information
    """
    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        
        # GRU to process temporal sequences
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional GRU output back to embed_dim
        self.projection = nn.Linear(self.hidden_dim * 2, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Patch embeddings across temporal dimension [T, N, D]
                T: Number of temporal acquisitions
                N: Number of patches per acquisition
                D: Embedding dimension
        Returns:
            aggregated: Temporally aggregated patch embeddings [N, D]
        """
        T, N, D = x.shape  # [T, N, D]
        
        # Reshape to process each patch separately through time
        x = x.permute(1, 0, 2)  # [N, T, D]
        
        # Apply GRU to each patch's temporal sequence
        output, _ = self.gru(x)  # output: [N, T, hidden_dim*2]
        
        # Take the final timestep's output
        final_output = output[:, -1, :]  # [N, hidden_dim*2]
        
        # Project back to embed_dim
        aggregated = self.projection(final_output)  # [N, D]
        
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
        dropout=0.1             # Dropout rate
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
            embed_dim=embed_dim  # Use full embedding dimension
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
        self.temporal_encoder = TemporalGRUEncoder(embed_dim=embed_dim)
    
    def forward(self, x, img_bbox=None):
        """
        Args:
            x: NAIP images [T, C, H, W] where T is the number of temporal acquisitions
            img_bbox: Bounding box information for spatial alignment
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
        
        # Aggregate across temporal dimension
        aggregated = self.temporal_encoder(patch_embeds)  # [n_patches, embed_dim]
        
        # Normalize features
        aggregated = F.normalize(aggregated, p=2, dim=1)  # [n_patches, embed_dim]
        
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
        dropout=0.1             # Dropout rate
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
        self.temporal_encoder = TemporalGRUEncoder(embed_dim=embed_dim)
    
    def forward(self, x, img_bbox=None):
        """
        Args:
            x: UAVSAR images [T, C, H, W] where T is the number of temporal acquisitions
            img_bbox: Bounding box information for spatial alignment
        Returns:
            patch_embed: Patch embeddings with positional information [num_patches, embed_dim]
        """
        device = next(self.parameters()).device
        
        if x is None or x.shape[0] == 0:
            # Return zero embeddings if no UAVSAR data is provided
            return torch.zeros(self.num_patches, self.embed_dim, device=device)  # [num_patches, embed_dim]
            
        T, C, H, W = x.shape  # [T, C, H, W]
        
        patch_embeds = []
        for t in range(T):
            # Apply initial embedding
            patches = self.patch_embed(x[t].unsqueeze(0))  # [1, embed_dim, H, W]
            
            # Reshape to [1, H*W, embed_dim]
            patches = patches.flatten(2).transpose(1, 2)  # [1, H*W, embed_dim]
            
            # Apply normalization
            patches = self.norm(patches)  # [1, H*W, embed_dim]
            patches = patches.squeeze(0)  # [H*W, embed_dim] = [n_patches, embed_dim]
            
            # Add positional encoding
            patches = self.pos_encoding(patches)  # [n_patches, embed_dim]
            
            # Apply transformer
            patches = self.transformer_block(patches)  # [n_patches, embed_dim]
            
            patch_embeds.append(patches)
        
        # Stack along temporal dimension
        patch_embeds = torch.stack(patch_embeds)  # [T, n_patches, embed_dim]
        
        # Aggregate across temporal dimension
        aggregated = self.temporal_encoder(patch_embeds)  # [n_patches, embed_dim]
        
        # Normalize features
        aggregated = F.normalize(aggregated, p=2, dim=1)  # [n_patches, embed_dim]
        
        return aggregated