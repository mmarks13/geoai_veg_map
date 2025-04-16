import os
import torch
import gc
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import traceback
from src.models.multimodal_model import MultimodalModelConfig
from src.training.multimodal_training import train_multimodal_model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load precomputed datasets
    data_dir = "data/processed/model_data/"
    orig_train_data_path = os.path.join(data_dir, "precomputed_training_tiles.pt")
    augmented_train_data_path = os.path.join(data_dir, "augmented_tiles_12k.pt")
    val_data_path = os.path.join(data_dir, "precomputed_validation_tiles.pt")

    # Define a base model configuration
    base_config = MultimodalModelConfig(
        # Core model parameters
        feature_dim=256,     # Feature dimension
        k=15,                # Number of neighbors for KNN
        up_ratio=2,          # Upsampling ratio
        pos_mlp_hdn=16,      # Hidden dimension for positional MLP
        pt_attn_dropout=0.05,
        
        # Granular attention head configurations
        extractor_lcl_heads=8,  # Local attention heads for feature extractor
        extractor_glbl_heads=4,  # Global attention heads for feature extractor
        expansion_lcl_heads=8,  # Local attention heads for feature expansion
        expansion_glbl_heads=4,  # Global attention heads for feature expansion
        refinement_lcl_heads=4,  # Local attention heads for feature refinement
        refinement_glbl_heads=4,  # Global attention heads for feature refinement
        
        # Deprecated/legacy parameters
        num_lcl_heads=4,      # Local attention heads (for backward compatibility)
        num_glbl_heads=4,     # Global attention heads (for backward compatibility)
        up_attn_hds=4,        # Legacy parameter (upsampling attention heads)
        fnl_attn_hds=2,       # Legacy parameter (final attention heads)
        up_concat=True,       # Legacy parameter (no longer used)
        up_beta=False,        # Legacy parameter
        
        # Modality flags
        use_naip=True,
        use_uavsar=True,
        
        # Imagery encoder parameters
        img_embed_dim=128,    # Dimension of patch embeddings
        img_num_patches=16,  # Number of output patch embeddings
        naip_dropout=0.05,
        uavsar_dropout=0.05,
        temporal_encoder='transformer',  # Type of temporal encoder
        
        # Fusion parameters
        fusion_type='cross_attention',
        max_dist_ratio=1.5,

        # Cross attention fusion parameters
        fusion_dropout=0.05,
        fusion_num_heads=4,
        position_encoding_dim=24,
        
        # Other parameters
        attr_dim=3,
        
    #     # # Checkpoint parameters
        checkpoint_path="/home/jovyan/geoai_veg_map/data/output/checkpoints/0414_LGPA_final_baseline_k15_f256_b8_e40.pth",
        # layers_to_load=[
        #     "feature_extractor.point_transformer.convs.0.lin.weight",
        #     "feature_extractor.point_transformer.convs.0.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.0.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.0.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.0.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.1.lin.weight",
        #     "feature_extractor.point_transformer.convs.1.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.1.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.1.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.1.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.2.lin.weight",
        #     "feature_extractor.point_transformer.convs.2.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.2.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.2.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.2.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.3.lin.weight",
        #     "feature_extractor.point_transformer.convs.3.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.3.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.3.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.3.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.4.lin.weight",
        #     "feature_extractor.point_transformer.convs.4.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.4.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.4.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.4.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.5.lin.weight",
        #     "feature_extractor.point_transformer.convs.5.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.5.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.5.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.5.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.6.lin.weight",
        #     "feature_extractor.point_transformer.convs.6.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.6.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.6.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.6.pos_nn.weight",
        #     "feature_extractor.point_transformer.convs.7.lin.weight",
        #     "feature_extractor.point_transformer.convs.7.lin_dst.weight",
        #     "feature_extractor.point_transformer.convs.7.lin_src.weight",
        #     "feature_extractor.point_transformer.convs.7.pos_nn.bias",
        #     "feature_extractor.point_transformer.convs.7.pos_nn.weight",
        #     "feature_extractor.point_transformer.projection.0.bias",
        #     "feature_extractor.point_transformer.projection.0.weight",
        #     "feature_extractor.point_transformer.projection.2.bias",
        #     "feature_extractor.point_transformer.projection.2.weight",
        #     "feature_extractor.point_transformer.ffn.0.bias",
        #     "feature_extractor.point_transformer.ffn.0.weight",
        #     "feature_extractor.point_transformer.ffn.2.bias",
        #     "feature_extractor.point_transformer.ffn.2.weight",
        #     "feature_extractor.point_transformer.ffn_norm.bias",
        #     "feature_extractor.point_transformer.ffn_norm.weight",
        #     "feature_extractor.pos_flash_attention.k_proj.bias",
        #     "feature_extractor.pos_flash_attention.k_proj.weight",
        #     "feature_extractor.pos_flash_attention.norm1.bias",
        #     "feature_extractor.pos_flash_attention.norm1.weight",
        #     "feature_extractor.pos_flash_attention.norm2.bias",
        #     "feature_extractor.pos_flash_attention.norm2.weight",
        #     "feature_extractor.pos_flash_attention.out_proj.bias",
        #     "feature_extractor.pos_flash_attention.out_proj.weight",
        #     "feature_extractor.pos_flash_attention.q_proj.bias",
        #     "feature_extractor.pos_flash_attention.q_proj.weight",
        #     "feature_extractor.pos_flash_attention.v_proj.bias",
        #     "feature_extractor.pos_flash_attention.v_proj.weight",
        # ]
    )

    # Define checkpoint directory
    checkpoint_dir = "data/output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        train_multimodal_model(
            train_data_paths=[orig_train_data_path, augmented_train_data_path],
            val_data_path=val_data_path,
            model_name="0415_fusion_e50_2e-4_full_backbone_dlr3",
            batch_size=8,
            checkpoint_dir=checkpoint_dir,
            model_config=base_config,
            enable_debug=True,
            num_epochs=50,
            lr_scheduler_type="onecycle",
            max_lr=2e-4,
            pct_start=0.06, 
            div_factor=25, 
            final_div_factor=100,
            dscrm_lr_ratio=3,
            # Optional: use cached shards if they exist
            use_cached_shards=True,
            shard_cache_dir="data/output/cached_shards"
        )
    except Exception as e:
        print("Training failed with an error:")
        traceback.print_exc()
    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared")


    # Define a base model configuration
    base_config = MultimodalModelConfig(
        # Core model parameters
        feature_dim=256,     # Feature dimension
        k=15,                # Number of neighbors for KNN
        up_ratio=2,          # Upsampling ratio
        pos_mlp_hdn=16,      # Hidden dimension for positional MLP
        pt_attn_dropout=0.05,
        
        # Granular attention head configurations
        extractor_lcl_heads=8,  # Local attention heads for feature extractor
        extractor_glbl_heads=4,  # Global attention heads for feature extractor
        expansion_lcl_heads=8,  # Local attention heads for feature expansion
        expansion_glbl_heads=4,  # Global attention heads for feature expansion
        refinement_lcl_heads=4,  # Local attention heads for feature refinement
        refinement_glbl_heads=4,  # Global attention heads for feature refinement
        
        # Deprecated/legacy parameters
        num_lcl_heads=4,      # Local attention heads (for backward compatibility)
        num_glbl_heads=4,     # Global attention heads (for backward compatibility)
        up_attn_hds=4,        # Legacy parameter (upsampling attention heads)
        fnl_attn_hds=2,       # Legacy parameter (final attention heads)
        up_concat=True,       # Legacy parameter (no longer used)
        up_beta=False,        # Legacy parameter
        
        # Modality flags
        use_naip=True,
        use_uavsar=True,
        
        # Imagery encoder parameters
        img_embed_dim=128,    # Dimension of patch embeddings
        img_num_patches=16,  # Number of output patch embeddings
        naip_dropout=0.05,
        uavsar_dropout=0.05,
        temporal_encoder='transformer',  # Type of temporal encoder
        
        # Fusion parameters
        fusion_type='cross_attention',
        max_dist_ratio=1.5,

        # Cross attention fusion parameters
        fusion_dropout=0.05,
        fusion_num_heads=4,
        position_encoding_dim=24,
        
        # Other parameters
        attr_dim=3,
        
    #     # # Checkpoint parameters
        checkpoint_path="/home/jovyan/geoai_veg_map/data/output/checkpoints/0414_LGPA_final_baseline_k15_f256_b8_e40.pth",
        layers_to_load=[
            "feature_extractor.point_transformer.convs.0.lin.weight",
            "feature_extractor.point_transformer.convs.0.lin_dst.weight",
            "feature_extractor.point_transformer.convs.0.lin_src.weight",
            "feature_extractor.point_transformer.convs.0.pos_nn.bias",
            "feature_extractor.point_transformer.convs.0.pos_nn.weight",
            "feature_extractor.point_transformer.convs.1.lin.weight",
            "feature_extractor.point_transformer.convs.1.lin_dst.weight",
            "feature_extractor.point_transformer.convs.1.lin_src.weight",
            "feature_extractor.point_transformer.convs.1.pos_nn.bias",
            "feature_extractor.point_transformer.convs.1.pos_nn.weight",
            "feature_extractor.point_transformer.convs.2.lin.weight",
            "feature_extractor.point_transformer.convs.2.lin_dst.weight",
            "feature_extractor.point_transformer.convs.2.lin_src.weight",
            "feature_extractor.point_transformer.convs.2.pos_nn.bias",
            "feature_extractor.point_transformer.convs.2.pos_nn.weight",
            "feature_extractor.point_transformer.convs.3.lin.weight",
            "feature_extractor.point_transformer.convs.3.lin_dst.weight",
            "feature_extractor.point_transformer.convs.3.lin_src.weight",
            "feature_extractor.point_transformer.convs.3.pos_nn.bias",
            "feature_extractor.point_transformer.convs.3.pos_nn.weight",
            "feature_extractor.point_transformer.convs.4.lin.weight",
            "feature_extractor.point_transformer.convs.4.lin_dst.weight",
            "feature_extractor.point_transformer.convs.4.lin_src.weight",
            "feature_extractor.point_transformer.convs.4.pos_nn.bias",
            "feature_extractor.point_transformer.convs.4.pos_nn.weight",
            "feature_extractor.point_transformer.convs.5.lin.weight",
            "feature_extractor.point_transformer.convs.5.lin_dst.weight",
            "feature_extractor.point_transformer.convs.5.lin_src.weight",
            "feature_extractor.point_transformer.convs.5.pos_nn.bias",
            "feature_extractor.point_transformer.convs.5.pos_nn.weight",
            "feature_extractor.point_transformer.convs.6.lin.weight",
            "feature_extractor.point_transformer.convs.6.lin_dst.weight",
            "feature_extractor.point_transformer.convs.6.lin_src.weight",
            "feature_extractor.point_transformer.convs.6.pos_nn.bias",
            "feature_extractor.point_transformer.convs.6.pos_nn.weight",
            "feature_extractor.point_transformer.convs.7.lin.weight",
            "feature_extractor.point_transformer.convs.7.lin_dst.weight",
            "feature_extractor.point_transformer.convs.7.lin_src.weight",
            "feature_extractor.point_transformer.convs.7.pos_nn.bias",
            "feature_extractor.point_transformer.convs.7.pos_nn.weight",
            "feature_extractor.point_transformer.projection.0.bias",
            "feature_extractor.point_transformer.projection.0.weight",
            "feature_extractor.point_transformer.projection.2.bias",
            "feature_extractor.point_transformer.projection.2.weight",
            "feature_extractor.point_transformer.ffn.0.bias",
            "feature_extractor.point_transformer.ffn.0.weight",
            "feature_extractor.point_transformer.ffn.2.bias",
            "feature_extractor.point_transformer.ffn.2.weight",
            "feature_extractor.point_transformer.ffn_norm.bias",
            "feature_extractor.point_transformer.ffn_norm.weight",
            "feature_extractor.pos_flash_attention.k_proj.bias",
            "feature_extractor.pos_flash_attention.k_proj.weight",
            "feature_extractor.pos_flash_attention.norm1.bias",
            "feature_extractor.pos_flash_attention.norm1.weight",
            "feature_extractor.pos_flash_attention.norm2.bias",
            "feature_extractor.pos_flash_attention.norm2.weight",
            "feature_extractor.pos_flash_attention.out_proj.bias",
            "feature_extractor.pos_flash_attention.out_proj.weight",
            "feature_extractor.pos_flash_attention.q_proj.bias",
            "feature_extractor.pos_flash_attention.q_proj.weight",
            "feature_extractor.pos_flash_attention.v_proj.bias",
            "feature_extractor.pos_flash_attention.v_proj.weight",
        ]
    )
    try:
        train_multimodal_model(
            train_data_paths=[orig_train_data_path, augmented_train_data_path],
            val_data_path=val_data_path,
            model_name="0415_fusion_e50_25e-5_ftextr_backbone_dlr5",
            batch_size=8,
            checkpoint_dir=checkpoint_dir,
            model_config=base_config,
            enable_debug=True,
            num_epochs=50,
            lr_scheduler_type="onecycle",
            max_lr=2.5e-4,
            pct_start=0.1, 
            div_factor=25, 
            final_div_factor=10,
            dscrm_lr_ratio=5,
            # Optional: use cached shards if they exist
            use_cached_shards=True,
            shard_cache_dir="data/output/cached_shards"
        )
    except Exception as e:
        print("Training failed with an error:")
        traceback.print_exc()
    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared")
if __name__ == "__main__":
    main()