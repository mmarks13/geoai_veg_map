import os

import torch
import gc
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import traceback
import gc
import os
from src.models.multimodal_model import MultimodalModelConfig
from src.training.multimodal_training import run_ablation_studies,train_multimodal_model
import random 

def timestamp():
    # Returns the current time in Pacific Standard Time
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load precomputed datasets
data_dir = "data/processed/model_data/"
orig_train_data_path = os.path.join(data_dir, "precomputed_training_tiles.pt")
augmented_train_data_path = os.path.join(data_dir, "augmented_tiles_12k.pt")
val_data_path = os.path.join(data_dir, "precomputed_validation_tiles.pt")

print("Loading precomputed data...")

# Load original training data
print(f"[{timestamp()}] Starting load of original training data from {orig_train_data_path}")
start_time = time.time()
orig_train_data = torch.load(orig_train_data_path, weights_only=False)
elapsed = time.time() - start_time
print(f"[{timestamp()}] Finished load of original training data in {elapsed:.2f} seconds")

# Load augmented training data
print(f"[{timestamp()}] Starting load of augmented training data from {augmented_train_data_path}")
start_time = time.time()
augmented_train_data = torch.load(augmented_train_data_path, weights_only=False)
elapsed = time.time() - start_time
print(f"[{timestamp()}] Finished load of augmented training data in {elapsed:.2f} seconds")

# Load validation data
print(f"[{timestamp()}] Starting load of validation data from {val_data_path}")
start_time = time.time()
val_data = torch.load(val_data_path, weights_only=False)
elapsed = time.time() - start_time
print(f"[{timestamp()}] Finished load of validation data in {elapsed:.2f} seconds")

# Combine training data
train_data = orig_train_data + augmented_train_data

print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")






torch.cuda.empty_cache()


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
    
    # Deprecated/legacy parameters (grouped together)
    num_lcl_heads=4,      # Local attention heads (for backward compatibility)
    num_glbl_heads=4,     # Global attention heads (for backward compatibility)
    up_attn_hds=4,        # Legacy parameter (upsampling attention heads)
    fnl_attn_hds=2,       # Legacy parameter (final attention heads)
    up_concat=True,       # Legacy parameter (no longer used)
    up_beta=False,        # Legacy parameter
    
    # Modality flags
    use_naip=False,
    use_uavsar=False,
    
    # Imagery encoder parameters
    img_embed_dim=64,    # Dimension of patch embeddings
    img_num_patches=16,  # Number of output patch embeddings
    naip_dropout=0.05,
    uavsar_dropout=0.05,
    temporal_encoder='gru',
    
    # Fusion parameters
    fusion_type='cross_attention',
    max_dist_ratio=1.5,

    # Cross attention fusion parameters
    fusion_dropout=0.0,
    fusion_num_heads=1,
    position_encoding_dim=24,
    
    # Other parameters
    attr_dim=3,
    
    # Checkpoint parameters (commented out in the original)
    # checkpoint_path="data/output/checkpoints/baseline_20250402-073326/0402_baseline_k15_f256_b5_e30.pth",
    # checkpoint_path="/home/jovyan/geoai_veg_map/data/output/checkpoints/0404_huber2m_3DEP_baseline_baseline_k15_f256_b6_e80.pth",
    # layers_to_load=["feature_extractor.pt_conv1.convs.0.weight", "feature_extractor.pt_conv2.convs.0.weight"],
    # layers_to_freeze=["feature_extractor.pt_conv1.convs.0.weight"]  # Freeze a subset of loaded layers
)


# Define checkpoint directory
checkpoint_dir = "data/output/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


try:
    # #Run all four ablation studies sequentially
    # run_ablation_studies(
    #     # train_dataset=random.sample(train_data, 2000),             
    #     # val_dataset=random.sample(val_data,300),           
    #     train_dataset=train_data,             
    #     val_dataset=val_data,       
    #     base_config=base_config,
    #     checkpoint_dir = checkpoint_dir,   
    #     batch_size=5,
    #     epochs=30,
    #     enable_debug=True,
    #     model_name="0402",
    #     lr_scheduler_type="onecycle",
    #     max_lr=3e-4,
    #     pct_start=0.15, 
    #     div_factor=25.0, 
    #     final_div_factor=1e2
    # )
    
    # # print("All ablation studies completed.")
    train_multimodal_model(
        train_dataset=train_data,
        val_dataset=val_data,
        # train_dataset=random.sample(train_data, 100),             
        # val_dataset=random.sample(val_data,40),          
        model_name="0411_LGPA_5e4_256ft_b10_e15_heads_8-4-8-4-4-4",
        batch_size=12,
        checkpoint_dir=checkpoint_dir,
        model_config=base_config,
        enable_debug=True,
        num_epochs=15,
        lr_scheduler_type="onecycle",
        max_lr=5e-4,
        pct_start=0.1, 
        div_factor=25, 
        final_div_factor=1
    )


except Exception as e:
    print("Training failed with an error:")
    
    traceback.print_exc()  
finally:
    # Clean up GPU memory

    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")