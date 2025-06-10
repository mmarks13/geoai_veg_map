import warnings


import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Filter out other common warnings
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# # Environment variable to suppress NCCL warnings
# os.environ["NCCL_DEBUG"] = "WARN"
# # Make NCCL more robust to communication issues
# os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
os.environ["NCCL_DEBUG"] = "INFO"
# Increase timeout to 30 minutes
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800"  # in seconds
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000000"  # 1MB buffer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import required libraries
import torch
import gc
import os
gc.collect()
torch.cuda.empty_cache()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Load precomputed datasets
data_dir = "data/processed/model_data/"
train_data_path = os.path.join(data_dir, "augmented_training_tiles_60k.pt")
val_data_path = os.path.join(data_dir, "precomputed_validation_tiles.pt")
print("Loading precomputed data...")

train_data = torch.load(train_data_path)
val_data = torch.load(val_data_path)
print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")

# Create directories for checkpoints and logs
checkpoint_dir = "data/output/model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(checkpoint_dir, "final_best"), exist_ok=True)



import traceback
import gc
import os
from src.models.multimodal_model import MultimodalModelConfig
from src.training.multimodal_training import run_ablation_studies,train_multimodal_model
import random 
gc.collect()
torch.cuda.empty_cache()

# Define a base model configuration
base_config = MultimodalModelConfig(
    # Core model parameters (retained from original)
    fnl_attn_hds=2,      # Final attention heads
    feature_dim=128,     # Feature dimension
    k=15,                # Number of neighbors for KNN
    up_attn_hds=2,       # Upsampling attention heads
    up_ratio=2,          # Upsampling ratio
    pos_mlp_hdn=16,      # Hidden dimension for positional MLP
    up_concat=True,      # Whether to concatenate attention heads
    up_beta=False,       # Whether to use beta parameter
    up_dropout=0.005,    # upsample Dropout rate
    
    use_naip = True,
    use_uavsar = True,
    
    # Imagery encoder parameters
    img_embed_dim=64,    # Dimension of patch embeddings
    img_num_patches=16,  # Number of output patch embeddings
    naip_dropout=0.10,
    uavsar_dropout=0.10,
    temporal_encoder= 'transformer',
    
    # Fusion parameters
    fusion_type = 'cross_attention',
    max_dist_ratio=1.5,

    
    # cross attention fusion parameters
    fusion_dropout = 0.10,
    fusion_num_heads = 4,
    position_encoding_dim = 24
    

)


# Define checkpoint directory
checkpoint_dir = "data/output/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


try:
    # Run all four ablation studies sequentially
    run_ablation_studies(
        # train_dataset=random.sample(train_data, 2000),             
        # val_dataset=random.sample(val_data,300),           
        train_dataset=train_data,             
        val_dataset=val_data,       
        base_config=base_config,
        checkpoint_dir = checkpoint_dir,             
        batch_size=1,
        epochs=25,
        enable_debug=True,
        model_name="0330_aug_60k",
        lr_scheduler_type="onecycle",
        max_lr=4e-4,
        pct_start=0.15, 
        div_factor=25.0, 
        final_div_factor=1e2
    )
    
    # # # print("All ablation studies completed.")
    # train_multimodal_model(
    #     # train_dataset=random.sampletrain_data, 2000),
    #     # val_dataset=random.sample(val_data,300),
    #     train_dataset=train_data,
    #     val_dataset=val_data,
    #     model_name="0329_agmnt_TEST_uavse_naip",
    #     batch_size=1,
    #     checkpoint_dir=checkpoint_dir,
    #     model_config=base_config,
    #     num_epochs=1,
    #     lr_scheduler_type="onecycle",
    #     max_lr=4e-4,
    #     pct_start=0.15, 
    #     div_factor=25.0, 
    #     final_div_factor=1e2
    # )
    
    
except Exception as e:
    print("Training failed with an error:")
    traceback.print_exc()  
finally:
    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")