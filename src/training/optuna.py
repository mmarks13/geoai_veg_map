import os
import json
import torch
import optuna
from datetime import datetime
from dataclasses import asdict

# Import the MultimodalModelConfig from the original file
from src.models.multimodal_model import MultimodalModelConfig

def optuna_objective(trial, train_dataset, val_dataset, model_name, batch_size, checkpoint_dir,
                    num_epochs, log_file, early_stopping_patience, temp_dir_root, tensorboard_log_dir,
                    enable_debug, accumulation_steps, visualize_samples,
                    lr_scheduler_type, max_lr, pct_start, div_factor, final_div_factor,
                    use_naip=False, use_uavsar=False, k=15, attr_dim=3):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name of the model
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train
        log_file: Path to log file
        early_stopping_patience: Number of epochs to wait before early stopping
        temp_dir_root: Root directory for temporary files
        tensorboard_log_dir: Directory for TensorBoard logs
        enable_debug: Whether to enable debug mode
        accumulation_steps: Number of steps to accumulate gradients
        visualize_samples: Whether to visualize samples
        lr_scheduler_type: Type of learning rate scheduler
        max_lr: Maximum learning rate
        pct_start: Percentage of warmup in OneCycleLR
        div_factor: Division factor for OneCycleLR
        final_div_factor: Final division factor for OneCycleLR
        use_naip: Whether to use NAIP imagery
        use_uavsar: Whether to use UAVSAR imagery
        k: k-value for KNN (default: 15)
        attr_dim: Attribute dimension (default: 3)
        
    Returns:
        best_val_loss: Best validation loss achieved during training
    """
    from src.training.multimodal_training import train_multimodal_model
    
    # Define hyperparameter search space using Optuna
    feature_dim = 32 #trial.suggest_int('feature_dim', 64, 256, step=32)
    up_attn_hds = 1 #trial.suggest_int('up_attn_hds', 1, 2)
    up_dropout = .005 #trial.suggest_float('up_dropout', 0.0, 0.03)
    fnl_attn_hds = 2 #trial.suggest_int('fnl_attn_hds', 2, 3)
    img_embed_dim = 16#trial.suggest_int('img_embed_dim', 32, 64, step=32)
    
    # SpatialFusion hyperparameters
    temperature = 0.01 #trial.suggest_float('temperature', 0.01, 0.5, log=True)
    max_dist_ratio = 1.5 #trial.suggest_categorical("max_dist_ratio", [1.3,1.5,1.7, 10]) 
    

    
    # Encoder dropout parameters
    naip_dropout = 0.1 #trial.suggest_float('naip_dropout', 0.0, 0.3, step=0.1)
    uavsar_dropout = 0.1 #trial.suggest_float('uavsar_dropout', 0.1, 0.3, step=0.1)

    temporal_encoder ='transformer'# trial.suggest_categorical("temporal_encoder", ['gru', 'transformer']) 
    fusion = 'cross_attention' #trial.suggest_categorical("fusion", ['spatial', 'cross_attention']) 
    fusion_dropout = 0.10 #trial.suggest_float('fusion_dropout', 0.05, 0., step = 0.05)
    
    fusion_num_heads = 2 #trial.suggest_int('fusion_num_heads', 2, 6, step=2)
    position_encoding_dim = 12 #trial.suggest_int('position_encoding_dim', 12, 24, step=12)
    
    
    # Print trial information to console
    print("\n" + "="*80)
    print(f"Starting Trial #{trial.number}")
    print("="*80)
    print("Parameters:")
    print(f"  feature_dim: {feature_dim}")
    print(f"  up_attn_hds: {up_attn_hds}")
    print(f"  up_dropout: {up_dropout:.4f}")
    print(f"  fnl_attn_hds: {fnl_attn_hds}")
    print(f"  img_embed_dim: {img_embed_dim}")
    print(f"  temperature: {temperature:.4f}")
    print(f"  max_dist_ratio: {max_dist_ratio:.4f}")
    print(f"  naip_dropout: {naip_dropout:.4f}")
    print(f"  uavsar_dropout: {uavsar_dropout:.4f}")
    print(f"  temporal_encoder: {temporal_encoder}")

    print(f"  fusion: {fusion}")
    print(f"  fusion_dropout: {fusion_dropout}")
    print(f"  fusion_num_heads: {fusion_num_heads}")
    print(f"  position_encoding_dim: {position_encoding_dim}")
    
    print(f"  k: {k}")
    print(f"  attr_dim: {attr_dim}")
    print(f"  Modalities: {'NAIP ' if use_naip else ''}{'UAVSAR ' if use_uavsar else ''}")
    print(f"  Batch size: {batch_size}, Num epochs: {num_epochs}")
    print("-"*80)
    
    # Create model config with trial-suggested parameters
    config = MultimodalModelConfig(
        feature_dim=feature_dim,
        up_attn_hds=up_attn_hds,
        up_dropout=up_dropout,
        fnl_attn_hds=fnl_attn_hds,
        img_embed_dim=img_embed_dim,
        # Include the additional parameters
        use_naip=use_naip,
        use_uavsar=use_uavsar,
        k=k,
        attr_dim=attr_dim,
        # Add SpatialFusion parameters
        temperature=temperature,
        max_dist_ratio=max_dist_ratio,
        # Add encoder dropout parameters
        naip_dropout=naip_dropout,
        uavsar_dropout=uavsar_dropout,
        temporal_encoder= temporal_encoder,
        # cross attention fusion parameters
        fusion_type = fusion,
        fusion_dropout = fusion_dropout,
        fusion_num_heads = fusion_num_heads,
        position_encoding_dim = position_encoding_dim
    )
    
    # Create a unique checkpoint directory for this trial
    trial_checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{trial.number}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    
    # Update log file to be trial-specific
    trial_log_file = log_file.replace(".txt", f"_trial_{trial.number}.txt")
    
    # Train the model with the trial-suggested hyperparameters
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    best_val_loss = train_multimodal_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=f"{model_name}_trial_{trial.number}",
        batch_size=batch_size,
        checkpoint_dir=trial_checkpoint_dir,
        model_config=config,
        num_epochs=num_epochs,
        log_file=trial_log_file,
        early_stopping_patience=early_stopping_patience,
        temp_dir_root=temp_dir_root,
        tensorboard_log_dir=tensorboard_log_dir,
        enable_debug=enable_debug,
        accumulation_steps=accumulation_steps,
        visualize_samples=visualize_samples,
        lr_scheduler_type=lr_scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print trial results
    print("-"*80)
    print(f"Trial #{trial.number} completed in {duration}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*80 + "\n")
    
    # Report intermediate values
    trial.report(best_val_loss, trial.number)
    
    # Handle pruning
    if trial.should_prune():
        print(f"Trial #{trial.number} was pruned based on intermediate results.")
        raise optuna.exceptions.TrialPruned()
    
    return best_val_loss


def find_unique_study_name(base_name, storage):
    """
    Find a unique study name by appending a number if necessary.
    
    Args:
        base_name: Base name for the study
        storage: Optuna storage URL
        
    Returns:
        unique_name: A unique study name
    """
    if storage is None:
        # For in-memory storage, we don't need to worry about duplicates
        return base_name
    
    name = base_name
    counter = 1
    
    while True:
        try:
            # Try to create a new study with this name and don't load if exists
            optuna.create_study(study_name=name, storage=storage, load_if_exists=False)
            # If we get here, the study was created and we need to delete it
            optuna.delete_study(study_name=name, storage=storage)
            return name  # This name is available
        except optuna.exceptions.DuplicatedStudyError:
            # Study already exists, try with a new name
            name = f"{base_name}{counter}"
            counter += 1
            print(f"Study name '{base_name}' already exists, trying '{name}'")


def run_optuna_study(train_dataset, val_dataset, model_name, batch_size, checkpoint_dir,
                    num_epochs=30, log_file="logs/optuna_study.log", early_stopping_patience=10,
                    temp_dir_root="data/output/tmp_shards", tensorboard_log_dir="data/output/tensorboard_logs",
                    enable_debug=False, accumulation_steps=3, visualize_samples=False,
                    lr_scheduler_type="onecycle", max_lr=5e-4, pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
                    use_naip=False, use_uavsar=False, k=15, attr_dim=3, n_trials=50, timeout=None,
                    study_name=None, storage=None):
    """
    Run an Optuna hyperparameter optimization study without TensorBoard callback.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name of the model
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train per trial
        log_file: Path to log file
        early_stopping_patience: Number of epochs to wait before early stopping
        temp_dir_root: Root directory for temporary files
        tensorboard_log_dir: Directory for TensorBoard logs
        enable_debug: Whether to enable debug mode
        accumulation_steps: Number of steps to accumulate gradients
        visualize_samples: Whether to visualize samples
        lr_scheduler_type: Type of learning rate scheduler
        max_lr: Maximum learning rate
        pct_start: Percentage of warmup in OneCycleLR
        div_factor: Division factor for OneCycleLR
        final_div_factor: Final division factor for OneCycleLR
        use_naip: Whether to use NAIP imagery
        use_uavsar: Whether to use UAVSAR imagery
        k: k-value for KNN (default: 15)
        attr_dim: Attribute dimension (default: 3)
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None for no timeout)
        study_name: Name of the study (None for auto-generated)
        storage: Storage for the study (None for in-memory)
        
    Returns:
        study: Completed Optuna study
    """
    # Set up logging directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Generate study name if not provided
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modality_str = f"_naip{int(use_naip)}_uavsar{int(use_uavsar)}"
        base_name = f"{model_name}{modality_str}_{timestamp}"
    else:
        base_name = study_name
    
    # Find a unique study name
    if storage is not None:
        study_name = find_unique_study_name(base_name, storage)
    else:
        study_name = base_name
    
    # Set up TensorBoard log directory if provided
    if tensorboard_log_dir:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # Print study information
    print("\n" + "="*80)
    print(f"Starting Optuna Study: {study_name}")
    print("="*80)
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout} seconds" if timeout else "Timeout: None")
    print(f"Storage: {storage if storage else 'In-memory'}")
    print(f"Modalities: {'NAIP ' if use_naip else ''}{'UAVSAR ' if use_uavsar else ''}")
    print(f"Fixed parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs per trial: {num_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  LR scheduler: {lr_scheduler_type} (max_lr={max_lr})")
    print(f"  k-value for KNN: {k}")
    print(f"  Attribute dimension: {attr_dim}")

    print("="*80 + "\n")
    
    # Create Optuna study with pruning
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",  # We want to minimize validation loss
        pruner=pruner,
        load_if_exists=False  # Important: we want to create a new study with our unique name
    )
    
    # Prepare the objective function with fixed parameters
    objective = lambda trial: optuna_objective(
        trial, train_dataset, val_dataset, model_name, batch_size, checkpoint_dir,
        num_epochs, log_file, early_stopping_patience, temp_dir_root, tensorboard_log_dir,
        enable_debug, accumulation_steps, visualize_samples,
        lr_scheduler_type, max_lr, pct_start, div_factor, final_div_factor,
        use_naip, use_uavsar, k, attr_dim
    )
    
    # Record start time
    study_start_time = datetime.now()
    print(f"Study started at {study_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the optimization
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        print("Study interrupted by user!")
    
    # Record end time
    study_end_time = datetime.now()
    study_duration = study_end_time - study_start_time
    
    print("\n" + "="*80)
    print(f"Study completed in {study_duration}")
    print("="*80)
    
    # Log the best parameters and trial
    best_trial = study.best_trial
    print(f"Best trial: #{best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.6f}")
    print("Best hyperparameters:")
    for param_name, param_value in best_trial.params.items():
        print(f"  {param_name}: {param_value}")
    
    # Save the best hyperparameters to a file
    best_params_dir = os.path.join(checkpoint_dir, "best_params")
    os.makedirs(best_params_dir, exist_ok=True)
    best_params_file = os.path.join(best_params_dir, f"{study_name}_best_params.json")
    
    with open(best_params_file, 'w') as f:
        json.dump({
            "study_name": study_name,
            "best_trial": best_trial.number,
            "best_val_loss": best_trial.value,
            "best_params": best_trial.params,
            "study_duration_seconds": study_duration.total_seconds(),
            "completed_trials": len(study.trials),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
    
    print(f"Best parameters saved to: {best_params_file}")
    
    # Try to generate plots if available (but don't fail if not)
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Save optimization history plot
        fig_history = plot_optimization_history(study)
        history_file = os.path.join(best_params_dir, f"{study_name}_optimization_history.png")
        fig_history.write_image(history_file)
        print(f"Optimization history saved to: {history_file}")
        
        # Save parameter importance plot
        fig_importance = plot_param_importances(study)
        importance_file = os.path.join(best_params_dir, f"{study_name}_param_importances.png")
        fig_importance.write_image(importance_file)
        print(f"Parameter importance saved to: {importance_file}")
        
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Could not generate visualization plots: {e}")
    
    return study


def create_model_from_best_params(study_path, modality_config=None):
    """
    Create a model configuration from the best parameters found in an Optuna study.
    
    Args:
        study_path: Path to the JSON file with best parameters
        modality_config: Optional dictionary with modality configurations to override
                         (use_naip, use_uavsar)
                         
    Returns:
        config: MultimodalModelConfig with best parameters
    """
    with open(study_path, 'r') as f:
        study_data = json.load(f)
    
    best_params = study_data["best_params"]
    
    # Print summary of loaded parameters
    print("\n" + "="*80)
    print(f"Loading best parameters from: {study_path}")
    print("="*80)
    print(f"Study name: {study_data.get('study_name', 'Unknown')}")
    print(f"Best trial: #{study_data.get('best_trial', 'Unknown')}")
    print(f"Best validation loss: {study_data.get('best_val_loss', 'Unknown'):.6f}")
    print("Parameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # Create base config from the best parameters
    config_dict = best_params.copy()
    
    # Override modality configuration if provided
    if modality_config is not None:
        print("\nOverriding modality configuration:")
        for key, value in modality_config.items():
            print(f"  {key}: {value}")
            config_dict[key] = value
    
    print("="*80 + "\n")
    
    # Create the config object
    config = MultimodalModelConfig(**config_dict)
    
    return config