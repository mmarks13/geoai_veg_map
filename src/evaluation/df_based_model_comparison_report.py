import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import random
from tqdm import tqdm
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import sys

# Configure matplotlib for better PDF output with rasterization
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better text rendering
mpl.rcParams['figure.dpi'] = 200   # Default figure DPI
mpl.rcParams['savefig.dpi'] = 150  # Default save DPI
mpl.rcParams['figure.autolayout'] = True  # Better layout
mpl.rcParams['path.simplify'] = True  # Simplify paths for better rendering
mpl.rcParams['path.simplify_threshold'] = 0.8  # Higher threshold for more simplification

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def plot_naip_imagery_row(naip_data, naip_dates=None, img_bbox=None, bbox_overlay=None, figsize=(15, 3), title=None):
    """
    Create a single row of NAIP imagery plots, ordered by date.
    
    Parameters:
    -----------
    naip_data : numpy.ndarray
        NAIP imagery with shape [n_images, 4, h, w] - 4 spectral bands (scaled 0-1)
    naip_dates : list[str], optional
        List of NAIP acquisition date strings
    img_bbox : numpy.ndarray, optional
        NAIP imagery bounding box [minx, miny, maxx, maxy]
    bbox_overlay : tuple, optional
        Bounding box to overlay on each image [minx, miny, maxx, maxy]
    figsize : tuple, optional
        Base figure size (width, height)
    title : str, optional
        Overall title for the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure containing all NAIP images
    """
    # Get the number of images
    n_images = naip_data.shape[0]
    
    # Check if we have dates for all images
    if naip_dates is None or len(naip_dates) < n_images:
        # Fill in missing dates
        if naip_dates is None:
            naip_dates = []
        naip_dates = naip_dates + [f"Unknown Date {i+1}" for i in range(len(naip_dates), n_images)]
    
    # Sort the images by date if possible
    try:
        # Convert string dates to datetime objects for sorting
        datetime_objects = [datetime.strptime(date, "%Y-%m-%d") for date in naip_dates]
        # Create pairs of (index, datetime)
        pairs = list(zip(range(n_images), datetime_objects))
        # Sort pairs by datetime
        pairs.sort(key=lambda x: x[1])
        # Get sorted indices
        sorted_indices = [pair[0] for pair in pairs]
    except (ValueError, TypeError):
        # If date parsing fails, just use original order
        sorted_indices = list(range(n_images))
    
    # Create the figure with all plots in a single row
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Ensure axes is always an array
    if n_images == 1:
        axes = np.array([axes])
    
    # Plot each image in order
    for i, idx in enumerate(sorted_indices):
        # Get the image and convert to numpy if needed
        img = naip_data[idx]
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy() if img.is_cuda else img.numpy()
        else:
            img_np = img
        
        # Create RGB image (use first 3 bands)
        if img_np.shape[0] >= 3:
            rgb = np.transpose(img_np[:3], (1, 2, 0))
            
            # Apply percentile-based contrast stretching for better visualization
            if rgb.max() > 0:
                p_low, p_high = np.percentile(rgb[rgb > 0], [2, 98])
                rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
                rgb = rgb_stretched * 255
            else:
                rgb = np.zeros_like(rgb)
            
            rgb = rgb.astype(np.uint8)
        else:
            # Single band grayscale image
            rgb = img_np[0]
            
            # Apply contrast stretching
            if rgb.max() > rgb.min():
                p_low, p_high = np.percentile(rgb, [2, 98])
                rgb = np.clip((rgb - p_low) / (p_high - p_low), 0, 1) * 255
                rgb = rgb.astype(np.uint8)
        
        # Calculate extent from img_bbox
        if img_bbox is not None:
            extent = (img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3])
            
            # Calculate the centroid of the img_bbox
            centroid_x = (img_bbox[0] + img_bbox[2]) / 2
            centroid_y = (img_bbox[1] + img_bbox[3]) / 2
            
            # Calculate the 10x10 meter bbox that shares the centroid with img_bbox
            half_width = 5.0  # half of 10 meters
            half_height = 5.0  # half of 10 meters
            
            inner_bbox = [
                centroid_x - half_width,   # minx
                centroid_y - half_height,  # miny
                centroid_x + half_width,   # maxx
                centroid_y + half_height   # maxy
            ]
        else:
            extent = (0, img_np.shape[2], 0, img_np.shape[1])
            inner_bbox = None
        
        # Plot the image
        axes[i].imshow(rgb, extent=extent, origin='upper')
        
        # Draw the 10x10 meter bounding box centered on img_bbox
        if inner_bbox is not None:
            rect = Rectangle((inner_bbox[0], inner_bbox[1]),
                            inner_bbox[2] - inner_bbox[0],
                            inner_bbox[3] - inner_bbox[1],
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
        
        # Add date as title
        date_str = naip_dates[idx] if idx < len(naip_dates) else f"Image {idx+1}"
        axes[i].set_title(f"{date_str}", fontsize=10)
        
        # Set ticks
        axes[i].tick_params(axis='both', which='both', labelsize=8)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.85)  # Make room for the overall title
    
    return fig

def generate_overview_pages(pdf, df, model_names, dpi=150):
    """
    Generate overview pages with performance metrics and plots.
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF document
    df : pandas.DataFrame
        DataFrame with model results
    model_names : list
        List of model names present in the DataFrame
    dpi : int, optional
        DPI for output
    """
    # First page: Overview and general statistics
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Multi-Model Evaluation Report", fontsize=16)
    
    # Add report generation time
    fig.text(0.5, 0.94, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             ha='center', fontsize=12)
    
    # Summary of models
    models_summary = "\n".join([f"- {model.capitalize()}" for model in model_names])
    
    fig.text(0.5, 0.85, "Models compared in this report:", ha='center', fontsize=12)
    fig.text(0.5, 0.8, models_summary, ha='center', fontsize=10)
    
    # Add dataset statistics
    stats_text = (
        f"Number of samples: {len(df)}\n"
        f"Input points (avg): {df['input_point_count'].mean():.1f}\n"
        f"Ground truth points (avg): {df['ground_truth_point_count'].mean():.1f}\n"
        f"Prediction points (avg): {df['pred_point_count'].mean():.1f}\n"
        f"Average canopy height change: {df['net_canopy_height_change'].mean():.3f} meters\n"
    )
    
    fig.text(0.5, 0.6, "Dataset Statistics:", ha='center', fontsize=12)
    fig.text(0.5, 0.5, stats_text, ha='center', fontsize=10)
    
    # Add the figure to the PDF with specified DPI
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    
    # Second page: Chamfer Distance Distribution
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Model Performance: Chamfer Distance", fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Extract Chamfer distances for all models
    chamfer_data = {}
    for i, model in enumerate(model_names):
        chamfer_col = f"{model}_chamfer_distance"
        if chamfer_col in df.columns:
            chamfer_data[model] = df[chamfer_col]
    
    # Create boxplot comparison
    sns.boxplot(data=pd.DataFrame(chamfer_data), ax=axes[0])
    axes[0].set_title('Chamfer Distance Comparison')
    axes[0].set_ylabel('Chamfer Distance')
    axes[0].set_xlabel('Model')
    axes[0].grid(True, alpha=0.3)
    
    # Create improvement over input plot
    improvement_data = {}
    for model in model_names:
        chamfer_col = f"{model}_chamfer_distance"
        if chamfer_col in df.columns:
            improvement = df['input_chamfer_distance'] / df[chamfer_col]
            improvement_data[model] = improvement
    
    sns.boxplot(data=pd.DataFrame(improvement_data), ax=axes[1])
    axes[1].set_title('Improvement Ratio vs Input')
    axes[1].set_ylabel('Input CD / Model CD')
    axes[1].set_xlabel('Model')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
    axes[1].legend()
    
    # Create a scatter plot of combined model improvement versus canopy height change
    best_model = 'combined' if 'combined_chamfer_distance' in df.columns else model_names[0]
    model_col = f"{best_model}_chamfer_distance"
    improvement = df['input_chamfer_distance'] / df[model_col]
    
    axes[2].scatter(df['net_canopy_height_change'], improvement, alpha=0.5)
    axes[2].set_title(f'Improvement vs Canopy Height Change ({best_model.capitalize()})')
    axes[2].set_xlabel('Net Canopy Height Change (m)')
    axes[2].set_ylabel('Improvement Ratio')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
    axes[2].legend()
    
    # Create sorted improvement plot for the best model
    sorted_improvement = improvement.sort_values()
    axes[3].plot(sorted_improvement.values, marker='.', markersize=2, linestyle='-', linewidth=1)
    axes[3].set_title(f'Sorted Improvement Ratio ({best_model.capitalize()})')
    axes[3].set_xlabel('Sample Rank')
    axes[3].set_ylabel('Improvement Ratio')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
    
    # Add threshold lines for high and low improvement
    if len(sorted_improvement) >= 30:
        low_threshold = sorted_improvement.iloc[29]
        high_threshold = sorted_improvement.iloc[-30]
        axes[3].axhline(y=low_threshold, color='orange', linestyle='--', alpha=0.7, 
                        label=f'Low Improvement Threshold: {low_threshold:.2f}')
        axes[3].axhline(y=high_threshold, color='green', linestyle='--', alpha=0.7,
                        label=f'High Improvement Threshold: {high_threshold:.2f}')
        axes[3].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    
    # Third page: InfoCD metrics if available
    if all(f"{model}_infocd" in df.columns for model in model_names):
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.suptitle("Model Performance: InfoCD Metrics", fontsize=16)
        
        # Extract InfoCD values for all models
        infocd_data = {}
        for model in model_names:
            infocd_col = f"{model}_infocd"
            if infocd_col in df.columns:
                infocd_data[model] = df[infocd_col]
        
        # Create boxplot comparison
        sns.boxplot(data=pd.DataFrame(infocd_data), ax=axes[0])
        axes[0].set_title('InfoCD Comparison')
        axes[0].set_ylabel('InfoCD')
        axes[0].set_xlabel('Model')
        axes[0].grid(True, alpha=0.3)
        
        # Create improvement over input plot
        infocd_improvement_data = {}
        for model in model_names:
            infocd_col = f"{model}_infocd"
            if infocd_col in df.columns:
                improvement = df['input_infocd'] / df[infocd_col]
                infocd_improvement_data[model] = improvement
        
        sns.boxplot(data=pd.DataFrame(infocd_improvement_data), ax=axes[1])
        axes[1].set_title('InfoCD Improvement Ratio vs Input')
        axes[1].set_ylabel('Input InfoCD / Model InfoCD')
        axes[1].set_xlabel('Model')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        axes[1].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

def generate_sample_page(
    pdf, 
    sample_row, 
    model_names, 
    sample_index, 
    sample_type, 
    dpi=150, 
    point_size=1, 
    point_alpha=0.5,
    wspace=0.3,  # Width space between subplots
    hspace=0.3,  # Height space between subplots
    axis_fontsize=8,  # Font size for axis labels
    color_palette=None  # Color palette to use
):
    """
    Generate a PDF page showing multiple model predictions for a single sample.
    
    Parameters:
    -----------
    pdf : PdfPages
        PDF document
    sample_row : pandas.Series
        Sample data row
    model_names : list
        List of model names present in the DataFrame
    sample_index : int
        Sample index within the current category
    sample_type : str
        Type of sample ('high-loss', 'low-improvement', 'high-improvement', 'random')
    dpi : int, optional
        DPI for output
    point_size : float, optional
        Size of points in scatter plots
    point_alpha : float, optional
        Alpha (transparency) of points
    wspace : float, optional
        Width space between subplots
    hspace : float, optional
        Height space between subplots
    axis_fontsize : int, optional
        Font size for axis labels
    color_palette : list, optional
        Color palette to use for point clouds
    """
    # Extract sample info
    tile_id = sample_row['tile_id']
    
    # Extract point clouds
    input_points = sample_row['input_points']
    gt_points = sample_row['ground_truth_points']
    
    # Create a figure for point cloud visualizations
    fig = plt.figure(figsize=(16, 9))
    
    # Add title based on sample type
    if sample_type == "high-loss":
        title = f"High-Loss Sample {sample_index+1}: Tile {tile_id}"
    elif sample_type == "low-improvement":
        title = f"Low-Improvement Sample {sample_index+1}: Tile {tile_id}"
    elif sample_type == "high-improvement":
        title = f"High-Improvement Sample {sample_index+1}: Tile {tile_id}"
    else:
        title = f"Random Sample {sample_index+1}: Tile {tile_id}"
    
    fig.suptitle(title, fontsize=14)
    
    # Define color mapping based on provided palette or default
    if color_palette is None:
        color_palette = ['#8B4513', '#A0522D', '#2E8B57', '#4682B4', '#800080', '#556B2F']
    
    # Map colors to specific point clouds
    colors = {
        'input': color_palette[0],
        'baseline': color_palette[1],
        'naip': color_palette[2],
        'uavsar': color_palette[3],
        'combined': color_palette[4],
        'combined_4x': color_palette[2],
        'combined_6x':color_palette[3],
        'combined_8x':color_palette[3],
        'ground_truth': color_palette[5]
    }
    
    # Define plot configurations
    plot_configs = [
        {'title': '3DEP Input (2014-2016)', 'points': input_points, 'color': colors['input'],
         'num_points': len(input_points)}
    ]
    
    # Add model plots in order: baseline, naip, uavsar, combined
    for model in model_names: #['baseline', 'naip', 'uavsar', 'combined']:
        if model in model_names:
            model_points_col = f"{model}_pred_points"
            model_cd_col = f"{model}_chamfer_distance"
            
            if model_points_col in sample_row and sample_row[model_points_col] is not None:
                plot_configs.append({
                    'title': f"{model.upper()} Model Prediction",
                    'points': sample_row[model_points_col],
                    'color': colors[model],
                    'loss': sample_row.get(model_cd_col, None),
                    'num_points': len(sample_row[model_points_col])
                })
    
    # Add ground truth as the last plot
    plot_configs.append({
        'title': 'UAV Ground Truth (2023-2024)', 
        'points': gt_points, 
        'color': colors['ground_truth'],
        'num_points': len(gt_points)
    })
    
    # Determine global axis limits
    all_points = [config['points'] for config in plot_configs]
    min_x = min([np.min(points[:, 0]) for points in all_points])
    max_x = max([np.max(points[:, 0]) for points in all_points])
    min_y = min([np.min(points[:, 1]) for points in all_points])
    max_y = max([np.max(points[:, 1]) for points in all_points])
    min_z = min([np.min(points[:, 2]) for points in all_points])
    max_z = max([np.max(points[:, 2]) for points in all_points])
    
    # Add padding to the limits
    padding = 0.05
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    min_x -= x_range * padding
    max_x += x_range * padding
    min_y -= y_range * padding
    max_y += y_range * padding
    min_z -= z_range * padding
    max_z += z_range * padding
    
    # Calculate z-range for dynamic vertical scaling
    z_range = max_z - min_z
    
    # Define dynamic vertical scaling factor based on z-range
    # Higher z-range means more vertical stretching
    if z_range <= 5:
        vertical_scale = 0.5  # Default for small z-range
    elif z_range <= 10:
        vertical_scale = 0.8
    elif z_range <= 20:
        vertical_scale = 1.2
    elif z_range <= 30:
        vertical_scale = 1.5
    else:
        vertical_scale = 2.0  # Max stretching for very tall terrain
    
    # Create point cloud plots in a single row
    num_plots = len(plot_configs)
    axes = []
    for i, config in enumerate(plot_configs):
        ax = fig.add_subplot(1, num_plots, i+1, projection='3d')
        axes.append(ax)
        
        points = config['points']
        color = config['color']
        
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], 
            s=point_size, 
            alpha=point_alpha, 
            c=color, 
            edgecolors='none',  # This removes the edge outline completely
            rasterized=True
        )
        
        # Create title with number of points and distance if available
        title_parts = [config['title']]
        title_parts.append(f"N={config.get('num_points', 0)}")
        
        if 'loss' in config and config['loss'] is not None:
            title_parts.append(f"CD={config['loss']:.4f}")
        
        ax.set_title('\n'.join(title_parts), fontsize=10)
        
        # Set consistent axis limits for all plots
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        
        # Set axis labels with configurable font size
        ax.set_xlabel('X', fontsize=axis_fontsize)
        ax.set_ylabel('Y', fontsize=axis_fontsize)
        ax.set_zlabel('Z', fontsize=axis_fontsize)
        
        # Apply dynamic vertical scaling based on the z-range
        ax.set_box_aspect([1, 1, vertical_scale])
        
        # Set tick label font size
        ax.tick_params(axis='both', which='both', labelsize=axis_fontsize)
    
    # Adjust subplot spacing with configurable parameters
    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=0.08, top=0.92, left=0.05, right=0.95)
    
    # Add the figure to the PDF with specified DPI for rasterization
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    
    # Add NAIP imagery if available
    if 'naip_images' in sample_row and sample_row['naip_images'] is not None:
        try:
            naip_data = sample_row['naip_images']
            naip_dates = sample_row.get('naip_dates', None)
            img_bbox = sample_row.get('img_bbox', None)
            
            # Create NAIP visualization
            naip_title = f"{title} - NAIP Imagery"
            
            # Adjust figure size based on number of NAIP images
            n_images = naip_data.shape[0]
            naip_figsize = (min(16, n_images * 4), 4)
            
            naip_fig = plot_naip_imagery_row(
                naip_data,
                naip_dates=naip_dates,
                img_bbox=img_bbox,
                figsize=naip_figsize,
                title=naip_title
            )
            
            # Add the figure to the PDF with specified DPI
            pdf.savefig(naip_fig, dpi=dpi)
            plt.close(naip_fig)
            
        except Exception as e:
            print(f"Error plotting NAIP imagery for sample {tile_id}: {e}")
            # Create a simple error page
            fig = plt.figure(figsize=(11, 4))
            fig.suptitle(f"{title} - NAIP Imagery (Error)", fontsize=14)
            fig.text(0.5, 0.5, f"Error plotting NAIP imagery: {e}", 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

def generate_report(
    df,
    output_dir="reports",
    model_names=None,
    specific_tile_ids=None,
    n_high_loss_samples=30,
    n_low_improvement_samples=30,
    n_high_improvement_samples=30,
    n_random_samples=30,
    dpi=150,
    point_size=1.0,
    point_alpha=0.5,
    wspace=0.3,
    hspace=0.3,
    axis_fontsize=8,
    color_palette=['#8B4513', '#A0522D', '#2E8B57', '#4682B4', '#800080', '#556B2F'],
    main_model='combined'
):
    """
    Generate a comprehensive PDF report for multiple model evaluation using pre-calculated DataFrame.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect model names if not provided
    if model_names is None:
        model_names = []
        for prefix in ['baseline', 'naip', 'uavsar', 'combined']:
            if f"{prefix}_pred_points" in df.columns:
                model_names.append(prefix)
        
        if not model_names:
            print("Error: Could not detect any models in the DataFrame")
            return
        
        print(f"Auto-detected models: {', '.join(model_names)}")
    
    # Ensure main_model is in model_names
    if main_model not in model_names:
        main_model = model_names[0]
        print(f"Warning: Specified main_model not found in DataFrame. Using {main_model} instead.")
    
    # Generate the PDF report
    report_path = os.path.join(output_dir, f"multi_model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    print(f"Generating report to {report_path}...")
    with PdfPages(report_path) as pdf:
        # Generate overview pages
        generate_overview_pages(pdf, df, model_names, dpi)
        
        # If specific tile_ids are provided, generate pages for these tiles first
        if specific_tile_ids is not None and len(specific_tile_ids) > 0:
            # Filter DataFrame to include only the specified tiles
            specific_tiles_df = df[df['tile_id'].isin(specific_tile_ids)]
            
            if not specific_tiles_df.empty:
                print(f"Adding {len(specific_tiles_df)} specified tiles...")

                
                # Generate a page for each specified tile
                for i, (_, sample) in enumerate(tqdm(specific_tiles_df.iterrows())):
                    generate_sample_page(
                        pdf, 
                        sample, 
                        model_names,
                        i, 
                        "top-improvement", 
                        dpi=dpi,
                        point_size=point_size,
                        point_alpha=point_alpha,
                        wspace=wspace,
                        hspace=hspace,
                        axis_fontsize=axis_fontsize,
                        color_palette=color_palette
                    )
            
            # # If we're only processing specific tiles, we can return here
            # print(f"Report saved to {report_path}")
            # return
        
        # If no specific tiles or we want additional samples, proceed with normal categories
        # 1. Sort by loss (high to low)
        main_loss_col = f"{main_model}_chamfer_distance"
        df_sorted_by_loss = df.sort_values(by=main_loss_col, ascending=False)
        
        # 2. Calculate improvement ratios (input loss / model loss)
        for model in model_names:
            model_loss_col = f"{model}_chamfer_distance"
            improvement_col = f"{model}_improvement_ratio"
            df[improvement_col] = df['input_chamfer_distance'] / df[model_loss_col]
        
        # 3. Sort by improvement ratio
        main_improvement_col = f"{main_model}_improvement_ratio"
        df_sorted_by_improvement = df.sort_values(by=main_improvement_col)
        
        # Select sample sets
        high_loss_samples = df_sorted_by_loss.head(n_high_loss_samples)
        low_improvement_samples = df_sorted_by_improvement.head(n_low_improvement_samples)
        high_improvement_samples = df.sort_values(by=main_improvement_col, ascending=False).head(n_high_improvement_samples)
        
        # Get remaining samples (not in any of the above sets)
        used_indices = set(high_loss_samples.index).union(
            set(low_improvement_samples.index),
            set(high_improvement_samples.index)
        )
        remaining_samples = df[~df.index.isin(used_indices)]
        
        # Select random samples from remaining
        random_samples = remaining_samples.sample(
            min(n_random_samples, len(remaining_samples))
        ) if len(remaining_samples) > 0 else pd.DataFrame()
        
        # Generate high loss sample pages
        if n_high_loss_samples > 0 and not high_loss_samples.empty:
            print(f"Adding {len(high_loss_samples)} highest loss samples...")
            for i, (_, sample) in enumerate(tqdm(high_loss_samples.iterrows())):
                generate_sample_page(
                    pdf, 
                    sample, 
                    model_names,
                    i, 
                    "high-loss", 
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    wspace=wspace,
                    hspace=hspace,
                    axis_fontsize=axis_fontsize,
                    color_palette=color_palette
                )
        
        # Generate low improvement sample pages
        if n_low_improvement_samples > 0 and not low_improvement_samples.empty:
            print(f"Adding {len(low_improvement_samples)} lowest improvement samples...")
            for i, (_, sample) in enumerate(tqdm(low_improvement_samples.iterrows())):
                generate_sample_page(
                    pdf, 
                    sample, 
                    model_names,
                    i, 
                    "low-improvement", 
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    wspace=wspace,
                    hspace=hspace,
                    axis_fontsize=axis_fontsize,
                    color_palette=color_palette
                )
        
        # Generate high improvement sample pages
        if n_high_improvement_samples > 0 and not high_improvement_samples.empty:
            print(f"Adding {len(high_improvement_samples)} highest improvement samples...")
            for i, (_, sample) in enumerate(tqdm(high_improvement_samples.iterrows())):
                generate_sample_page(
                    pdf, 
                    sample, 
                    model_names,
                    i, 
                    "high-improvement", 
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    wspace=wspace,
                    hspace=hspace,
                    axis_fontsize=axis_fontsize,
                    color_palette=color_palette
                )
        
        # Generate random sample pages
        if n_random_samples > 0 and not random_samples.empty:
            print(f"Adding {len(random_samples)} random samples...")
            for i, (_, sample) in enumerate(tqdm(random_samples.iterrows())):
                generate_sample_page(
                    pdf, 
                    sample, 
                    model_names,
                    i, 
                    "random", 
                    dpi=dpi,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    wspace=wspace,
                    hspace=hspace,
                    axis_fontsize=axis_fontsize,
                    color_palette=color_palette
                )
    
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    # Load the DataFrame
    df_path = "data/processed/model_data/point_cloud_comparison_df_0516_e100_w8x.pkl"
    print(f"Loading DataFrame from {df_path}...")
    df = pd.read_pickle(df_path)
    print(f"Loaded DataFrame with {len(df)} samples")
    print(df.columns)
    print(df.head(5))
    # Check for required columns
    required_columns = ['baseline_chamfer_distance', 'combined_chamfer_distance', 'input_chamfer_distance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Configure filter parameters
    z_std_percentile = 50  # Percentile threshold for Z standard deviation
    
    # Calculate ground truth point cloud statistics
    print("Calculating point cloud statistics...")
    df['gt_max_z'] = df['ground_truth_points'].apply(lambda points: np.max(points[:, 2]) if points is not None else 0)
    df['gt_z_std'] = df['ground_truth_points'].apply(lambda points: np.std(points[:, 2]) if points is not None else 0)
    
    # Calculate Z standard deviation threshold at specified percentile
    z_std_threshold = df['gt_z_std'].quantile(z_std_percentile/100)
    print(f"Z standard deviation threshold (at {z_std_percentile}th percentile): {z_std_threshold:.4f}")
    
    # Create an empty list to store all selected tile IDs
    selected_tile_ids = []
    manual_tile_list = [
        'tile_11401',
        'tile_15957',
        'tile_19336',
        'tile_30473',
        'tile_35298',
        'tile_34892',
        'tile_288',
        'tile_16640',
        'tile_23569',
        'tile_15958',
        'tile_35161',
        'tile_22288',
        'tile_24455',
        'tile_24133',
        'tile_23710',
        'tile_13481',
        'tile_14268',
        'tile_34112',
        'tile_20853',
        'tile_32188',
        'tile_33405',
        'tile_19335',
        'tile_30356',
        'tile_34647',
        "tile_15957", "tile_15958", "tile_31561", "tile_30473"
    ]

    

    # ---- Filter 1: Baseline to Combined improvement ----
    # Calculate improvement ratio between baseline and combined models
    df['baseline_to_combined_ratio'] = df['baseline_chamfer_distance'] / df['combined_8x_chamfer_distance'].replace(0, float('inf'))
    
    # Get top 100 tiles with highest improvement ratio
    filtered_tiles = df.sort_values(by='baseline_to_combined_ratio', ascending=False).head(100)['tile_id'].tolist()
    print(f"Selected {len(filtered_tiles)} tiles with highest baseline-to-combined improvement")
    selected_tile_ids.extend(filtered_tiles)
    
    # ---- Filter 2: Lowest Combined CD with high terrain ----
    # Get top 50 tiles with lowest combined CD where max z > 15 and z std > threshold
    high_terrain_filter = (df['gt_max_z'] > 10) & (df['gt_z_std'] > z_std_threshold)
    filtered_tiles = df[high_terrain_filter].sort_values(by='combined_8x_chamfer_distance').head(100)['tile_id'].tolist()
    print(f"Selected {len(filtered_tiles)} tiles with lowest combined CD, terrain height > 15m, and Z std > {z_std_threshold:.4f}")
    selected_tile_ids.extend(filtered_tiles)
    
    # ---- Filter 3: Biggest input-to-combined improvement with moderate terrain ----
    # Calculate improvement ratio between input and combined
    df['input_to_combined_ratio'] = df['input_chamfer_distance'] / df['combined_8x_chamfer_distance'].replace(0, float('inf'))
    
    # Apply filter criteria
    combined_filter = (df['gt_max_z'] > 10) & (df['input_chamfer_distance'] < 2) & (df['gt_z_std'] > z_std_threshold)
    filtered_tiles = df[combined_filter].sort_values(by='input_to_combined_ratio', ascending=False).head(100)['tile_id'].tolist()
    print(f"Selected {len(filtered_tiles)} tiles with best input-to-combined improvement, terrain > 10m, input CD < 2, and Z std > {z_std_threshold:.4f}")
    selected_tile_ids.extend(filtered_tiles)
    selected_tile_ids.extend(manual_tile_list)
    # Remove duplicates while preserving order
    unique_tile_ids = []
    for tile_id in selected_tile_ids:
        if tile_id not in unique_tile_ids:
            unique_tile_ids.append(tile_id)
    
    print(f"Selected {len(selected_tile_ids)} total tiles, {len(unique_tile_ids)} unique tiles after deduplication")
    
    # Generate report with the selected tile IDs
    generate_report(
        df=df,
        output_dir="data/output/reports",
        model_names=['baseline', 'combined','combined_8x'],
        specific_tile_ids=unique_tile_ids,
        n_high_loss_samples=0,
        n_low_improvement_samples=0,
        n_high_improvement_samples=0,
        n_random_samples=0,
        dpi=350,
        point_size=0.6,
        point_alpha=0.6,
        wspace=0.3,
        hspace=0.3,
        axis_fontsize=8,
        color_palette=['#996B33', '#809933', '#339952', '#337D99', '#663399', '#99337D'],
        main_model='combined_8x'
    )