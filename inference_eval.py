import torch
import matplotlib.pyplot as plt
from point_transformer_upsampler import (
    precompute_knn_inplace,
    TransformerPointUpsampler,
    normalize_pair,
    chamfer_distance,
    hausdorff_distance
)

def run_inference_and_visualize_1plot_w_rotation(
    trained_model, 
    train_df, 
    index=0,  
    device='cuda', 
    width=12, 
    height=5,  # Adjusted height for a tighter layout
    hide_labels=False,
    elev2=10,  
    azim2=120,
    overall_title=None  # New parameter for the overall title
):
    """
    Runs inference on a single sample (index),
    visualizes it from two different perspectives,
    and displays Chamfer Distance (CD) and Hausdorff Distance (HD).

    Args:
        trained_model: The model to run inference.
        train_df: DataFrame or list containing the samples.
        index (int): Index of the sample to visualize.
        device (str): The device to run computations on.
        width (int): Width of the figure.
        height (int): Height of the figure.
        hide_labels (bool): Option to hide axis labels.
        elev2 (int): Elevation angle for the rotated view.
        azim2 (int): Azimuth angle for the rotated view.
        overall_title (str or None): Overall title for the figure.
    """

    def process_sample(index):
        """Extracts, normalizes, runs inference, and computes distance metrics."""
        sample = train_df[index]
        dep_points_raw = sample['dep_points']
        uav_points_raw = sample['uav_points']  
        edge_index = sample['dep_edge_index']

        # Normalize
        dep_points_norm, uav_points_norm, center, scale = normalize_pair(dep_points_raw, uav_points_raw)

        # Move to device
        dep_points_norm = dep_points_norm.to(device)
        uav_points_norm = uav_points_norm.to(device)
        edge_index      = edge_index.to(device)

        # Inference in normalized space
        trained_model.eval()
        with torch.no_grad():
            pred_points_norm = trained_model(dep_points_norm, edge_index)  

        # Compute Chamfer & Hausdorff Distances
        orig_chamfer_dist = chamfer_distance(dep_points_norm, uav_points_norm).item()
        upsmpl_chamfer_dist = chamfer_distance(pred_points_norm, uav_points_norm).item()

        orig_hausdorff_dist = hausdorff_distance(dep_points_norm, uav_points_norm)
        upsmpl_hausdorff_dist = hausdorff_distance(pred_points_norm, uav_points_norm)
        
        return (
            dep_points_norm.cpu(),
            uav_points_norm.cpu(),
            pred_points_norm.cpu(),
            orig_chamfer_dist,
            upsmpl_chamfer_dist,
            orig_hausdorff_dist,
            upsmpl_hausdorff_dist
        )

    # Process sample
    dep, uav, pred, chamfer_orig, chamfer_upsmpl, hausdorff_orig, hausdorff_upsmpl = process_sample(index)
    
    chamfer_diff = chamfer_upsmpl - chamfer_orig
    hausdorff_diff = hausdorff_upsmpl - hausdorff_orig
    
    # Create a 2-row, 4-column figure layout
    fig, axes = plt.subplots(2, 4, figsize=(width, height), subplot_kw={'projection': '3d'})

    # Set overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=14)

    def configure_axes(ax, title, num_points, elev=20, azim=30):
        """Helper function to configure axis formatting, view angles, and add point count."""
        ax.view_init(elev=elev, azim=azim)  
        ax.set_xticks([])  # Remove ticks for a cleaner look
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"{title}\n({num_points} pts)", fontsize=10, pad=0)  # Added point count

    def add_distance_labels(ax, x, y, chamfer_dist, hausdorff_dist, ttl):
        """Add Chamfer Distance (CD) and Hausdorff Distance (HD) labels to the plot."""
        ax.text2D(x, y + 0.15, f"{ttl}", transform=ax.transAxes, fontsize=10, color='black')
        ax.text2D(x, y + 0.05, f"CD: {chamfer_dist:.4f}", transform=ax.transAxes, fontsize=8, color='black')
        ax.text2D(x, y - 0.05, f"HD: {hausdorff_dist:.4f}", transform=ax.transAxes, fontsize=8, color='black')
        
    def add_difference_label(ax, x, y, chamfer_diff, hausdorff_diff):
        """Add a label comparing the difference in distances with color-coding."""
    
        # Determine which is closer
        chamfer_closer = " (3DEP Closer)" if chamfer_diff > 0 else " (Upsampled Closer)"
        hausdorff_closer = " (3DEP Closer)" if hausdorff_diff > 0 else " (Upsampled Closer)"
    
        # Set color: Blue if Upsampled is closer, Red if 3DEP is closer
        chamfer_color = "red" if chamfer_diff > 0 else "blue"
        hausdorff_color = "red" if hausdorff_diff > 0 else "blue"
    
        # Format and display the text with colors
        ax.text2D(
            x, y + 0.05, 
            f"Δ CD: {chamfer_diff:+.4f}{chamfer_closer}", 
            transform=ax.transAxes, fontsize=9, color=chamfer_color
        )
        ax.text2D(
            x, y - 0.05, 
            f"Δ HD: {hausdorff_diff:+.4f}{hausdorff_closer}", 
            transform=ax.transAxes, fontsize=9, color=hausdorff_color
        )
 
    # First row: Default viewpoint
    axes[0, 0].scatter(dep[:, 0], dep[:, 1], dep[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[0, 0], "3DEP", len(dep))
    
    axes[0, 1].scatter(uav[:, 0], uav[:, 1], uav[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[0, 1], "UAV", len(uav))
    
    axes[0, 2].scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[0, 2], "Upsampled", len(pred))

    # Distance Metrics Box (Text Only)
    axes[0, 3].axis("off")
    axes[0, 3].text2D(0.5, 1, f"Index: {index}", transform=axes[0, 3].transAxes, fontsize=8, color='black')
    add_distance_labels(axes[0, 3], 0.05, 0.75, chamfer_upsmpl, hausdorff_upsmpl, "Upsampled vs UAV Dist")
    add_distance_labels(axes[0, 3], 0.05, 0.25, chamfer_orig, hausdorff_orig, "3DEP vs UAV Dist")
    add_difference_label(axes[0, 3], 0.05, 0.0, chamfer_diff, hausdorff_diff)  # Added difference label

    
    # Second row: Rotated viewpoint
    axes[1, 0].scatter(dep[:, 0], dep[:, 1], dep[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[1, 0], "3DEP Rotated", len(dep), elev=elev2, azim=azim2)
    
    axes[1, 1].scatter(uav[:, 0], uav[:, 1], uav[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[1, 1], "UAV Rotated", len(uav), elev=elev2, azim=azim2)
    
    axes[1, 2].scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[1, 2], "Upsampled Rotated", len(pred), elev=elev2, azim=azim2)

    # Distance Metrics Box (Text Only, Duplicate for Second Row)
    axes[1, 3].axis("off")
    # (Optional) Add more text or leave blank

    # **Reduce whitespace between plots**
    plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0, wspace=-0.6, hspace=0.1)

    plt.show()

