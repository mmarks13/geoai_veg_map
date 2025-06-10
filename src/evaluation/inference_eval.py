import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer distance between two point clouds.
    pc1: [N1, 3]
    pc2: [N2, 3]
    """
    dist = torch.cdist(pc1, pc2)  # [N1, N2]
    min_dist_pc1, _ = dist.min(dim=1)  # [N1]
    min_dist_pc2, _ = dist.min(dim=0)  # [N2]
    return min_dist_pc1.mean() + min_dist_pc2.mean()
    
def hausdorff_distance(pc1, pc2):
    """Computes the bidirectional Hausdorff distance between two point clouds."""
    dist = torch.cdist(pc1, pc2)  # [N1, N2]
    min_dist_pc1, _ = dist.min(dim=1)  # [N1]
    min_dist_pc2, _ = dist.min(dim=0)  # [N2]
    return max(min_dist_pc1.max().item(), min_dist_pc2.max().item())
    
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
        """Extracts data, runs inference, and computes distance metrics."""
        sample = train_df[index]
        
        # Use precomputed normalized points
        dep_points_norm = sample['dep_points_norm']
        uav_points_norm = sample['uav_points_norm']
        
        # Get the edge index from the appropriate source
        if 'dep_edge_index' in sample:
            edge_index = sample['dep_edge_index']
        elif 'knn_edge_indices' in sample and 30 in sample['knn_edge_indices']:  # Default to k=30
            edge_index = sample['knn_edge_indices'][30]
        else:
            # Compute KNN if not available
            print(f"Warning: No precomputed edges found for sample {index}, computing KNN")
            edge_index = knn_graph(dep_points_norm, k=30, loop=False)
            edge_index = to_undirected(edge_index, num_nodes=dep_points_norm.size(0))

        # Move to device
        dep_points_norm = dep_points_norm.to(device)
        uav_points_norm = uav_points_norm.to(device)
        edge_index = edge_index.to(device)

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

def run_inference_and_visualize_2plots(
    trained_model, 
    model_data, 
    index1=0, 
    index2=1, 
    device='cuda', 
    width=14, 
    height=3,
    hide_labels=False
):
    """
    Runs inference on two different samples (index1 & index2),
    visualizes both in a single row of six plots with an empty separator,
    and displays Chamfer Distance (CD) and Hausdorff Distance (HD).
    """

    def process_sample(index):
        """Extracts data, runs inference, and computes distance metrics."""
        sample = model_data[index]
        
        # Use precomputed normalized points
        dep_points_norm = sample['dep_points_norm']  # [N_dep, 3]
        uav_points_norm = sample['uav_points_norm']  # [N_uav, 3]
        
        # Get the edge index from the appropriate source
        if 'dep_edge_index' in sample:
            edge_index = sample['dep_edge_index']
        elif 'knn_edge_indices' in sample and 30 in sample['knn_edge_indices']:  # Default to k=30
            edge_index = sample['knn_edge_indices'][30]
        else:
            # Compute KNN if not available
            print(f"Warning: No precomputed edges found for sample {index}, computing KNN")
            edge_index = knn_graph(dep_points_norm, k=30, loop=False)
            edge_index = to_undirected(edge_index, num_nodes=dep_points_norm.size(0))

        # Move to device
        dep_points_norm = dep_points_norm.to(device)
        uav_points_norm = uav_points_norm.to(device)
        edge_index = edge_index.to(device)

        # Run inference
        trained_model.eval()
        with torch.no_grad():
            pred_points_norm = trained_model(dep_points_norm, edge_index)

        # Compute distances
        orig_chamfer_dist = chamfer_distance(dep_points_norm, uav_points_norm)
        upsmpl_chamfer_dist = chamfer_distance(pred_points_norm, uav_points_norm)

        orig_hausdorff_dist = hausdorff_distance(dep_points_norm, uav_points_norm)
        upsmpl_hausdorff_dist = hausdorff_distance(pred_points_norm, uav_points_norm)
        
        # Return CPU data for plotting
        return (
            dep_points_norm.cpu(),
            uav_points_norm.cpu(),
            pred_points_norm.cpu(),
            orig_chamfer_dist,
            upsmpl_chamfer_dist,
            orig_hausdorff_dist,
            upsmpl_hausdorff_dist
        )

    # Process both samples
    dep1, uav1, pred1, chamfer1_orig, chamfer1_upsmpl, hausdorff1_orig, hausdorff1_upsmpl = process_sample(index1)
    dep2, uav2, pred2, chamfer2_orig, chamfer2_upsmpl, hausdorff2_orig, hausdorff2_upsmpl = process_sample(index2)
    
    # Compute overall axis limits for sample index1 (dep1, uav1, pred1)
    pts1_all = torch.cat([dep1, uav1, pred1], dim=0)
    x_min1, x_max1 = pts1_all[:, 0].min().item(), pts1_all[:, 0].max().item()
    y_min1, y_max1 = pts1_all[:, 1].min().item(), pts1_all[:, 1].max().item()
    z_min1, z_max1 = pts1_all[:, 2].min().item(), pts1_all[:, 2].max().item()

    # Compute overall axis limits for sample index2 (dep2, uav2, pred2)
    pts2_all = torch.cat([dep2, uav2, pred2], dim=0)
    x_min2, x_max2 = pts2_all[:, 0].min().item(), pts2_all[:, 0].max().item()
    y_min2, y_max2 = pts2_all[:, 1].min().item(), pts2_all[:, 1].max().item()
    z_min2, z_max2 = pts2_all[:, 2].min().item(), pts2_all[:, 2].max().item()

    # Create a single row of subplots with an empty separator
    fig, axes = plt.subplots(1, 7, figsize=(width, height), subplot_kw={'projection': '3d'})

    def configure_axes(ax, title):
        """Helper function to configure axis formatting."""
        if hide_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title('')  # Remove title
        else:
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6, pad=0)
            ax.ticklabel_format(style='plain', axis='both')
            ax.xaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_visible(False)
            ax.zaxis.get_offset_text().set_visible(False)

    def add_distance_labels(ax, chamfer_dist, hausdorff_dist):
        """Add Chamfer Distance (CD) and Hausdorff Distance (HD) labels to the plot."""
        ax.text2D(0, 0.1, f"CD: {chamfer_dist:.4f}", transform=ax.transAxes, fontsize=7, color='black')
        ax.text2D(0, 0, f"HD: {hausdorff_dist:.4f}", transform=ax.transAxes, fontsize=7, color='black')

    # First three plots => Sample index1
    axes[0].scatter(dep1[:, 0], dep1[:, 1], dep1[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[0], f"3DEP ({index1})")
    add_distance_labels(axes[0], chamfer1_orig, hausdorff1_orig)

    axes[1].scatter(uav1[:, 0], uav1[:, 1], uav1[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[1], f"UAV ({index1})")

    axes[2].scatter(pred1[:, 0], pred1[:, 1], pred1[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[2], f"Upsampled ({index1})")
    add_distance_labels(axes[2], chamfer1_upsmpl, hausdorff1_upsmpl)

    # Empty plot for separation
    axes[3].axis("off")

    # Next three plots => Sample index2
    axes[4].scatter(dep2[:, 0], dep2[:, 1], dep2[:, 2], c='blue', s=0.1, alpha=0.2)
    configure_axes(axes[4], f"3DEP ({index2})")
    add_distance_labels(axes[4], chamfer2_orig, hausdorff2_orig)

    axes[5].scatter(uav2[:, 0], uav2[:, 1], uav2[:, 2], c='green', s=0.1, alpha=0.2)
    configure_axes(axes[5], f"UAV ({index2})")

    axes[6].scatter(pred2[:, 0], pred2[:, 1], pred2[:, 2], c='red', s=0.1, alpha=0.2)
    configure_axes(axes[6], f"Upsampled ({index2})")
    add_distance_labels(axes[6], chamfer2_upsmpl, hausdorff2_upsmpl)

    # --- Set the same axis limits for each group ---
    # For sample index1 subplots (axes 0, 1, 2):
    for ax in [axes[0], axes[1], axes[2]]:
        ax.set_xlim(x_min1, x_max1)
        ax.set_ylim(y_min1, y_max1)
        ax.set_zlim(z_min1, z_max1)
    
    # For sample index2 subplots (axes 4, 5, 6):
    for ax in [axes[4], axes[5], axes[6]]:
        ax.set_xlim(x_min2, x_max2)
        ax.set_ylim(y_min2, y_max2)
        ax.set_zlim(z_min2, z_max2)
    
    # Adjust layout to remove excess whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)

    plt.show()