import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from collections.abc import Sequence
import os
import pandas as pd
os.chdir('/home/jovyan/geoai_veg_map')





def visualize_tile_point_clouds(
    df,
    tile_id: str,
    middle_cols: list[str] = [],  # List of middle column names (0-2 items)
    point_size: float = 1.0,
    point_alpha: float = 0.8,
    colors: list[str] = None,  # Will be set based on number of plots
    elev: float = 25,
    azim: float = -60,
    figsize: tuple[float, float] = (15, 4),
    margins: dict | None = None,
    title_prefixes: list[str] | None = None,
):
    """
    Side-by-side 3-D scatter comparison of point clouds for one tile.
    Always shows input and ground truth, with optional 1-2 middle plots.

    Parameters
    ----------
    middle_cols : list[str]
        Names of columns for middle plots (between input and ground truth).
        Can be empty, or contain 1-2 column names.
    title_prefixes : list[str] | None
        Custom first-line text for the panels. Should have length = 2 + len(middle_cols)
        If None, defaults will be used.
    colors : list[str] | None
        Colors for each plot. Should have length = 2 + len(middle_cols)
        If None, defaults will be used.
    """
    # Validate middle_cols
    if len(middle_cols) > 2:
        raise ValueError("middle_cols can have at most 2 items")
    
    # Total number of plots
    n_plots = 2 + len(middle_cols)  # Input + Ground Truth + middle plots
    
    # Default colors based on number of plots
    if colors is None:
        if n_plots == 2:  # Just input and GT
            colors = ["#D55E00", "#009E73"]
        elif n_plots == 3:  # Input, 1 middle, GT
            colors = ["#D55E00", "#0072B2", "#009E73"]
        else:  # Input, 2 middle, GT
            colors = ["#D55E00", "#0072B2", "#56B4E9", "#009E73"]
    
    # ------------------------------------------------------------------ #
    # 1) Look up row
    row = df.loc[df["tile_id"] == tile_id]
    if row.empty:
        raise ValueError(f"No row found for tile_id={tile_id!r}")
    row = row.iloc[0]

    # 2) Prepare clouds
    clouds = []
    # Always include input
    clouds.append(row["input_points"])
    
    # Add middle clouds
    for col in middle_cols:
        if col not in row:
            raise ValueError(f"Column '{col}' not found in dataframe")
        clouds.append(row[col])
    
    # Always include ground truth
    clouds.append(row["ground_truth_points"])

    # 3) Determine Chamfer columns for middle plots using the mapping
    chamfer_mapping = {
        "combined_pred_points": "combined_chamfer_distance",
        "naip_pred_points": "naip_chamfer_distance",
        "uavsar_pred_points": "uavsar_chamfer_distance",
        "baseline_pred_points": "baseline_chamfer_distance",
    }
    
    chamfer_cols = []
    for col in middle_cols:
        if col in chamfer_mapping and chamfer_mapping[col] in row:
            chamfer_cols.append(chamfer_mapping[col])
        else:
            # Try a fallback by replacing "_points" with "_chamfer_distance"
            possible_chamfer = col.replace("_points", "_chamfer_distance")
            if possible_chamfer in row:
                chamfer_cols.append(possible_chamfer)
            else:
                chamfer_cols.append(None)

    # 4) Title prefixes (first line)
    if title_prefixes is None:
        # Default title for input
        title_prefixes = ["Input"]
        
        # Default titles for middle plots
        for col in middle_cols:
            prefix = col.replace("_pred_points", "").upper()
            title_prefixes.append(prefix)
        
        # Default title for ground truth
        title_prefixes.append("Ground Truth")
    
    if len(title_prefixes) != n_plots:
        raise ValueError(f"title_prefixes must have {n_plots} elements")

    # 5) Assemble full two-line titles
    titles = []
    
    # Input title with chamfer distance
    titles.append(
        f"{title_prefixes[0]}\n"
        f"({len(clouds[0]):,} pts, CD={row['input_chamfer_distance']:.3f})"
    )
    
    # Middle plot titles with chamfer distances if available
    for i, (col, chamfer_col) in enumerate(zip(middle_cols, chamfer_cols)):
        if chamfer_col and chamfer_col in row:
            titles.append(
                f"{title_prefixes[i+1]}\n"
                f"({len(clouds[i+1]):,} pts, CD={row[chamfer_col]:.3f})"
            )
        else:
            titles.append(
                f"{title_prefixes[i+1]}\n"
                f"({len(clouds[i+1]):,} pts)"
            )
    
    # Ground truth title
    titles.append(
        f"{title_prefixes[-1]}\n"
        f"({len(clouds[-1]):,} pts)"
    )

    # ------------------------------------------------------------------ #
    # 6) Shared bounds with 2% padding
    all_pts = np.vstack(clouds)
    mins, maxs = all_pts.min(0), all_pts.max(0)
    pad = 0.02 * (maxs - mins)
    mins, maxs = mins - pad, maxs + pad
    
    # Calculate z_scale based on data dimensions
    # X/Y range is standardized to 10, so adjust z-scale accordingly
    x_range = 10  # Fixed standard range
    z_range = maxs[2] - mins[2]
    z_scale = z_range / x_range

    # 7) Plot
    fig = plt.figure(figsize=figsize)
    
    # Create custom GridSpec with the appropriate number of columns
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, n_plots)
    
    for i, (pts, title, col) in enumerate(zip(clouds, titles, colors), 0):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=point_size, c=col, alpha=point_alpha, linewidths=0
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect([1, 1, z_scale])  # Use calculated z_scale
        
        # Hack: aggressively reduce all margins
        ax.dist = 8  # Reduce distance to the 3D plot (smaller value = bigger plot)
        
        # Remove all decorations to save space
        ax.set_axis_off()
        
        # Add just the title
        ax.set_title(title, fontsize=10, pad=-5)  # Negative pad for title
        
        # Directly manipulate position - the hacky part!
        # Expand each axis beyond its normal bounds
        pos = ax.get_position()
        offset = 0.2  # How much to expand each subplot
        
        # Calculate new position with overlap
        left = max(0, pos.x0 - offset/2)
        width = pos.width + offset
        
        # Apply the new position
        ax.set_position([left, pos.y0, width, pos.height])

    # Apply margins
    default_margins = dict(left=-0.1, right=1.1, bottom=0.05, top=0.95, wspace=-0.5)
    if margins is None:
        fig.subplots_adjust(**default_margins)
    else:
        fig.subplots_adjust(**margins)

    return fig








import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def visualize_tiles_small_multiples(
    df,
    tile_ids,
    pred_col=None,  # Keep for backwards compatibility
    pred_cols=None,  # New parameter for list of prediction columns
    n_cols=2,
    point_size=0.8,
    point_alpha=0.7,
    colors=("#D55E00", "#0072B2", "#009E73", "#CC79A7"),  # Added 4th color
    elev=25,
    azim=-60,
    fig_width=16,
    fig_height_per_row=4,
    title_prefixes=("Input", "Predicted", "Ground Truth"),  # Auto-handles 3 or 4
    title_fontsize=9,
    horizontal_extent=10.0,
    intra_tile_wspace=-0.15,  # Within each tile group
    inter_tile_wspace=0.05,    # Between tile groups
    inter_tile_hspace=0.1,     # Between tile rows
    plot_dist=9,
    title_pad=-8,
    subplot_expand=0.15,
    subplot_position_scale=1.0,
    tight_margins=True,
    fig_margins=None,
    axis_limit_pad=0.02,
    show_chamfer=True,
    chamfer_mapping=None,
    # Overall title parameters
    overall_title=None,
    overall_title_fontsize=16,
    overall_title_y=0.98,
    overall_title_weight='bold',
    overall_title_pad=20,
):
    """
    Create small multiples visualization of point clouds for multiple tiles.
    Each tile shows: input | predicted(s) | ground truth in a row.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing point cloud data
    tile_ids : list
        List of tile IDs to visualize
    pred_col : str, optional
        Single prediction column name (for backwards compatibility)
    pred_cols : list of str, optional
        List of 1-2 prediction column names. If provided, pred_col is ignored.
    n_cols : int
        Number of tile groups per figure row (default: 2)
    point_size : float
        Size of points in scatter plot
    point_alpha : float
        Transparency of points
    colors : tuple
        Colors for plots. Should have 3 colors for single prediction or 4 for dual prediction.
        Default: ("#D55E00", "#0072B2", "#009E73", "#CC79A7")
    elev, azim : float
        3D viewing angles
    fig_width : float
        Total figure width
    fig_height_per_row : float
        Height allocated per row of tiles
    title_prefixes : tuple
        Prefix text for each column's title. Should have 3 elements for single prediction
        or 4 elements for dual prediction. Auto-expands if needed.
    title_fontsize : int
        Font size for titles
    horizontal_extent : float
        Standard horizontal extent for X/Y axes
    intra_tile_wspace : float
        Horizontal spacing WITHIN each tile group
        Negative values create overlap (-1 to 1)
    inter_tile_wspace : float
        Horizontal spacing BETWEEN tile groups
        Negative values create overlap (-1 to 1)
    inter_tile_hspace : float
        Vertical spacing between tile rows
        Negative values create overlap (-1 to 1)
    plot_dist : float
        3D plot distance (smaller = bigger plot)
    title_pad : float
        Padding for titles (negative to bring closer)
    subplot_expand : float
        Factor to expand each subplot beyond boundaries (0-1)
    subplot_position_scale : float
        Scale factor for subplot positioning (>1 makes subplots larger, <1 smaller)
        Use values like 1.2-1.5 to aggressively reduce whitespace
    tight_margins : bool
        Whether to use tight figure margins (overridden by fig_margins if provided)
    fig_margins : dict
        Specific figure margins as dict with keys: left, right, bottom, top
        Values should be between 0 and 1. If None, uses tight_margins setting
    axis_limit_pad : float
        Padding factor for axis limits (0.0 = no padding, 0.02 = 2% padding)
        Smaller values make plots fill more of their space
    show_chamfer : bool
        Whether to show Chamfer distance in titles
    chamfer_mapping : dict
        Custom mapping from pred column names to chamfer column names
    overall_title : str, optional
        Overall title for the entire figure. If None, no overall title is added.
    overall_title_fontsize : int
        Font size for the overall title (default: 16)
    overall_title_y : float
        Y position for overall title (0-1, where 1 is top of figure)
    overall_title_weight : str
        Font weight for overall title ('normal', 'bold', etc.)
    overall_title_pad : float
        Padding between overall title and subplots
        
    Returns
    -------
    fig : matplotlib Figure
    """
    # Handle backwards compatibility and parameter validation
    if pred_cols is None:
        if pred_col is None:
            raise ValueError("Either pred_col or pred_cols must be provided")
        pred_cols = [pred_col]
    elif pred_col is not None:
        raise ValueError("Cannot specify both pred_col and pred_cols. Use pred_cols for multiple predictions.")
    
    n_pred_cols = len(pred_cols)
    if n_pred_cols not in [1, 2]:
        raise ValueError("pred_cols must contain 1 or 2 columns")
    
    # Total plots per tile: input + prediction(s) + ground truth
    plots_per_tile = 2 + n_pred_cols
    
    # Handle title_prefixes - auto-expand if needed
    if len(title_prefixes) == 3 and plots_per_tile == 4:
        # Expand 3 titles to 4 by splitting the middle one
        title_prefixes = (title_prefixes[0], "Predicted 1", "Predicted 2", title_prefixes[2])
    elif len(title_prefixes) == 4 and plots_per_tile == 3:
        # Contract 4 titles to 3 by combining middle ones
        title_prefixes = (title_prefixes[0], "Predicted", title_prefixes[3])
    elif len(title_prefixes) != plots_per_tile:
        raise ValueError(f"title_prefixes must have {plots_per_tile} elements for {n_pred_cols} prediction column(s)")
    
    # Ensure we have enough colors
    if len(colors) < plots_per_tile:
        raise ValueError(f"colors must have at least {plots_per_tile} colors for {n_pred_cols} prediction column(s)")
    
    n_tiles = len(tile_ids)
    n_rows = int(np.ceil(n_tiles / n_cols))
    
    # Set up chamfer mapping
    if chamfer_mapping is None:
        chamfer_mapping = {
            "combined_pred_points": "combined_chamfer_distance",
            "naip_pred_points": "naip_chamfer_distance",
            "uavsar_pred_points": "uavsar_chamfer_distance",
            "baseline_pred_points": "baseline_chamfer_distance",
            "combined_4x_pred_points": "combined_4x_chamfer_distance",
            "combined_6x_pred_points": "combined_6x_chamfer_distance",
        }
    
    # Determine chamfer columns for predictions
    pred_chamfer_cols = []
    for pred_col_name in pred_cols:
        if pred_col_name in chamfer_mapping:
            pred_chamfer_cols.append(chamfer_mapping[pred_col_name])
        else:
            # Try fallback
            possible_chamfer = pred_col_name.replace("_points", "_chamfer_distance")
            pred_chamfer_cols.append(possible_chamfer)
    
    # Create figure
    fig_height = fig_height_per_row * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Apply figure margins
    if fig_margins is not None:
        fig.subplots_adjust(**fig_margins)
    elif tight_margins:
        # Adjust top margin if we have an overall title
        top_margin = 0.95 if overall_title else 0.99
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=top_margin)
    
    # Add overall title if provided
    if overall_title is not None:
        fig.suptitle(
            overall_title,
            fontsize=overall_title_fontsize,
            y=overall_title_y,
            weight=overall_title_weight,
        )
    
    # Create main GridSpec for tile groups
    if n_cols > 1:
        # Calculate column widths - each tile group has plots_per_tile plots
        col_widths = []
        for i in range(n_cols):
            col_widths.extend([1] * plots_per_tile)  # plots_per_tile plots per tile
            if i < n_cols - 1:  # Add spacer between groups (not after last)
                col_widths.append(inter_tile_wspace)
        
        gs_main = GridSpec(n_rows, len(col_widths), 
                          width_ratios=col_widths,
                          hspace=inter_tile_hspace)
    else:
        # Single column, no spacers needed
        gs_main = GridSpec(n_rows, plots_per_tile, hspace=inter_tile_hspace)
    
    for idx, tile_id in enumerate(tile_ids):
        # Get row for this tile
        row = df.loc[df["tile_id"] == tile_id]
        if row.empty:
            print(f"Warning: No row found for tile_id={tile_id!r}, skipping")
            continue
        row = row.iloc[0]
        
        # Get point clouds
        input_pts = row["input_points"]
        pred_pts_list = [row[pred_col_name] for pred_col_name in pred_cols]
        gt_pts = row["ground_truth_points"]
        
        # Combine all clouds for this tile: input + predictions + ground truth
        clouds = [input_pts] + pred_pts_list + [gt_pts]
        
        # Compute bounds for this tile
        all_pts = np.vstack(clouds)
        mins, maxs = all_pts.min(0), all_pts.max(0)
        
        # Apply padding to axis limits if specified
        if axis_limit_pad > 0:
            pad = axis_limit_pad * (maxs - mins)
            mins, maxs = mins - pad, maxs + pad
        
        # Calculate standard view window
        centers = (mins + maxs) / 2
        half_extent = horizontal_extent / 2
        xlims = (centers[0] - half_extent, centers[0] + half_extent)
        ylims = (centers[1] - half_extent, centers[1] + half_extent)
        zlims = (mins[2], maxs[2])
        z_scale = (zlims[1] - zlims[0]) / horizontal_extent
        
        # Determine grid position
        grid_row = idx // n_cols
        grid_col_group = idx % n_cols
        
        # Create titles
        titles = []
        
        # Input title
        if show_chamfer and "input_chamfer_distance" in row:
            titles.append(
                f"{title_prefixes[0]}\n"
                f"({len(input_pts):,} pts, CD={row['input_chamfer_distance']:.3f})"
            )
        else:
            titles.append(f"{title_prefixes[0]}\n({len(input_pts):,} pts)")
        
        # Prediction titles
        for i, (pred_col_name, pred_pts, pred_chamfer_col) in enumerate(zip(pred_cols, pred_pts_list, pred_chamfer_cols)):
            title_idx = 1 + i  # 1 for first prediction, 2 for second prediction
            if show_chamfer and pred_chamfer_col and pred_chamfer_col in row:
                titles.append(
                    f"{title_prefixes[title_idx]}\n"
                    f"({len(pred_pts):,} pts, CD={row[pred_chamfer_col]:.3f})"
                )
            else:
                titles.append(f"{title_prefixes[title_idx]}\n({len(pred_pts):,} pts)")
        
        # Ground truth title
        titles.append(f"{title_prefixes[-1]}\n({len(gt_pts):,} pts)")
        
        # Calculate starting column index for this tile group
        if n_cols > 1:
            # Account for spacer columns: each group has plots_per_tile + 1 spacer (except last)
            start_col = grid_col_group * (plots_per_tile + 1) if grid_col_group < n_cols - 1 else grid_col_group * plots_per_tile + (n_cols - 1)
        else:
            start_col = 0
        
        # Create a sub-gridspec for this tile group with intra-tile spacing
        if n_cols > 1:
            gs_tile = gs_main[grid_row, start_col:start_col+plots_per_tile].subgridspec(
                1, plots_per_tile, wspace=intra_tile_wspace
            )
        else:
            gs_tile = gs_main[grid_row, :].subgridspec(
                1, plots_per_tile, wspace=intra_tile_wspace
            )
        
        # Plot all subplots for this tile
        for i, (pts, title, color) in enumerate(zip(clouds, titles, colors)):
            ax = fig.add_subplot(gs_tile[0, i], projection="3d")
            
            # Plot points
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=point_size, c=color, alpha=point_alpha, linewidths=0
            )
            
            # Set view and limits
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.set_box_aspect([1, 1, z_scale])
            ax.set_axis_off()
            ax.dist = plot_dist
            
            ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
            
            # Expand subplot if requested
            if subplot_expand > 0 or subplot_position_scale != 1.0:
                pos = ax.get_position()
                
                # Apply expansion
                width = pos.width * (1 + subplot_expand)
                height = pos.height * (1 + subplot_expand)
                
                # Apply position scaling
                width *= subplot_position_scale
                height *= subplot_position_scale
                
                # Center the expanded subplot
                left = pos.x0 - (width - pos.width) / 2
                bottom = pos.y0 - (height - pos.height) / 2
                
                # Ensure we don't go outside figure bounds
                left = max(0, min(left, 1 - width))
                bottom = max(0, min(bottom, 1 - height))
                
                ax.set_position([left, bottom, width, height])
    
    return fig


# Backward compatible (3 plots per tile):
# fig = visualize_tiles_small_multiples(
#     df,
#     tile_ids=["tile_15957", "tile_15958"],
#     pred_col="combined_pred_points",  # Single prediction column
#     overall_title="Single Prediction Model"
# )

# NEW: Only input and ground truth (2 plots per tile):
# fig = visualize_tiles_small_multiples(
#     df,
#     tile_ids=["tile_15957", "tile_15958"],
#     pred_col=None,  # No predictions
#     title_prefixes=("Aerial LiDAR", "UAV LiDAR"),  # Only 2 titles needed
#     overall_title="Input vs Ground Truth Comparison"
# )

# Alternative syntax for input + ground truth only:
# fig = visualize_tiles_small_multiples(
#     df,
#     tile_ids=["tile_15957", "tile_15958"],
#     pred_cols=[],  # Empty list = no predictions
#     overall_title="Input vs Ground Truth Comparison"
# )

# New functionality with 1 prediction column (3 plots per tile):
# fig = visualize_tiles_small_multiples(
#     df,
#     tile_ids=["tile_15957", "tile_15958"],
#     pred_cols=["combined_pred_points"],  # List with 1 prediction column
#     overall_title="Single Prediction Model"
# )

# New functionality with 2 prediction columns (4 plots per tile):
# fig = visualize_tiles_small_multiples(
#     df,
#     tile_ids=["tile_15957", "tile_15958"],
#     pred_cols=["combined_pred_points", "naip_pred_points"],  # 2 prediction columns
#     title_prefixes=("Aerial LiDAR\n2015-18", "Combined Model", "NAIP Model", "UAV LiDAR\n2023-24"),
#     overall_title="Model Comparison: Combined vs NAIP"
# )


from matplotlib.gridspec import GridSpec

def visualize_tiles_paired_spacing(
    df,
    tile_ids,
    point_size=1.0,
    point_alpha=0.8,
    colors=("#D55E00", "#009E73"),
    elev=25,
    azim=-60,
    figsize=(16, 8),
    title_prefixes=None,
    title_fontsize=10,
    horizontal_extent=10.0,
    wspace=-0.15,          # Negative for overlap between columns
    hspace=-0.2,           # Negative for overlap between rows
    plot_dist=8,           # Controls 3D plot distance (smaller = bigger plot)
    title_pad=-5,          # Negative padding for titles
    expand_factor=0.15,    # How much to expand each subplot beyond its boundaries
    tight_margins=True,    # Use tight figure margins
):
    """
    Visualize four tiles as Input | GT pairs (2 rows), with extra space between tile pairs.
    """
    if len(tile_ids) != 4:
        raise ValueError("Please provide exactly four tile IDs")
    if title_prefixes is None:
        title_prefixes = ("Input", "Ground Truth")
    if len(title_prefixes) != 2:
        raise ValueError("`title_prefixes` must have exactly two elements.")

    # Prepare point clouds and titles
    all_clouds, all_titles = [], []
    for tile_id in tile_ids:
        row = df.loc[df["tile_id"] == tile_id]
        if row.empty:
            raise ValueError(f"No row found for tile_id={tile_id!r}")
        row = row.iloc[0]
        input_pts = row["input_points"]
        gt_pts = row["ground_truth_points"]
        all_clouds.append((input_pts, gt_pts))
        titles = (
            f"{title_prefixes[0]}\n({len(input_pts):,} pts)",
            f"{title_prefixes[1]}\n({len(gt_pts):,} pts)",
        )
        all_titles.append(titles)

    # Compute shared global axis limits for all clouds
    all_points = np.vstack([pt for pair in all_clouds for pt in pair])
    mins, maxs = all_points.min(0), all_points.max(0)
    centers = (mins + maxs) / 2
    half_extent = horizontal_extent / 2
    xlims = (centers[0] - half_extent, centers[0] + half_extent)
    ylims = (centers[1] - half_extent, centers[1] + half_extent)
    zlims = (mins[2], maxs[2])
    z_scale = (zlims[1] - zlims[0]) / horizontal_extent

    # Create figure with tight borders if requested
    fig = plt.figure(figsize=figsize)
    if tight_margins:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    
    # GridSpec with potentially negative spacing values
    gs = GridSpec(2, 5, 
                 width_ratios=[1, 1, 0.12, 1, 1], 
                 wspace=wspace,   # Can be negative
                 hspace=hspace)   # Can be negative

    for tile_pair in range(2):  # There are 2 tile pairs (rows)
        for in_out in range(2): # 0: input, 1: gt
            # Left tile of the row
            tile_idx = tile_pair * 2 + 0
            col_idx = in_out
            ax = fig.add_subplot(gs[tile_pair, col_idx], projection="3d")
            pts = all_clouds[tile_idx][in_out]
            col = colors[in_out]
            title = all_titles[tile_idx][in_out]
            
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=point_size, c=col, alpha=point_alpha, linewidths=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.set_box_aspect([1, 1, z_scale])
            ax.set_axis_off()
            ax.dist = plot_dist  # Control 3D plot distance
            ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
            
            # Expand subplot beyond its normal boundaries
            if expand_factor > 0:
                pos = ax.get_position()
                width = pos.width * (1 + expand_factor)
                height = pos.height * (1 + expand_factor)
                left = max(0, pos.x0 - (width - pos.width)/2)
                bottom = max(0, pos.y0 - (height - pos.height)/2)
                ax.set_position([left, bottom, width, height])

            # Right tile of the row
            tile_idx = tile_pair * 2 + 1
            col_idx = in_out + 3
            ax = fig.add_subplot(gs[tile_pair, col_idx], projection="3d")
            pts = all_clouds[tile_idx][in_out]
            col = colors[in_out]
            title = all_titles[tile_idx][in_out]
            
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=point_size, c=col, alpha=point_alpha, linewidths=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.set_box_aspect([1, 1, z_scale])
            ax.set_axis_off()
            ax.dist = plot_dist  # Control 3D plot distance
            ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
            
            # Expand subplot beyond its normal boundaries
            if expand_factor > 0:
                pos = ax.get_position()
                width = pos.width * (1 + expand_factor)
                height = pos.height * (1 + expand_factor)
                left = max(0, pos.x0 - (width - pos.width)/2)
                bottom = max(0, pos.y0 - (height - pos.height)/2)
                ax.set_position([left, bottom, width, height])

    return fig




from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

def save_and_crop_figure(fig, filename, crop_box=None, dpi=300, **savefig_kwargs):
    """
    Save figure and optionally crop it using PIL.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Output filename
    crop_box : tuple or None
        Crop box as (left, top, right, bottom) in pixels, or None for no cropping
        Can also be a tuple of ratios (0-1) that will be converted to pixels
    dpi : int
        Resolution for saving
    **savefig_kwargs : dict
        Additional arguments passed to fig.savefig()
    """
    # Set default savefig parameters
    savefig_params = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    savefig_params.update(savefig_kwargs)
    
    # Save the figure
    fig.savefig(filename, **savefig_params)
    
    # Crop if requested
    if crop_box is not None:
        crop_image(filename, crop_box, filename)  # Overwrite original
        print(f"Figure saved and cropped to: {filename}")
    else:
        print(f"Figure saved to: {filename}")

def crop_image(input_path, crop_box, output_path=None):
    """
    Crop an image using PIL.
    
    Parameters
    ----------
    input_path : str
        Path to input image
    crop_box : tuple
        Crop box as (left, top, right, bottom) in pixels
        Or as ratios (0-1) that will be converted to pixels
    output_path : str or None
        Output path. If None, overwrites input
    """
    if output_path is None:
        output_path = input_path
    
    # Open image
    img = Image.open(input_path)
    width, height = img.size
    
    # Convert ratios to pixels if needed
    left, top, right, bottom = crop_box
    if all(0 <= val <= 1 for val in crop_box):
        # Assume ratios, convert to pixels
        left = int(left * width)
        top = int(top * height)
        right = int(right * width)
        bottom = int(bottom * height)
    
    # Crop and save
    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)
    img.close()
    cropped.close()

def save_with_smart_crop(fig, filename, crop_margins=None, dpi=300, **savefig_kwargs):
    """
    Save figure with smart cropping based on content bounds.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Output filename
    crop_margins : dict or None
        Margins to add/remove: {'left': 0.1, 'right': 0.1, 'top': 0.1, 'bottom': 0.1}
        Values are in inches. Negative values crop more aggressively.
    dpi : int
        Resolution for saving
    **savefig_kwargs : dict
        Additional arguments passed to fig.savefig()
    """
    if crop_margins is None:
        crop_margins = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    
    # Get figure size in inches
    width_inch, height_inch = fig.get_size_inches()
    
    # Calculate crop boundaries
    left = crop_margins.get('left', 0)
    right = width_inch - crop_margins.get('right', 0)
    bottom = crop_margins.get('bottom', 0)
    top = height_inch - crop_margins.get('top', 0)
    
    # Ensure valid boundaries
    left = max(0, left)
    right = min(width_inch, right)
    bottom = max(0, bottom)
    top = min(height_inch, top)
    
    if left >= right or bottom >= top:
        print("Warning: Invalid crop boundaries, saving without cropping")
        bbox_inches = 'tight'
    else:
        bbox_inches = Bbox([[left, bottom], [right, top]])
    
    # Set default savefig parameters
    savefig_params = {
        'dpi': dpi,
        'bbox_inches': bbox_inches,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    savefig_params.update(savefig_kwargs)
    
    # Save the figure
    fig.savefig(filename, **savefig_params)
    print(f"Figure saved with smart crop to: {filename}")

def save_with_percentage_crop(fig, filename, crop_percent=0.1, dpi=300, **savefig_kwargs):
    """
    Save figure with percentage-based cropping from all sides.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Output filename
    crop_percent : float
        Percentage to crop from each side (0.1 = 10%)
    dpi : int
        Resolution for saving
    """
    # Get figure size
    width_inch, height_inch = fig.get_size_inches()
    
    # Calculate crop boundaries
    left = width_inch * crop_percent
    right = width_inch * (1 - crop_percent)
    bottom = height_inch * crop_percent
    top = height_inch * (1 - crop_percent)
    
    bbox_inches = Bbox([[left, bottom], [right, top]])
    
    # Set default savefig parameters
    savefig_params = {
        'dpi': dpi,
        'bbox_inches': bbox_inches,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    savefig_params.update(savefig_kwargs)
    
    # Save the figure
    fig.savefig(filename, **savefig_params)
    print(f"Figure saved with {crop_percent*100}% crop to: {filename}")

# Example usage:

# Method 1: PIL-based cropping
# save_and_crop_figure(
#     fig, 
#     "cropped_visualization.png",
#     crop_box=(100, 50, 1800, 1000),  # pixels: (left, top, right, bottom)
#     dpi=300
# )

# Method 2: Ratio-based cropping (crop 10% from each side)
# save_and_crop_figure(
#     fig,
#     "ratio_cropped.png", 
#     crop_box=(0.1, 0.1, 0.9, 0.9),  # ratios: (left, top, right, bottom)
#     dpi=300
# )

# Method 3: Smart cropping with inch-based margins
# save_with_smart_crop(
#     fig,
#     "smart_cropped.png",
#     crop_margins={'left': 0.5, 'right': 0.5, 'top': 1.0, 'bottom': 0.2},
#     dpi=300
# )

# Method 4: Percentage-based uniform cropping
# save_with_percentage_crop(
#     fig,
#     "percentage_cropped.png",
#     crop_percent=0.15,  # Crop 15% from all sides
#     dpi=300
# )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_chamfer_distance(df, model_order=None, whisker_percentiles=[5, 95], x_max=None, 
                                 model_label_size=12, colors=None, box_width=0.6,
                                 figsize=(10, 6), save_path=None, dpi=300, title="Chamfer Distance Distribution By Model"):
    """
    Create a horizontal boxplot showing Chamfer distance distribution for each model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the model comparison data
    model_order : list, optional
        List of model keys in desired order from top to bottom. 
        Options: ['input', 'baseline', 'naip', 'uavsar', 'fused']
        Default order: ['input', 'baseline', 'naip', 'uavsar', 'fused']
    whisker_percentiles : list, optional
        Two-element list specifying whisker percentiles [lower, upper]. Default is [5, 95]
    x_max : float, optional
        Maximum value for x-axis (Chamfer Distance axis). If None, automatically determined
    model_label_size : int, optional
        Font size for model category labels on y-axis. Default is 12
    colors : str or list, optional
        Color palette for boxes. If str (single hex color), all boxes will be that color.
        If list, should contain hex colors for each box. If None, uses default palette.
    box_width : float, optional
        Width of the boxes. Default is 0.6. Higher values make wider boxes (max ~0.9)
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 6)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved
    dpi : int, optional
        Resolution for saved figure. Default is 300
    title : str, optional
        Plot title. Default is "Chamfer Distance Distribution By Model"
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Define column mapping for Chamfer Distance
    cd_cols = {
        "input":     "input_chamfer_distance",
        "baseline":  "baseline_chamfer_distance", 
        "naip":      "naip_chamfer_distance",
        "uavsar":    "uavsar_chamfer_distance",
        "fused":     "combined_chamfer_distance",
    }
    
    # Model display labels
    model_labels = {
        "input": "Input",
        "baseline": "Baseline",
        "naip": "NAIP", 
        "uavsar": "UAVSAR",
        "fused": "Fused"
    }
    
    # Set default model order if not provided
    if model_order is None:
        model_order = ["input", "baseline", "naip", "uavsar", "fused"]
    
    # Validate model_order contains valid keys
    valid_models = set(cd_cols.keys())
    if not set(model_order).issubset(valid_models):
        invalid_models = set(model_order) - valid_models
        raise ValueError(f"Invalid model keys: {invalid_models}. Valid options: {list(valid_models)}")
    
    # Create model order list with labels
    model_display_order = [(model_key, model_labels[model_key]) for model_key in model_order]
    
    # Prepare data for boxplot
    plot_data = []
    plot_labels = []
    
    for model_key, model_label in model_display_order:
        col_name = cd_cols[model_key]
        # Remove any infinite or NaN values for clean statistics
        clean_data = df[col_name].replace([np.inf, -np.inf], np.nan).dropna()
        plot_data.append(clean_data.values)
        plot_labels.append(model_label)
    
    # Reverse order for horizontal boxplot (top to bottom display)
    plot_data = plot_data[::-1]
    plot_labels = plot_labels[::-1]
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle color palette
    if colors is None:
        # Default colors
        all_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']  
        box_colors = all_colors[:len(plot_data)]
    elif isinstance(colors, str):
        # Single color for all boxes
        box_colors = [colors] * len(plot_data)
    elif isinstance(colors, list):
        # List of colors provided
        if len(colors) < len(plot_data):
            raise ValueError(f"Not enough colors provided. Need {len(plot_data)} colors, got {len(colors)}")
        box_colors = colors[:len(plot_data)]
    else:
        raise ValueError("colors must be None, a hex color string, or a list of hex color strings")
    
    # Create the horizontal boxplot
    bp = ax.boxplot(plot_data, 
                    tick_labels=plot_labels,
                    vert=False,  # Horizontal orientation
                    patch_artist=True,  # Allow filling boxes with colors
                    showfliers=False,  # Don't show outlier dots
                    whis=whisker_percentiles,  # Set whisker percentiles
                    widths=box_width)  # Set box width
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style the other boxplot elements
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.0)
    
    # Make median lines more prominent
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2.0)
    
    # Customize the plot
    ax.set_xlabel('Chamfer Distance (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    
    # Set model label sizes (y-axis tick labels)
    ax.tick_params(axis='y', labelsize=model_label_size)
    
    # Improve layout
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Set x-axis limits
    ax.set_xlim(left=0)
    if x_max is not None:
        ax.set_xlim(right=x_max)
    
    # Add subtle styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_rq3_unified_scatter(df, model_order=None, y_max=None, colors=None, 
                            alpha=0.3, trend_line_width=2, point_size=1,
                            cd_percentile_cap=None, height_percentile_cap=None,
                            figsize=(12, 8), save_path=None, dpi=300, 
                            title="Reconstruction Error vs Canopy Height Change By Model"):
    """
    Create a unified scatter plot showing the relationship between canopy height change 
    and reconstruction error, with losses on the left (negative) and gains on the right (positive).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the model comparison data
    model_order : list, optional
        List of model keys to include and their order.
        Options: ['input', 'baseline', 'naip', 'uavsar', 'fused']
        Default: ['baseline', 'naip', 'uavsar', 'fused'] (excludes input)
    y_max : float, optional
        Maximum value for y-axis (Chamfer Distance). If None, automatically determined
    colors : str or list, optional
        Color palette. If str (single hex color), all models will be shades of that color.
        If list, should contain hex colors for each model. If None, uses default palette.
    alpha : float, optional
        Transparency of scatter points. Default is 0.3
    trend_line_width : float, optional
        Width of trend lines. Default is 2
    point_size : float, optional
        Size of scatter points. Default is 1
    cd_percentile_cap : float, optional
        Percentile to cap Chamfer Distance values (e.g., 95.0 to cap at 95th percentile).
        If None, no capping is applied.
    height_percentile_cap : float, optional
        Percentile to cap canopy height change values separately for gains and losses.
        If None, no capping is applied.
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 8)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved
    dpi : int, optional
        Resolution for saved figure. Default is 300
    title : str, optional
        Main plot title. Default is "Reconstruction Error vs Canopy Height Change"
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Define column mapping for Chamfer Distance
    cd_cols = {
        "input":     "input_chamfer_distance",
        "baseline":  "baseline_chamfer_distance", 
        "naip":      "naip_chamfer_distance",
        "uavsar":    "uavsar_chamfer_distance",
        "fused":     "combined_chamfer_distance",
    }
    
    # Model display labels
    model_labels = {
        "input": "Input",
        "baseline": "Baseline",
        "naip": "NAIP", 
        "uavsar": "UAVSAR",
        "fused": "Fused"
    }
    
    # Set default model order if not provided (exclude input by default)
    if model_order is None:
        model_order = ["baseline", "naip", "uavsar", "fused"]
    
    # Validate model_order contains valid keys
    valid_models = set(cd_cols.keys())
    if not set(model_order).issubset(valid_models):
        invalid_models = set(model_order) - valid_models
        raise ValueError(f"Invalid model keys: {invalid_models}. Valid options: {list(valid_models)}")
    
    # Prepare data - clean and apply Chamfer Distance outlier removal
    numeric_cols = ["net_canopy_height_change"] + [cd_cols[model] for model in model_order]
    clean_df = (
        df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    
    # Apply percentile capping for Chamfer Distance outliers
    if cd_percentile_cap is not None:
        for model in model_order:
            col_name = cd_cols[model]
            cd_threshold = np.percentile(clean_df[col_name], cd_percentile_cap)
            clean_df[col_name] = np.minimum(clean_df[col_name], cd_threshold)
    
    # Split into gains and losses for height capping
    gains_mask = clean_df["net_canopy_height_change"] >= 0
    losses_mask = clean_df["net_canopy_height_change"] < 0
    
    # Apply percentile capping for height change outliers separately for gains and losses
    if height_percentile_cap is not None:
        # Cap gains (positive values)
        if gains_mask.sum() > 0:
            gains_values = clean_df.loc[gains_mask, "net_canopy_height_change"]
            gains_threshold = np.percentile(gains_values, height_percentile_cap)
            clean_df.loc[gains_mask, "net_canopy_height_change"] = np.minimum(gains_values, gains_threshold)
        
        # Cap losses (negative values) - use absolute values for percentile calculation
        if losses_mask.sum() > 0:
            losses_values = clean_df.loc[losses_mask, "net_canopy_height_change"]
            abs_losses = losses_values.abs()
            losses_threshold = np.percentile(abs_losses, height_percentile_cap)
            # Keep negative sign but cap the absolute value
            clean_df.loc[losses_mask, "net_canopy_height_change"] = np.maximum(losses_values, -losses_threshold)
    
    # Count final sample sizes
    final_gains_count = (clean_df["net_canopy_height_change"] >= 0).sum()
    final_losses_count = (clean_df["net_canopy_height_change"] < 0).sum()
    
    # Handle color palette
    if colors is None:
        # Default colors
        all_colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c', '#e67e22']  
        model_colors = {model: all_colors[i] for i, model in enumerate(model_order)}
    elif isinstance(colors, str):
        # Generate shades of single color
        base_color = colors
        model_colors = {}
        for i, model in enumerate(model_order):
            # Create variations by adjusting brightness
            model_colors[model] = base_color
    elif isinstance(colors, list):
        # List of colors provided
        if len(colors) < len(model_order):
            raise ValueError(f"Not enough colors provided. Need {len(model_order)} colors, got {len(colors)}")
        model_colors = {model: colors[i] for i, model in enumerate(model_order)}
    else:
        raise ValueError("colors must be None, a hex color string, or a list of hex color strings")
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate x-axis range for padding
    x_min = clean_df["net_canopy_height_change"].min()
    x_max = clean_df["net_canopy_height_change"].max()
    x_range = x_max - x_min
    x_padding = x_range * 0.15  # 15% padding on each side for labels
    
    # Plot data for each model
    for model in model_order:
        col_name = cd_cols[model]
        model_label = model_labels[model]
        color = model_colors[model]
        
        # Get x and y data
        x_data = clean_df["net_canopy_height_change"]
        y_data = clean_df[col_name]
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=alpha, s=point_size, 
                  color=color, label=model_label, edgecolors='none')
        
        # Trend line using LOWESS
        if len(x_data) > 10:  # Need enough points for trend line
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data.iloc[sorted_indices]
            y_sorted = y_data.iloc[sorted_indices]
            
            # LOWESS smoothing
            smoothed = lowess(y_sorted, x_sorted, frac=0.3)
            trend_x, trend_y = smoothed[:, 0], smoothed[:, 1]
            
            line = ax.plot(trend_x, trend_y, color=color, linewidth=trend_line_width)[0]
            
            # Add labels at both ends of trend line
            if len(trend_x) > 0:
                # Left end label (losses side)
                ax.text(trend_x[0], trend_y[0], f'{model_label}  ', 
                       color=color, fontweight='bold', fontsize=16,
                       verticalalignment='center', horizontalalignment='right')
                
                # Right end label (gains side)
                ax.text(trend_x[-1], trend_y[-1], f'  {model_label}', 
                       color=color, fontweight='bold', fontsize=10,
                       verticalalignment='center', horizontalalignment='left')
    
    # Set x-axis limits with padding for labels
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    
    # Add vertical line at x=0 to separate gains and losses
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Net Canopy Height Change (m)', fontsize=16)#, fontweight='bold')
    ax.set_ylabel('Chamfer Distance (m)', fontsize=16)#, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=10)

    # Set tick label sizes
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add text annotations for sample sizes
    ax.text(0.02, 0.98, f'Losses: N={final_losses_count}', 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(0.98, 0.98, f'Gains: N={final_gains_count}', 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add directional labels for canopy losses and gains
    # Calculate positions relative to zero in data coordinates
    y_bottom = ax.get_ylim()[0]
    label_y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1 # 2% of y-range above bottom
    label_x_offset = x_range * 0.02  # 2% of x-range from zero
    
    ax.text(-label_x_offset, y_bottom + label_y_offset, '⟵ Canopy Losses', 
            fontsize=12, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    ax.text(label_x_offset, y_bottom + label_y_offset, 'Canopy Gains ⟶', 
            fontsize=12, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    

    # Set y-axis limits
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)
    else:
        ax.set_ylim(bottom=0)
    
    # Add subtle styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


if __name__ == "__main__":

    df_path = "data/processed/model_data/point_cloud_comparison_df_0516_e100_w8x.pkl"

    df = pd.read_pickle(df_path)


    intro_pt_clouds = visualize_tiles_paired_spacing(
        df,
        tile_ids=["tile_15957", "tile_15958", "tile_31561", "tile_30473"],
        point_size=1.8,
        point_alpha=0.6,
        colors=("#D55E00", "#009E73"),
        title_prefixes = ("Aerial LiDAR\n2015-18", "UAV LiDAR\n2023-24"),
        title_fontsize=20,
        horizontal_extent=10.0,
        figsize = (20, 20),
    )

    save_and_crop_figure(
        intro_pt_clouds, 
        "manuscript/figures/intro_pt_clouds.png",
        (0.04, 0.0, 0.96, .985),  # ratios: (left, top, right, bottom)
        dpi=300
    )


    single_veg_loss_example = visualize_tile_point_clouds(
        df,
        tile_id="tile_31561",
        middle_cols=["baseline_pred_points","combined_pred_points"],
        point_size=1,
        point_alpha=0.7,
        elev = 4,
        azim=  -50,
        figsize=(10, 5),
        margins=dict(left=-0.2, right=1, bottom=0.05, top=0.95, wspace=-0.27),
        title_prefixes = ("Aerial LiDAR (2015-18)", "Predicted - LiDAR Only","Predicted - Optical+SAR", "UAV LiDAR (2023-24)")
    )


    save_and_crop_figure(
        single_veg_loss_example, 
        "manuscript/figures/single_veg_loss_example.png",
        (0.04, 0.0, 0.96, .985),  # ratios: (left, top, right, bottom)
        dpi=1000
    )



    veg_growth = ['tile_21167', 'tile_17898', 'tile_34378', 'tile_27176', 'tile_22627', 'tile_31490', 'tile_16730', 'tile_24896', 'tile_21162', 'tile_33122', 'tile_16869', 'tile_20241']
    veg_growth_8x = visualize_tiles_small_multiples(
        df,
        tile_ids=veg_growth[0:8],
        pred_col="combined_8x_pred_points",
        n_cols=2,
        fig_width=8,
        fig_height_per_row=2,
        intra_tile_wspace=-0.05,   # Tight spacing within each tile group
        inter_tile_wspace=0.15,     # More space between tile groups
        inter_tile_hspace=0.05,    # Small space between rows  
        # Whitespace reduction
        axis_limit_pad=0.0,          # No padding around data
        subplot_position_scale=1.1,   # Make subplots X% larger
        subplot_expand=0,          # Additional X% expansion
        plot_dist=7,                 # Closer camera for bigger plots
        # Tight figure margins
        fig_margins=dict(left=-0.0, right=1, bottom=0.005, top=0.995),

        title_prefixes=("Aerial LiDAR", "Predicted 8x", "UAV LiDAR"),
        point_size=0.4,
        point_alpha=0.5,
        elev=25,
        azim=-60,
        title_fontsize=10,
        title_pad=-40,
        tight_margins=True,
        show_chamfer=False,
        chamfer_mapping=None,
        overall_title="Vegetation Growth Prediction Examples: Aerial to UAV LiDAR",
        overall_title_fontsize=14,
        overall_title_y=1.09,
        overall_title_weight='bold',
        overall_title_pad=0,
    )


    save_and_crop_figure(
        veg_growth_8x, 
        "manuscript/figures/veg_growth_8x.png",
        (0.03, 0.0, 0.97, 1),  # ratios: (left, top, right, bottom)
        dpi=1000
    )


    
    veg_growth2_8x = visualize_tiles_small_multiples(
        df,
        tile_ids=veg_growth[8:12],
        pred_col="combined_8x_pred_points",
        n_cols=2,
        fig_width=8,
        fig_height_per_row=2,
        intra_tile_wspace=-0.05,   # Tight spacing within each tile group
        inter_tile_wspace=0.15,     # More space between tile groups
        inter_tile_hspace=0.05,    # Small space between rows  
        # Whitespace reduction
        axis_limit_pad=0.0,          # No padding around data
        subplot_position_scale=1.1,   # Make subplots X% larger
        subplot_expand=0,          # Additional X% expansion
        plot_dist=7,                 # Closer camera for bigger plots
        # Tight figure margins
        fig_margins=dict(left=-0.0, right=1, bottom=0.005, top=0.995),

        title_prefixes=("Aerial LiDAR", "Predicted 8x", "UAV LiDAR"),
        point_size=0.4,
        point_alpha=0.5,
        elev=25,
        azim=-60,
        title_fontsize=10,
        title_pad=-40,
        tight_margins=True,
        show_chamfer=False,
        chamfer_mapping=None,
        overall_title="Vegetation Growth Prediction Examples: Aerial to UAV LiDAR",
        overall_title_fontsize=14,
        overall_title_y=1.09,
        overall_title_weight='bold',
        overall_title_pad=0,
    )


    save_and_crop_figure(
        veg_growth2_8x, 
        "manuscript/figures/veg_growth2_8x.png",
        (0.03, 0.0, 0.97, 1),  # ratios: (left, top, right, bottom)
        dpi=1000
    )



    veg_loss = [ "tile_31561",'tile_19330', 'tile_21821', 'tile_27602', 'tile_18855', 'tile_19090', 'tile_26069', 'tile_29323', 'tile_19094', 'tile_28456','tile_22954', 'tile_27815','tile_29727','tile_18851']
    veg_loss_2x = visualize_tiles_small_multiples(
        df,
        tile_ids=veg_loss[0:8],
        pred_col="combined_pred_points",
        n_cols=2,
        fig_width=8,
        fig_height_per_row=2,
        intra_tile_wspace=-0.14,   # Tight spacing within each tile group
        inter_tile_wspace=0.05,     # More space between tile groups
        inter_tile_hspace=0.0,    # Small space between rows  
        # Whitespace reduction
        axis_limit_pad=0.0,          # No padding around data
        subplot_position_scale=1.1,   # Make subplots X% larger
        subplot_expand=0,          # Additional X% expansion
        plot_dist=2,                 # Closer camera for bigger plots
        # Tight figure margins
        fig_margins=dict(left=0, right=1, bottom=0.005, top=0.995),

        title_prefixes=("Aerial LiDAR", "Predicted 2x", "UAV LiDAR"),
        point_size=0.4,
        point_alpha=0.5,
        elev=20,
        azim=-80,
        title_fontsize=10,
        title_pad=-40,
        tight_margins=True,
        show_chamfer=False,
        chamfer_mapping=None,
        overall_title="Vegetation Loss Prediction Examples: Aerial to UAV LiDAR",
        overall_title_fontsize=14,
        overall_title_y=1.09,
        overall_title_weight='bold',
        overall_title_pad=0,
    )


    save_and_crop_figure(
        veg_loss_2x, 
        "manuscript/figures/veg_loss_2x.png",
        (0.045, 0.0, 0.955, .98),  # ratios: (left, top, right, bottom)
        dpi=1000
    )




    veg_loss2_2x = visualize_tiles_small_multiples(
        df,
        tile_ids=veg_loss[8:16],
        pred_col="combined_pred_points",
        n_cols=2,
        fig_width=8,
        fig_height_per_row=2,
        intra_tile_wspace=-0.14,   # Tight spacing within each tile group
        inter_tile_wspace=0.05,     # More space between tile groups
        inter_tile_hspace=0.0,    # Small space between rows  
        # Whitespace reduction
        axis_limit_pad=0.0,          # No padding around data
        subplot_position_scale=1.1,   # Make subplots X% larger
        subplot_expand=0,          # Additional X% expansion
        plot_dist=2,                 # Closer camera for bigger plots
        # Tight figure margins
        fig_margins=dict(left=0, right=1, bottom=0.005, top=0.995),

        title_prefixes=("Aerial LiDAR", "Predicted 2x", "UAV LiDAR"),
        point_size=0.4,
        point_alpha=0.5,
        elev=20,
        azim=-80,
        title_fontsize=10,
        title_pad=-40,
        tight_margins=True,
        show_chamfer=False,
        chamfer_mapping=None,
        overall_title="Vegetation Loss Prediction Examples: Aerial to UAV LiDAR",
        overall_title_fontsize=14,
        overall_title_y=1.09,
        overall_title_weight='bold',
        overall_title_pad=0,
    )

    save_and_crop_figure(
        veg_loss2_2x, 
        "manuscript/figures/veg_loss2_2x.png",
        (0.045, 0.0, 0.955, .98),  # ratios: (left, top, right, bottom)
        dpi=1000
    )




    baseline_prediction_example = visualize_tiles_small_multiples(
        df,
        tile_ids=['tile_971','tile_15957','tile_30473','tile_19335'],
        pred_col="baseline_pred_points",
        n_cols=2,
        fig_width=8,
        fig_height_per_row=2,
        intra_tile_wspace=-0.07,   # Tight spacing within each tile group
        inter_tile_wspace=0.08,     # More space between tile groups
        inter_tile_hspace=-0.18,    # Small space between rows  
        # Whitespace reduction
        axis_limit_pad=0.0,          # No padding around data
        subplot_position_scale=1.1,   # Make subplots X% larger
        subplot_expand=0,          # Additional X% expansion
        plot_dist=7,                 # Closer camera for bigger plots
        # Tight figure margins
        fig_margins=dict(left=-0.0, right=1, bottom=0.000, top=1),

        title_prefixes=("Aerial LiDAR", "Baseline Prediction", "UAV LiDAR"),
        point_size=0.4,
        point_alpha=0.5,
        elev=25,
        azim=-60,
        title_fontsize=9,
        title_pad=-40,
        tight_margins=True,
        show_chamfer=False,
        chamfer_mapping=None,
        overall_title="Baseline (LiDAR only) Prediction Examples",
        overall_title_fontsize=14,
        overall_title_y=1.08,
        overall_title_weight='bold',
        overall_title_pad=0,
    )

    save_and_crop_figure(
        baseline_prediction_example, 
        "manuscript/figures/baseline_prediction_example.png",
        (0.04, 0.0, 0.96, .985),  # ratios: (left, top, right, bottom)
        dpi=1000
    )

    baseline_v_input_boxplot, _  = plot_boxplot_chamfer_distance(df,   model_order=['input','baseline'],
                                    x_max=5.7,whisker_percentiles = [10, 90],
                                    colors =  [ "#bcce78ff","#5a7c65ff"], title = "Chamfer Distance Distribution - Baseline vs Input",
                                    model_label_size=17, figsize=(10, 2.6), box_width  = 0.8)

    save_and_crop_figure(
        baseline_v_input_boxplot, 
        "manuscript/figures/baseline_v_input_boxplot.png",
        (0.0, 0.0, 1, 1),  # ratios: (left, top, right, bottom)
        dpi=500
    )
    
    
    boxplot_by_model, _ = plot_boxplot_chamfer_distance(df,  model_order=['baseline', 'uavsar', 'naip', 'fused'],
                                x_max=1.3,whisker_percentiles = [10, 90],
                                colors = ["#bcce78ff","#bcce78ff", "#bcce78ff","#5a7c65ff"],
                                model_label_size=17, figsize=(7, 3), box_width  = 0.8)

    save_and_crop_figure(
        boxplot_by_model, 
        "manuscript/figures/boxplot_by_model.png",
        (0.0, 0.0, 1, 1),  # ratios: (left, top, right, bottom)
        dpi=500
    )


    error_vs_cnpy_chng, ax = plot_rq3_unified_scatter(df, y_max=4.3,cd_percentile_cap=99.5, height_percentile_cap=99.5, point_size = 2,figsize=(14, 6))

    save_and_crop_figure(
        error_vs_cnpy_chng, 
        "manuscript/figures/error_vs_cnpy_chng.png",
        (0.0, 0.0, 1, 1),  # ratios: (left, top, right, bottom)
        dpi=1000
    )


   