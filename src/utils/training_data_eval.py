import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from rasterio.transform import from_origin
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from matplotlib.ticker import FuncFormatter

# --- Helper Functions ---

def tensor_to_structured_array(tensor):
    data = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    arr = np.zeros(data.shape[0], dtype=[('X', data.dtype), ('Y', data.dtype), ('Z', data.dtype)])
    arr['X'] = data[:, 0]
    arr['Y'] = data[:, 1]
    arr['Z'] = data[:, 2]
    return arr

def print_tile_debug_info(data_dict, tile_index=0):
    bbox = data_dict.get('bbox')
    bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if bbox is not None else "N/A"
    print(f"Tile index: {tile_index} | Bounding box: {bbox_str}")
    
    if 'uav_points' in data_dict:
        uav_np = data_dict['uav_points'].cpu().numpy() if data_dict['uav_points'].is_cuda else data_dict['uav_points'].numpy()
        num_uav = uav_np.shape[0]
        uav_min = np.min(uav_np[:, :3], axis=0)
        uav_max = np.max(uav_np[:, :3], axis=0)
        print(f"UAV pts: {num_uav:,} | Min: {uav_min.tolist()} | Max: {uav_max.tolist()}")
    if 'dep_points' in data_dict:
        dep_np = data_dict['dep_points'].cpu().numpy() if data_dict['dep_points'].is_cuda else data_dict['dep_points'].numpy()
        num_dep = dep_np.shape[0]
        dep_min = np.min(dep_np[:, :3], axis=0)
        dep_max = np.max(dep_np[:, :3], axis=0)
        diff_min = (uav_min - dep_min).tolist() if 'uav_points' in data_dict else "N/A"
        diff_max = (uav_max - dep_max).tolist() if 'uav_points' in data_dict else "N/A"
        print(f"3DEP pts: {num_dep:,} | Min: {dep_min.tolist()} | Max: {dep_max.tolist()} | Diff Min: {diff_min} | Diff Max: {diff_max}")
    if 'uavsar_imgs' in data_dict and data_dict['uavsar_imgs']:
        meta = max(enumerate(data_dict.get('uavsar_imgs_meta', [])), key=lambda x: pd.Timestamp(x[1]['datetime']))[1]
        uavsar_img = data_dict['uavsar_imgs'][0]
        shape = uavsar_img.shape
        bands = meta.get('bands', list(range(1, shape[0]+1))) if shape[0] > 1 else "N/A"
        date = meta.get('date', meta.get('datetime', 'N/A'))
        print(f"UAVSAR: {shape} | Bands: {bands} | Date: {date}")
    if 'naip_imgs' in data_dict and data_dict['naip_imgs']:
        meta = max(enumerate(data_dict.get('naip_imgs_meta', [])), key=lambda x: pd.Timestamp(x[1]['datetime']))[1]
        naip_img = data_dict['naip_imgs'][0]
        shape = naip_img.shape
        bands = meta.get('bands', list(range(1, shape[0]+1))) if shape[0] > 1 else "N/A"
        date = meta.get('date', meta.get('datetime', 'N/A'))
        res = data_dict.get('naip_resolution', "N/A")
        num_naip = len(data_dict['naip_imgs'])
        print(f"NAIP: Res: {res} m, {num_naip} imgs | Img1: {shape} | Bands: {bands} | Date: {date}")
    print("-" * 80)

def reconstruct_raster_from_tensor(img_tensor, meta):
    bbox = meta.get('bbox')
    resolution = meta.get('resolution')
    if bbox is None or resolution is None:
        raise ValueError("Metadata must contain 'bbox' and 'resolution'.")
    xmin, ymin, xmax, ymax = bbox
    transform = from_origin(xmin, ymax, resolution, resolution)
    img_np = img_tensor.cpu().numpy() if img_tensor.is_cuda else img_tensor.numpy()
    if img_np.ndim == 4:
        img_np = img_np[0]
    if img_np.ndim == 3:
        if img_np.shape[0] >= 3:
            rgb = img_np[:3, :, :]
            rgb = np.transpose(rgb, (1, 2, 0))
        else:
            rgb = img_np[0, :, :]
    else:
        rgb = img_np
    extent = (xmin, xmax, ymin, ymax)
    return rgb, extent, transform

def compute_dsm(points, bbox, resolution=0.5):
    xmin, ymin, xmax, ymax = bbox
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    dsm = np.full((height, width), np.nan, dtype=np.float32)
    cols = np.floor((points[:, 0] - xmin) / resolution).astype(int)
    rows = np.floor((points[:, 1] - ymin) / resolution).astype(int)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols = cols[valid]
    rows = rows[valid]
    zvals = points[valid, 2]
    for r, c, z in zip(rows, cols, zvals):
        if np.isnan(dsm[r, c]) or z > dsm[r, c]:
            dsm[r, c] = z
    return dsm

def plot_pointcloud_on_axis(ax, structured_pc, xlim, ylim, zlim, title, standard_ticks=None):
    x = structured_pc['X']
    y = structured_pc['Y']
    z = structured_pc['Z']
    ax.scatter(x, y, z, s=0.1, c=z, cmap='viridis')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)
    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

def plot_dsm(ax, dsm, extent, title, standard_ticks=None):
    im = ax.imshow(dsm, cmap='terrain', extent=extent, origin='lower')
    ax.set_title(title, fontsize=10)

    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_uavsar_grid(ax, img_tensor, meta, standard_ticks=None, band_names=None):
    try:
        raster, extent, _ = reconstruct_raster_from_tensor(img_tensor, meta)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        return
    
    # Default to displaying first band in grayscale if bands not specified
    if band_names is None or len(band_names) < 3:
        hh_band = raster[:, :, 0] if raster.ndim == 3 else raster
        vmin, vmax = np.percentile(hh_band, 2), np.percentile(hh_band, 98)
        ax.imshow(hh_band, cmap='gray', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
    else:
        # Get band indices from metadata
        bands_map = {}
        if 'bands' in meta:
            for i, band_name in enumerate(meta['bands']):
                bands_map[band_name] = i
        else:
            # Default mapping if not specified in metadata
            possible_bands = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]
            for i, band_name in enumerate(possible_bands):
                if i < (raster.shape[2] if raster.ndim == 3 else 1):
                    bands_map[band_name] = i
        
        # Create RGB composite
        rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.float32)
        
        # Extract bands for R, G, B channels
        for i, band_name in enumerate(band_names[:3]):
            if band_name in bands_map:
                band_idx = bands_map[band_name]
                if raster.ndim == 3 and band_idx < raster.shape[2]:
                    band_data = raster[:, :, band_idx]
                elif band_idx == 0:
                    band_data = raster
                else:
                    # Skip if band doesn't exist
                    continue
                
                # Normalize to 0-1 range for this channel
                vmin, vmax = np.percentile(band_data, 2), np.percentile(band_data, 98)
                normalized = np.clip((band_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
                rgb[:, :, i] = normalized
        
        ax.imshow(rgb, extent=extent, origin='upper')
    
    date_str = meta.get('date', meta.get('datetime', ''))
    title = f"{date_str}"
    if band_names and len(band_names) >= 3:
        title += f"\n{band_names[0]}-{band_names[1]}-{band_names[2]}"
    ax.set_title(title, fontsize=10)
    
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))

def plot_naip_grid(ax, img_tensor, meta, bbox_overlay=None, standard_ticks=None):
    try:
        rgb, extent, _ = reconstruct_raster_from_tensor(img_tensor, meta)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        return
    
    ax.imshow(rgb.astype(np.uint8), extent=extent, origin='upper')
    
    # Draw the original bounding box on top (if provided)
    if bbox_overlay is not None:
        rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                         bbox_overlay[2]-bbox_overlay[0],
                         bbox_overlay[3]-bbox_overlay[1],
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    date_str = meta.get('date', meta.get('datetime', ''))
    ax.set_title(f"{date_str}", fontsize=10)
    # ax.set_xlabel("X (CRS)")
    # ax.set_ylabel("Y (CRS)")
    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))



# --- Main Visualization Function ---

def visualize_tile_data(data_dict, tile_index=0, elev=45, azim=45, modalities=None, uavsar_bands=None):
    """
    Visualize selected modalities from a tile data dictionary.
    All plots will use standardized ticks based on the bounding box.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing multimodal data.
    tile_index : int, optional
        Index of the tile to visualize when multiple tiles are available.
    elev : float, optional
        Elevation angle for 3D plots.
    azim : float, optional
        Azimuth angle for 3D plots.
    modalities : list of str, optional
        List of modalities to visualize. Valid values are "PointClouds", "DSMs", "UAVSAR", "NAIP".
    uavsar_bands : list of str, optional
        List of three band names to use for UAVSAR RGB composite. 
        Valid values are ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"].
        If None, the first band will be displayed in grayscale.
    """
    if modalities is None:
        modalities = ["PointClouds", "DSMs", "NAIP"]
    
    print_tile_debug_info(data_dict, tile_index)
    
    figures = {}
    
    # Get the bounding box for standardized ticks
    bbox = data_dict.get('bbox')
    if bbox is None:
        print("Warning: No bounding box found in the data dictionary. Using point cloud extents instead.")
        uav_arr = tensor_to_structured_array(data_dict['uav_points'])
        dep_arr = tensor_to_structured_array(data_dict['dep_points'])
        all_x = np.concatenate([uav_arr['X'], dep_arr['X']])
        all_y = np.concatenate([uav_arr['Y'], dep_arr['Y']])
        bbox = (np.min(all_x), np.min(all_y), np.max(all_x), np.max(all_y))
    
    # Generate standardized ticks based on the bounding box
    x_ticks = np.linspace(bbox[0], bbox[2], num=3)
    y_ticks = np.linspace(bbox[1], bbox[3], num=3)

    def apply_standard_ticks(ax, is_3d=False):
        # Custom formatter that rounds to one decimal place and shows only last 3 digits
        def format_tick(x, pos):
            # Round to one decimal place
            rounded = round(x, 1)
            # Get the last 3 digits plus the decimal place
            last_digits = rounded % 1000
            # Format with one decimal place
            return f"{last_digits:.1f}"
        
        formatter = FuncFormatter(format_tick)
        
        if is_3d:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            # Apply the custom formatter
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            # Force the offset text to be hidden
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        else:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            # Apply the custom formatter
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            # Force the offset text to be hidden
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.tick_params(axis='both', which='both', labelsize=6)
        
    # Modified helper functions that apply standard ticks internally
    
    def plot_pointcloud_with_ticks(ax, structured_pc, xlim, ylim, zlim, title):
        x = structured_pc['X']
        y = structured_pc['Y']
        z = structured_pc['Z']
        ax.scatter(x, y, z, s=0.1, c=z, cmap='viridis')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=True)
    
    def plot_dsm_with_ticks(ax, dsm, extent, title):
        im = ax.imshow(dsm, cmap='terrain', extent=extent, origin='lower')
        ax.set_title(title, fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_uavsar_with_ticks(ax, img_tensor, meta, band_names=None, bbox_overlay=None):
        try:
            raster, extent, _ = reconstruct_raster_from_tensor(img_tensor, meta)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            return
        
        # Default to displaying first band in grayscale if bands not specified
        if band_names is None or len(band_names) < 3:
            hh_band = raster[:, :, 0] if raster.ndim == 3 else raster
            vmin, vmax = np.percentile(hh_band, 2), np.percentile(hh_band, 98)
            ax.imshow(hh_band, cmap='gray', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
        else:
            # Get band indices from metadata
            bands_map = {}
            if 'bands' in meta:
                for i, band_name in enumerate(meta['bands']):
                    bands_map[band_name] = i
            else:
                # Default mapping if not specified in metadata
                possible_bands = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]
                for i, band_name in enumerate(possible_bands):
                    if i < (raster.shape[2] if raster.ndim == 3 else 1):
                        bands_map[band_name] = i
            
            # Create RGB composite
            rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.float32)
            
            # Extract bands for R, G, B channels
            for i, band_name in enumerate(band_names[:3]):
                if band_name in bands_map:
                    band_idx = bands_map[band_name]
                    if raster.ndim == 3 and band_idx < raster.shape[2]:
                        band_data = raster[:, :, band_idx]
                    elif band_idx == 0:
                        band_data = raster
                    else:
                        # Skip if band doesn't exist
                        continue
                    
                    # Normalize to 0-1 range for this channel
                    vmin, vmax = np.percentile(band_data, 2), np.percentile(band_data, 98)
                    normalized = np.clip((band_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    rgb[:, :, i] = normalized
            
            ax.imshow(rgb, extent=extent, origin='upper')
        
        # Draw the original bounding box on top (if provided)
        if bbox_overlay is not None:
            rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                             bbox_overlay[2]-bbox_overlay[0],
                             bbox_overlay[3]-bbox_overlay[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        date_str = meta.get('date', meta.get('datetime', ''))
        title = f"{date_str}"
        if band_names and len(band_names) >= 3:
            title += f"\n{band_names[0]}-{band_names[1]}-{band_names[2]}"
        ax.set_title(title, fontsize=10)
        
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
    
    def plot_naip_with_ticks(ax, img_tensor, meta, bbox_overlay=None):
        try:
            rgb, extent, _ = reconstruct_raster_from_tensor(img_tensor, meta)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            return
        
        ax.imshow(rgb.astype(np.uint8), extent=extent, origin='upper')
        
        # Draw the original bounding box on top (if provided)
        if bbox_overlay is not None:
            rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                             bbox_overlay[2]-bbox_overlay[0],
                             bbox_overlay[3]-bbox_overlay[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        date_str = meta.get('date', meta.get('datetime', ''))
        ax.set_title(f"{date_str}", fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
        
    # Compute common variables for point clouds and DSMs.
    uav_arr = tensor_to_structured_array(data_dict['uav_points'])
    dep_arr = tensor_to_structured_array(data_dict['dep_points'])
    all_x = np.concatenate([uav_arr['X'], dep_arr['X']])
    all_y = np.concatenate([uav_arr['Y'], dep_arr['Y']])
    all_z = np.concatenate([uav_arr['Z'], dep_arr['Z']])
    xlim = (np.min(all_x), np.max(all_x))
    ylim = (np.min(all_y), np.max(all_y))
    zlim = (np.min(all_z), np.max(all_z))
    
    # Downsampled UAV points
    num_uav = data_dict['uav_points'].shape[0]
    mask = data_dict.get('uav_downsample_mask', np.ones(num_uav, dtype=bool))
    down_uav = data_dict['uav_points'][mask]
    num_down = down_uav.shape[0]
    num_dep = data_dict['dep_points'].shape[0]
    
    # Figure for Point Clouds
    if "PointClouds" in modalities:
        # Calculate natural limits for point clouds
        pc_xlim = (np.min(all_x), np.max(all_x))
        pc_ylim = (np.min(all_y), np.max(all_y))
        pc_zlim = (np.min(all_z), np.max(all_z))
        
        fig_pc = plt.figure(figsize=(18, 4))
        ax_uav = fig_pc.add_subplot(131, projection='3d')
        ax_uav_ds = fig_pc.add_subplot(132, projection='3d')
        ax_dep = fig_pc.add_subplot(133, projection='3d')
        
        # Use our new helper functions with built-in tick formatting
        plot_pointcloud_with_ticks(ax_uav, uav_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"UAV LiDAR ({num_uav:,} pts)")
        
        down_arr = tensor_to_structured_array(down_uav)
        plot_pointcloud_with_ticks(ax_uav_ds, down_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"Downsampled UAV ({num_down:,} pts)")
        
        plot_pointcloud_with_ticks(ax_dep, dep_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"3DEP LiDAR ({num_dep:,} pts)")
        
        figures["PointClouds"] = fig_pc

    # Figure for DSMs
    if "DSMs" in modalities:
        dsm_extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        uav_np = data_dict['uav_points'].cpu().numpy() if data_dict['uav_points'].is_cuda else data_dict['uav_points'].numpy()
        dep_np = data_dict['dep_points'].cpu().numpy() if data_dict['dep_points'].is_cuda else data_dict['dep_points'].numpy()
        dsm_uav = compute_dsm(uav_np, bbox, resolution=0.3333333)
        dsm_dep = compute_dsm(dep_np, bbox, resolution=0.3333333)
        
        fig_dsm, (ax_dsm_uav, ax_dsm_dep) = plt.subplots(1, 2, figsize=(9, 2))
        plot_dsm_with_ticks(ax_dsm_uav, dsm_uav, dsm_extent, title="UAV DSM (max Z)")
        plot_dsm_with_ticks(ax_dsm_dep, dsm_dep, dsm_extent, title="3DEP DSM (max Z)")
        
        figures["DSMs"] = fig_dsm

    # Figure for UAVSAR images with 3 plots per row and band composites
    if "UAVSAR" in modalities and data_dict.get('uavsar_imgs', []):
        uavsar_imgs = data_dict['uavsar_imgs']
        n_uavsar = len(uavsar_imgs)
        
        # Change from 2 to 3 plots per row
        n_cols_uavsar = 3
        n_rows_uavsar = math.ceil(n_uavsar / n_cols_uavsar)
        
        fig_uavsar, axes_uavsar = plt.subplots(n_rows_uavsar, n_cols_uavsar, figsize=(10, 3 * n_rows_uavsar))
        
        # Handle single row case
        if n_rows_uavsar == 1:
            axes_uavsar = np.array([axes_uavsar])
        
        for i in range(n_uavsar):
            row = i // n_cols_uavsar
            col = i % n_cols_uavsar
            plot_uavsar_with_ticks(axes_uavsar[row, col], uavsar_imgs[i], 
                                 data_dict['uavsar_imgs_meta'][i],
                                 band_names=uavsar_bands, 
                                 bbox_overlay=bbox)
        
        # Hide unused axes
        for i in range(n_uavsar, n_rows_uavsar * n_cols_uavsar):
            row = i // n_cols_uavsar
            col = i % n_cols_uavsar
            axes_uavsar[row, col].axis("off")
        
        figures["UAVSAR"] = fig_uavsar

    # Figure for NAIP images with bbox overlay
    if "NAIP" in modalities and data_dict.get('naip_imgs', []):
        naip_imgs = data_dict['naip_imgs']
        n_naip = len(naip_imgs)
        
        # Change from 2 to 3 plots per row
        n_cols_naip = 3
        n_rows_naip = math.ceil(n_naip / n_cols_naip)
        
        # Adjust figure size for 3 columns
        fig_naip, axes_naip = plt.subplots(n_rows_naip, n_cols_naip, figsize=(10, 3 * n_rows_naip))
        
        # Ensure axes_naip is always a 2D array even with just one row
        if n_rows_naip == 1:
            axes_naip = np.array([axes_naip])
        
        for i in range(n_naip):
            # Calculate row and column indices for 3 columns
            row = i // n_cols_naip
            col = i % n_cols_naip
            plot_naip_with_ticks(axes_naip[row, col], naip_imgs[i], 
                               data_dict['naip_imgs_meta'][i], 
                               bbox_overlay=bbox)
        
        # Hide unused axes
        for i in range(n_naip, n_rows_naip * n_cols_naip):
            row = i // n_cols_naip
            col = i % n_cols_naip
            axes_naip[row, col].axis("off")
        
        figures["NAIP"] = fig_naip
    
    return figures

def load_and_visualize_from_file(file_path, tile_index=0, **kwargs):
    try:
        data = torch.load(file_path)
        if isinstance(data, list) and len(data) > 0:
            if tile_index < len(data):
                return visualize_tile_data(data[tile_index], tile_index=tile_index, **kwargs)
            else:
                print(f"Tile index {tile_index} is out of range; file contains {len(data)} tiles.")
                return None
        else:
            print("File does not contain a list of tile data or is empty.")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Example usage:
# figs = load_and_visualize_from_file("path/to/your/file.pt", tile_index=0, 
#                                     modalities=["PointClouds", "DSMs", "UAVSAR", "NAIP"],
#                                     uavsar_bands=["HHHH", "HVHV", "VVVV"])
# if figs is not None:
#     figs["PointClouds"].show()
#     figs["DSMs"].show()
#     figs["UAVSAR"].show()
#     figs["NAIP"].show()