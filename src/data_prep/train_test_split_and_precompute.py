"""
# Geospatial Tile Dataset Splitter

A tool to split PyTorch geospatial data tiles into training, validation, and test sets
based on spatial intersection with reference polygons, while enforcing data quality constraints.

This version supports the new flattened data structure where all data is at the top level.

## Example Usage

Basic usage with default parameters:
```
python split_train_test_val_tiles.py
```

Custom usage with specific parameters:
```
python split_train_test_val_tiles.py \
    --pt-file data/my_tiles.pt \
    --geojson-file data/test_regions.geojson \
    --output-dir data/split_output \
    --min-uav-points 6000 \
    --min-dep-points 300 \
    --min-uav-to-dep-ratio 1.5 \
    --test-val-ratio 0.6 \
    --random-seed 123
```

## Output
The script creates the following files in the output directory:
- training_tiles.pt: Tiles for training that don't intersect with test polygons
- test_tiles.pt: Tiles for testing that intersect with test polygons
- validation_tiles.pt: Subset of test tiles reserved for validation (if enabled)
- invalid_tiles.pt: Tiles that didn't meet the minimum validation criteria
- split_log.txt: Detailed log of the splitting process
"""


import torch
import geopandas as gpd
from shapely.geometry import box
import os
import random
import numpy as np
import argparse
import logging
import torch.serialization
torch.serialization.add_safe_globals({'dict': dict, 'list': list, 'tensor': torch.Tensor})
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import warnings
import sys
# Force immediate output flushing
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


def check_point_cloud_coverage(points, bbox, grid_size=1.0, min_points_per_cell=1, min_coverage_pct=80):
    """
    Check if a point cloud has sufficient coverage across a grid.

    Parameters:
    -----------
    points : torch.Tensor
        Point cloud tensor of shape (N, 3) containing (x, y, z) coordinates
    bbox : list or tuple
        Bounding box of the tile in format [minx, miny, maxx, maxy]
    grid_size : float
        Size of each grid cell in meters
    min_points_per_cell : int
        Minimum number of points required in a cell to consider it "covered"
    min_coverage_pct : float
        Minimum percentage of cells that must be covered for the point cloud to be considered complete

    Returns:
    --------
    bool, dict
        A tuple of (is_complete, stats). If the coverage is sufficient, is_complete is True.
        stats is a dictionary with coverage statistics.
    """
    # Extract x and y coordinates from the point cloud
    x = points[:, 0]
    y = points[:, 1]

    # Use the provided bbox instead of calculating from points
    x_min, y_min, x_max, y_max = bbox

    # Calculate the number of cells in each dimension
    # Add a small epsilon to ensure upper bound is included
    x_cells = int((x_max - x_min) / grid_size + 1e-6) # NECESSARY CHANGE: Removed + 1
    y_cells = int((y_max - y_min) / grid_size + 1e-6) # NECESSARY CHANGE: Removed + 1

    # Handle the case of empty or near-point point clouds
    if x_cells <= 1 or y_cells <= 1: # Note: Condition changed slightly by removing +1, x_cells/y_cells can now be 0 or 1. Original handles <=1.
        # Original structure returned this specific dict for <=1 case
        return False, {
            "total_cells": 0,
            "covered_cells": 0,
            "coverage_pct": 0,
            "avg_points_per_covered_cell": 0 # Match original return type (likely float)
        }

    # Compute which cell each point belongs to
    # This is a fast vectorized operation that assigns each point to a grid cell
    x_bin = ((x - x_min) / grid_size).long()
    y_bin = ((y - y_min) / grid_size).long()

    # NECESSARY CHANGE: Clamp bin indices to the valid range [0, cells-1]
    # Ensures bins calculated as 10 (e.g., float32) are mapped to 9 when x_cells is 10.
    x_bin = torch.clamp(x_bin, 0, x_cells - 1)
    y_bin = torch.clamp(y_bin, 0, y_cells - 1)
    # END NECESSARY CHANGE

    # Create a unique ID for each cell (row-major order)
    cell_ids = y_bin * x_cells + x_bin

    # Count the number of points in each cell using bincount
    # This is much faster than using a loop or groupby operations
    unique_cell_ids, cell_counts = torch.unique(cell_ids, return_counts=True)

    # Count cells with sufficient points
    cells_with_min_points = (cell_counts >= min_points_per_cell).sum().item()

    # Calculate total possible cells in the grid
    total_cells = x_cells * y_cells # Value is now 100 for 10x10m grid

    # Calculate coverage percentage
    coverage_pct = (cells_with_min_points / total_cells) * 100 # Value now calculated using total_cells=100

    # Check if coverage meets the threshold
    is_complete = coverage_pct >= min_coverage_pct

    # Gather statistics for reporting
    stats = {
        "total_cells": total_cells,
        "covered_cells": cells_with_min_points,
        "coverage_pct": coverage_pct,
        # Original calculation - potential edge case if cell_counts is empty remains as per original code
        "avg_points_per_covered_cell": cell_counts.float().mean().item()
    }

    return is_complete, stats


    
def validate_tile(tile, min_uav_points=10000, min_dep_points=500, 
                 min_uav_coverage_pct=80, min_points_per_cell=1,
                 min_uav_to_dep_ratio=None):
    """
    Validate that a tile meets the minimum criteria.
    This function supports both the original nested structure and the new flattened structure.
    
    Parameters:
    -----------
    tile : dict
        The tile dictionary to validate.
    min_uav_points : int
        Minimum number of UAV points required.
    min_dep_points : int
        Minimum number of DEP points required.
    min_uav_coverage_pct : float
        Minimum percentage of grid cells that must be covered in UAV point cloud.
    min_points_per_cell : int
        Minimum number of points required in a grid cell to consider it "covered".
    min_uav_to_dep_ratio : float or None
        If provided, validates that the number of UAV points is at least this many times 
        greater than the number of DEP points. If None, this check is skipped.
        
    Returns:
    --------
    bool, str
        A tuple of (is_valid, reason). If the tile is valid, is_valid is True and reason is None.
        If the tile is invalid, is_valid is False and reason is a string explaining why.
    """
    # Access UAV points based on whether the structure is flattened or nested
    uav_points = None
    
    # Try to access using the flattened structure first
    if 'uav_points' in tile and tile['uav_points'] is not None:
        uav_points = tile['uav_points']
    # Fall back to the original nested structure if needed
    elif 'point_clouds' in tile and tile['point_clouds'] is not None:
        if 'uav_points' in tile['point_clouds'] and tile['point_clouds']['uav_points'] is not None:
            uav_points = tile['point_clouds']['uav_points']
    
    # Check if we have UAV points
    if uav_points is None:
        return False, "Missing UAV points"
    
    # Check number of UAV points
    if uav_points.shape[0] < min_uav_points:
        return False, f"Insufficient UAV points: {uav_points.shape[0]} < {min_uav_points}"
    
    # Check if 'bbox' exists in the tile
    if 'bbox' not in tile or tile['bbox'] is None:
        return False, "Missing bounding box information"
    
    # Check UAV point cloud coverage using the tile's bbox
    is_complete, coverage_stats = check_point_cloud_coverage(
        uav_points, 
        bbox=tile['bbox'],
        grid_size=1.0,
        min_points_per_cell=min_points_per_cell,
        min_coverage_pct=min_uav_coverage_pct
    )
    
    if not is_complete:
        return False, f"Insufficient UAV point cloud coverage: {coverage_stats['coverage_pct']:.1f}% < {min_uav_coverage_pct}%"
    
    # Access DEP points based on whether the structure is flattened or nested
    dep_points = None
    
    # Try to access using the flattened structure first
    if 'dep_points' in tile and tile['dep_points'] is not None:
        dep_points = tile['dep_points']
    # Fall back to the original nested structure if needed
    elif 'point_clouds' in tile and tile['point_clouds'] is not None:
        if 'dep_points' in tile['point_clouds'] and tile['point_clouds']['dep_points'] is not None:
            dep_points = tile['point_clouds']['dep_points']
    
    # Check if we have DEP points
    if dep_points is None:
        return False, "Missing DEP points"
    
    # Check number of DEP points
    if dep_points.shape[0] < min_dep_points:
        return False, f"Insufficient DEP points: {dep_points.shape[0]} < {min_dep_points}"
    
    # Check the ratio of UAV points to DEP points, if a minimum ratio is specified
    if min_uav_to_dep_ratio is not None:
        uav_to_dep_ratio = uav_points.shape[0] / dep_points.shape[0]
        if uav_to_dep_ratio < min_uav_to_dep_ratio:
            return False, f"Insufficient UAV to DEP point ratio: {uav_to_dep_ratio:.2f} < {min_uav_to_dep_ratio}"
    
    return True, None

def split_dataset(
    pt_file_path='data/processed/model_data/combined_training_data.pt',
    geojson_file_path='data/processed/test_val_polygons.geojson',
    output_dir='data/processed/model_data/split',
    create_val_set=True,
    test_val_split_ratio=0.5,
    min_uav_points=10000,
    min_dep_points=500,
    min_uav_coverage_pct=80,
    min_points_per_cell=1,
    min_uav_to_dep_ratio=None,
    random_seed=42,
    remove_duplicates=True  # New parameter to control duplicate removal
):
    """
    Split a dataset of geospatial tiles into training, validation, and test sets 
    based on whether their bounding boxes intersect with polygons in a GeoJSON file.
    
    Parameters:
    -----------
    pt_file_path : str
        Path to the PyTorch file containing the tiles.
    geojson_file_path : str
        Path to the GeoJSON file containing the test/validation polygons.
    output_dir : str
        Directory where the split datasets will be saved.
    create_val_set : bool
        Whether to create a validation set from the test set.
    test_val_split_ratio : float
        Ratio for splitting the test set into test and validation sets (0.5 means 50% test, 50% validation).
    min_uav_points : int
        Minimum number of UAV points required for a valid tile.
    min_dep_points : int
        Minimum number of DEP points required for a valid tile.
    min_uav_coverage_pct : float
        Minimum percentage of grid cells that must be covered in UAV point cloud.
    min_points_per_cell : int
        Minimum points per grid cell to consider it covered.
    min_uav_to_dep_ratio : float or None
        If provided, validates that the number of UAV points is at least this many times 
        greater than the number of DEP points.
    random_seed : int
        Random seed for reproducibility.
    remove_duplicates : bool
        Whether to check for and remove duplicate tile_ids.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'split_log.txt'), mode='w')
        ]
    )
    
    # Load the PyTorch file containing the tiles
    logging.info("Loading the PyTorch file...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                                message="You are using `torch.load` with `weights_only=False`")
        tiles = torch.load(pt_file_path, map_location='cpu', mmap=True)
    logging.info(f"Loaded {len(tiles)} tiles from the PyTorch file.")
    
    # Check for and remove duplicate tile_ids if requested
    duplicate_count = 0
    if remove_duplicates:
        logging.info("Checking for duplicate tile_ids...")
        unique_tiles = []
        seen_tile_ids = set()
        duplicate_ids = {}
        
        for tile in tiles:
            # Make sure tile_id exists in the tile dictionary
            if 'tile_id' not in tile:
                logging.warning(f"Tile without 'tile_id' field found. Keeping it as is.")
                unique_tiles.append(tile)
                continue
                
            tile_id = tile['tile_id']
            
            if tile_id in seen_tile_ids:
                duplicate_count += 1
                if tile_id not in duplicate_ids:
                    duplicate_ids[tile_id] = 1
                else:
                    duplicate_ids[tile_id] += 1
            else:
                seen_tile_ids.add(tile_id)
                unique_tiles.append(tile)
        
        if duplicate_count > 0:
            logging.info(f"Found {duplicate_count} duplicate tiles with {len(duplicate_ids)} unique duplicate tile_ids.")
            for tile_id, count in duplicate_ids.items():
                logging.info(f"  - Tile ID {tile_id} appeared {count + 1} times")
            tiles = unique_tiles
            logging.info(f"Dataset now contains {len(tiles)} unique tiles after removing duplicates.")
    
    # Load the GeoJSON file containing the test/validation polygons
    logging.info("Loading the GeoJSON file...")
    test_val_polygons = gpd.read_file(geojson_file_path)
    logging.info(f"Loaded {len(test_val_polygons)} polygons from the GeoJSON file.")
    
    # Check if the GeoJSON CRS is different from UTM 11N and reproject if needed
    utm_11n_crs = 'EPSG:32611'  # UTM 11N
    if test_val_polygons.crs is None:
        logging.warning("GeoJSON CRS is not defined. Assuming it's in UTM 11N.")
        test_val_polygons.crs = utm_11n_crs
    elif test_val_polygons.crs.to_string() != utm_11n_crs:
        logging.info(f"Reprojecting from {test_val_polygons.crs} to {utm_11n_crs}")
        test_val_polygons = test_val_polygons.to_crs(utm_11n_crs)
    
    # Validate each tile
    logging.info("Validating tiles...")
    valid_tiles = []
    invalid_tiles = []
    invalid_reasons = {}
    coverage_stats = {}
    
    for i, tile in enumerate(tiles):
        is_valid, reason = validate_tile(
            tile, 
            min_uav_points=min_uav_points, 
            min_dep_points=min_dep_points,
            min_uav_coverage_pct=min_uav_coverage_pct,
            min_points_per_cell=min_points_per_cell,
            min_uav_to_dep_ratio=min_uav_to_dep_ratio
        )
        if is_valid:
            valid_tiles.append((i, tile))
        else:
            invalid_tiles.append((i, tile))
            if reason not in invalid_reasons:
                invalid_reasons[reason] = 0
            invalid_reasons[reason] += 1
    
    # Log validation results
    logging.info(f"Found {len(valid_tiles)} valid tiles and {len(invalid_tiles)} invalid tiles.")
    logging.info("Invalid tiles by reason:")
    for reason, count in invalid_reasons.items():
        logging.info(f"  - {reason}: {count} tiles")
    
    # Create GeoDataFrame from the valid tile bounding boxes
    logging.info("Creating GeoDataFrame from valid tile bounding boxes...")
    tile_geoms = []
    valid_indices = []
    
    for idx, tile in valid_tiles:
        minx, miny, maxx, maxy = tile['bbox']
        tile_geoms.append(box(minx, miny, maxx, maxy))
        valid_indices.append(idx)
    
    tile_gdf = gpd.GeoDataFrame(geometry=tile_geoms, crs=utm_11n_crs)
    tile_gdf['tile_idx'] = valid_indices
    
    # Perform a spatial join to find which tiles intersect with test/val polygons
    logging.info("Performing spatial join...")
    joined = gpd.sjoin(tile_gdf, test_val_polygons, predicate='intersects', how='left')
    
    # Get indices of tiles that intersect with test/val polygons
    test_indices = set(joined.loc[~joined['index_right'].isna(), 'tile_idx'])
    logging.info(f"Found {len(test_indices)} valid tiles that intersect with test/val polygons.")
    
    # Create training and test sets
    training_tiles = []
    test_tiles = []
    
    for idx, tile in valid_tiles:
        if idx in test_indices:
            test_tiles.append(tile)
        else:
            training_tiles.append(tile)
    
    # Print statistics
    logging.info("\nSplit statistics:")
    logging.info(f"Total tiles: {len(tiles)}")
    logging.info(f"Valid tiles: {len(valid_tiles)}")
    logging.info(f"Invalid tiles: {len(invalid_tiles)}")
    logging.info(f"Training tiles: {len(training_tiles)}")
    logging.info(f"Test tiles: {len(test_tiles)}")
    
    # Optionally split the test set further into test and validation sets
    val_tiles = []
    if create_val_set and test_tiles:
        # Shuffle the test tiles for a random split
        random.shuffle(test_tiles)
        
        # Calculate split index
        test_val_split_idx = int(len(test_tiles) * test_val_split_ratio)
        
        # Split the test tiles
        val_tiles = test_tiles[test_val_split_idx:]
        test_tiles = test_tiles[:test_val_split_idx]
        
        logging.info(f"Validation tiles: {len(val_tiles)}")
        logging.info(f"Final test tiles: {len(test_tiles)}")
    elif create_val_set and not test_tiles:
        logging.warning("No test tiles found. Cannot create validation set.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the training, validation, and test sets as separate PyTorch files
    logging.info("\nSaving the split datasets...")
    
    train_path = os.path.join(output_dir, 'training_tiles.pt')
    torch.save(training_tiles, train_path)
    logging.info(f"Saved {len(training_tiles)} training tiles to '{train_path}'")
    
    if val_tiles:
        val_path = os.path.join(output_dir, 'validation_tiles.pt')
        torch.save(val_tiles, val_path)
        logging.info(f"Saved {len(val_tiles)} validation tiles to '{val_path}'")
    
    if test_tiles:
        test_path = os.path.join(output_dir, 'test_tiles.pt')
        torch.save(test_tiles, test_path)
        logging.info(f"Saved {len(test_tiles)} test tiles to '{test_path}'")
    
    # Save invalid tiles for inspection if needed
    if invalid_tiles:
        invalid_path = os.path.join(output_dir, 'invalid_tiles.pt')
        invalid_tiles_list = [tile for _, tile in invalid_tiles]
        torch.save(invalid_tiles_list, invalid_path)
        logging.info(f"Saved {len(invalid_tiles_list)} invalid tiles to '{invalid_path}'")
        
    logging.info("\nDone!")
    
    # Return statistics
    return {
        'statistics': {
            'total_tiles': len(tiles),
            'valid_tiles': len(valid_tiles),
            'invalid_tiles': len(invalid_tiles),
            'duplicate_tiles_removed': duplicate_count if remove_duplicates else 0,
            'training_tiles': len(training_tiles),
            'test_tiles': len(test_tiles),
            'validation_tiles': len(val_tiles) if val_tiles else 0
        },
        'training_tiles': training_tiles,
        'test_tiles': test_tiles, 
        'validation_tiles': val_tiles if val_tiles else []
    }


import torch
from torch.utils.data import Dataset
from datetime import datetime
from typing import Dict, Any, List
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import numpy as np
import argparse
import sys

def get_torch_dtype(precision: int) -> torch.dtype:
    """Convert numerical precision value to PyTorch dtype."""
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    elif precision == 64:
        return torch.float64
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 16, 32, or 64.")

##########################################
# Utility Functions for Date Parsing
##########################################

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string using multiple methods.
    
    First, attempts to use datetime.fromisoformat (which supports ISO 8601 strings
    with timezone offsets, e.g., "2023-10-25T00:00:00+00:00").
    If that fails, it will try common strptime formats.
    
    Returns:
      A datetime object.
    """
    try:
        # This handles ISO 8601 with timezone offsets.
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    # Fallback: Try common formats.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} not recognized.")


def compute_relative_dates(dates: List[str], reference_date: datetime) -> torch.Tensor:
    """
    Compute relative dates (in days) for a list of date strings relative to the reference_date.
    Both the reference_date and each date in the list are converted to date objects (ignoring time).
    
    Returns:
      A tensor of shape [n_images, 1] with the relative day differences.
    """
    ref_date_only = reference_date.date()
    rel_dates = []
    for date_str in dates:
        d = parse_date(date_str)
        d_date_only = d.date()
        delta_days = (d_date_only - ref_date_only).days
        rel_dates.append(float(delta_days))
    return torch.tensor(rel_dates, dtype=torch.float32).unsqueeze(1)

##########################################
# Band Statistics Computation
##########################################

def compute_global_band_statistics(data_list, imagery_type='naip'):
    """
    Compute global mean and standard deviation for each band across the entire dataset.
    Handles NA, NaN, and Inf values by ignoring them in the computation.
    
    Inputs:
        data_list: List of sample dictionaries.
        imagery_type: Either 'naip' or 'uavsar'.
        
    Returns:
        means: Tensor of shape [n_bands] with mean values for each band.
        stds: Tensor of shape [n_bands] with standard deviation values for each band.
        invalid_counts: Dict with counts of invalid values per band.
    """
    # Check if imagery exists in the dataset
    img_key = f'{imagery_type}_imgs'
    if not any(img_key in sample for sample in data_list):
        print(f"No {imagery_type} imagery found in the dataset.")
        return None, None, None
    
    # First find number of bands from first valid sample
    n_bands = None
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            n_bands = sample[img_key].shape[1]
            break
    
    if n_bands is None:
        print(f"Could not determine number of bands for {imagery_type}.")
        return None, None, None
    
    # Initialize statistics
    sum_pixels = torch.zeros(n_bands, dtype=torch.float64)
    count_valid_pixels = torch.zeros(n_bands, dtype=torch.int64)
    invalid_counts = {
        'nan': torch.zeros(n_bands, dtype=torch.int64),
        'inf': torch.zeros(n_bands, dtype=torch.int64)
    }
    
    # First pass: compute mean and count invalid values
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            imgs = sample[img_key]  # Shape: [n_images, n_bands, height, width]
            
            # Process each band separately to handle invalid values
            for band_idx in range(n_bands):
                band_data = imgs[:, band_idx, :, :]  # Shape: [n_images, height, width]
                
                # Count NaN values
                nan_mask = torch.isnan(band_data)
                invalid_counts['nan'][band_idx] += nan_mask.sum().item()
                
                # Count Inf values
                inf_mask = torch.isinf(band_data)
                invalid_counts['inf'][band_idx] += inf_mask.sum().item()
                
                # Create combined mask of valid values
                valid_mask = ~(nan_mask | inf_mask)
                
                # Sum valid values and count them
                sum_pixels[band_idx] += band_data[valid_mask].sum().to(torch.float64)
                count_valid_pixels[band_idx] += valid_mask.sum().item()
    
    # Print invalid value statistics
    total_pixels = sum(count_valid_pixels).item()
    for band_idx in range(n_bands):
        total_invalid = invalid_counts['nan'][band_idx] + invalid_counts['inf'][band_idx]
        print(f"{imagery_type} Band {band_idx}:{count_valid_pixels[band_idx]} valid pixels, {total_invalid} invalid values ({invalid_counts['nan'][band_idx]} NaN, {invalid_counts['inf'][band_idx]} Inf)")
    
    # Check if we have enough valid pixels
    if torch.any(count_valid_pixels == 0):
        print(f"Some {imagery_type} bands have no valid pixels!")
        zero_bands = torch.where(count_valid_pixels == 0)[0].tolist()
        print(f"Bands with no valid pixels: {zero_bands}")
        return None, None, invalid_counts
    
    # Calculate means
    means = sum_pixels / count_valid_pixels
    
    # Second pass: compute standard deviation
    sum_squared_diff = torch.zeros_like(means)
    
    for sample in data_list:
        if img_key in sample and sample[img_key] is not None:
            imgs = sample[img_key]
            
            # Process each band separately
            for band_idx in range(n_bands):
                band_data = imgs[:, band_idx, :, :]
                
                # Create mask of valid values
                valid_mask = ~(torch.isnan(band_data) | torch.isinf(band_data))
                
                # Calculate squared differences for valid values only
                valid_data = band_data[valid_mask]
                if valid_data.numel() > 0:  # Check if we have any valid data
                    diff = valid_data - means[band_idx].to(valid_data.dtype)
                    sum_squared_diff[band_idx] += (diff * diff).sum().to(torch.float64)
    
    # Calculate standard deviations
    stds = torch.sqrt(sum_squared_diff / count_valid_pixels)
    
    # Ensure no zero standard deviations
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    
    return means, stds, invalid_counts

##########################################
# Imagery Preprocessing Functions
##########################################

def preprocess_naip_imagery(tile: Dict[str, Any], reference_date: datetime, 
                          naip_means=None, naip_stds=None, dtype: torch.dtype = torch.float16) -> Dict[str, Any]:
    """
    Preprocess NAIP imagery from the flattened data structure.
    Rescales uint8 values from [0, 255] to [0, 1] range.
    Handles NA, NaN, and Inf values by setting them to 0 after normalization.
    Converts normalized values to specified precision for memory efficiency.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'naip_imgs': Tensor of shape [n_images, 4, h, w] (4 spectral bands)
         - 'naip_dates': List of date strings
         - 'naip_ids': List of image IDs
         - 'naip_img_bbox': NAIP imagery bounding box [minx, miny, maxx, maxy]
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      naip_means: Not used for this normalization method.
      naip_stds: Not used for this normalization method.
      dtype: PyTorch dtype to use for the output tensors.
      
    Returns:
      A dictionary with:
         - 'images': The normalized NAIP imagery tensor in specified dtype format
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The NAIP imagery bounding box
    """
    # Get NAIP imagery tensor
    images = tile['naip_imgs'].clone()  # Tensor: [n_images, 4, h, w]
    
    # Identify invalid values before normalization
    invalid_mask = torch.isnan(images) | torch.isinf(images)
    
    # Simple rescaling from [0, 255] to [0, 1]
    images = images.float() / 255.0
    
    # Set any invalid values to 0
    images[invalid_mask] = 0.0
    
    # Convert to specified dtype for memory efficiency
    images = images.to(dtype)
    
    # Get dates and compute relative dates
    dates = tile['naip_dates']
    relative_dates = compute_relative_dates(dates, reference_date)
    
    return {
        'images': images,                  # Normalized image tensor: [n_images, 4, h, w] in specified dtype
        'ids': tile['naip_ids'],           # List of image IDs
        'dates': dates,
        'relative_dates': relative_dates,  # Tensor: [n_images, 1]
        'img_bbox': tile['naip_img_bbox'], # Bounding box
        'bands': tile['naip_bands']        # Band information
    }



def preprocess_uavsar_imagery(tile: Dict[str, Any], reference_date: datetime, 
                             uavsar_means=None, uavsar_stds=None, dtype: torch.dtype = torch.float32,
                             max_images_per_group: int = 8) -> Dict[str, Any]:
    """
    Preprocess UAVSAR imagery from the flattened data structure, handling variable numbers of images
    associated with distinct acquisition events (where events can span consecutive days).
    Only keeps acquisition events with two or more images.
    
    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'uavsar_imgs': Tensor of shape [n_images, n_bands, h, w]
         - 'uavsar_dates': List of date strings
         - 'uavsar_ids': List of image IDs
         - 'uavsar_img_bbox': UAVSAR imagery bounding box
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      uavsar_means: Optional tensor of shape [n_bands] with mean values for each band.
      uavsar_stds: Optional tensor of shape [n_bands] with standard deviation values for each band.
      dtype: PyTorch dtype to use for the output tensors.
      max_images_per_group: Maximum number of images to keep per acquisition event (G_max).
      
    Returns:
      A dictionary with:
         - 'images': Padded tensor of shape [T, G_max, n_bands, h, w]
         - 'attention_mask': Boolean mask of shape [T, G_max]
         - 'relative_dates': Tensor of shape [T, 1] with relative dates
         - 'dates': List of T representative dates
         - 'ids': List of lists containing image IDs
         - 'invalid_mask': Boolean mask of shape [T, G_max, n_bands, h, w]
         - 'img_bbox': The UAVSAR imagery bounding box
         - 'bands': Band information
    """
    # Get UAVSAR imagery tensor
    images = tile['uavsar_imgs'].clone()  # Tensor: [n_images, n_bands, h, w]
    dates_str = tile['uavsar_dates']
    ids = tile['uavsar_ids']
    
    # Step 1: Filter out images with all invalid pixels
    n_images = images.shape[0]
    valid_image_mask = torch.zeros(n_images, dtype=torch.bool, device=images.device)
    
    for img_idx in range(n_images):
        img = images[img_idx]  # Shape: [n_bands, h, w]
        valid_image_mask[img_idx] = not (torch.isnan(img) | torch.isinf(img)).all()
    
    invalid_count = n_images - valid_image_mask.sum().item()
    if invalid_count > 0:
        print(f"Removing {invalid_count} UAVSAR images with all invalid values.")
    
    if valid_image_mask.sum() == 0:
        print("WARNING: All UAVSAR images have invalid values only!")
        return None
    
    # Apply filtering
    images = images[valid_image_mask]
    dates_str = [date for i, date in enumerate(dates_str) if valid_image_mask[i]]
    ids = [id for i, id in enumerate(ids) if valid_image_mask[i]]
    
    # Step 2: Parse dates and sort chronologically
    parsed_dates = []
    for date_str in dates_str:
        parsed_date = parse_date(date_str).date()  # Convert to date object (ignore time)
        parsed_dates.append(parsed_date)
    
    # Create tuples of (image, date, id) and sort by date
    sorted_data = sorted(zip(images, parsed_dates, ids, dates_str), key=lambda x: x[1])
    
    # Unpack sorted data
    sorted_images = [item[0] for item in sorted_data]
    sorted_dates = [item[1] for item in sorted_data]
    sorted_ids = [item[2] for item in sorted_data]
    sorted_date_strs = [item[3] for item in sorted_data]
    
    # Step 3: Group by consecutive dates (acquisition events)
    groups = []
    current_group = {'images': [], 'dates': [], 'ids': [], 'date_strs': []}
    
    for i, (img, date, id_, date_str) in enumerate(zip(sorted_images, sorted_dates, sorted_ids, sorted_date_strs)):
        if i == 0:  # First image always starts a group
            current_group['images'].append(img)
            current_group['dates'].append(date)
            current_group['ids'].append(id_)
            current_group['date_strs'].append(date_str)
        else:
            # Check if current date is more than 1 day after the last date in the current group
            days_diff = (date - current_group['dates'][-1]).days
            if days_diff > 1:
                # Start a new group
                groups.append(current_group)
                current_group = {'images': [img], 'dates': [date], 'ids': [id_], 'date_strs': [date_str]}
            else:
                # Add to current group
                current_group['images'].append(img)
                current_group['dates'].append(date)
                current_group['ids'].append(id_)
                current_group['date_strs'].append(date_str)
    
    # Add the last group if it's not empty
    if current_group['images']:
        groups.append(current_group)
    
    # Step 4: Filter groups to keep only those with two or more images
    original_group_count = len(groups)
    groups = [group for group in groups if len(group['images']) >= 2]
    filtered_count = original_group_count - len(groups)
    
    # if filtered_count > 0:
    #     print(f"Filtered out {filtered_count} acquisition events with fewer than 2 images.")
    
    if len(groups) == 0:
        print("WARNING: No valid UAVSAR acquisition events with 2+ images found after filtering!")
        return None
    
    # Get tensor dimensions
    n_bands, h, w = images[0].shape
    T = len(groups)  # Number of acquisition events
    G_max = max_images_per_group  # Maximum images per event
    
    # Initialize padded tensors and masks
    device = images[0].device
    padded_images = torch.zeros((T, G_max, n_bands, h, w), dtype=images[0].dtype, device=device)
    attention_mask = torch.zeros((T, G_max), dtype=torch.bool, device=device)
    invalid_mask = torch.zeros((T, G_max, n_bands, h, w), dtype=torch.bool, device=device)
    
    # Lists for metadata
    group_dates = []  # Representative date for each group
    group_date_strs = []  # String representation of representative dates
    group_ids = []  # IDs for each group
    
    # Step 5: Populate padded tensors
    for t, group in enumerate(groups):
        actual_count = len(group['images'])
        count_to_pad = min(actual_count, G_max)
        
        if actual_count > G_max:
            print(f"WARNING: Group {t} has {actual_count} images, but only {G_max} will be used (truncating).")
        
        # Copy images into padded tensor
        for i in range(count_to_pad):
            padded_images[t, i] = group['images'][i]
            attention_mask[t, i] = True
            
            # Mark invalid pixels in the original images
            invalid_mask[t, i] = torch.isnan(group['images'][i]) | torch.isinf(group['images'][i])
        
        # Store representative date (first date in group) and IDs
        group_dates.append(group['dates'][0])
        group_date_strs.append(group['date_strs'][0])
        group_ids.append(group['ids'][:count_to_pad])  # Truncate if necessary
    
    # Step 6: Normalize the padded images
    if uavsar_means is not None and uavsar_stds is not None:
        # Create copies of the original tensors as we'll modify them
        padded_images_normalized = padded_images.clone()
        
        # Set invalid values to means temporarily for normalization
        for band_idx in range(n_bands):
            band_invalid_mask = invalid_mask[..., band_idx, :, :]
            if band_invalid_mask.any():
                padded_images_normalized[..., band_idx, :, :][band_invalid_mask] = uavsar_means[band_idx].to(padded_images.dtype)
        
        # Reshape means and stds for broadcasting: [1, 1, C, 1, 1]
        means = uavsar_means.view(1, 1, -1, 1, 1).to(padded_images.dtype)
        stds = uavsar_stds.view(1, 1, -1, 1, 1).to(padded_images.dtype)
        
        # Normalize
        padded_images_normalized = (padded_images_normalized - means) / stds
        
        # Find any new invalid values created during normalization
        new_invalid_mask = torch.isnan(padded_images_normalized) | torch.isinf(padded_images_normalized)
        padded_images_normalized[new_invalid_mask] = 0.0
        
        # Update the invalid mask
        invalid_mask = invalid_mask | new_invalid_mask
        
        # Zero out padded positions using the attention mask
        float_attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        padded_images_normalized = padded_images_normalized * float_attention_mask
        
        # Convert to specified dtype
        padded_images = padded_images_normalized.to(dtype)
    
    # Step 7: Compute relative dates for representative dates
    relative_dates = compute_relative_dates(group_date_strs, reference_date)
    
    return {
        'images': padded_images,                # Shape: [T, G_max, n_bands, h, w]
        'attention_mask': attention_mask,       # Shape: [T, G_max]
        'relative_dates': relative_dates,       # Shape: [T, 1]
        'dates': group_date_strs,               # List of T representative dates
        'ids': group_ids,                       # List of lists: [[ids for group 0], [ids for group 1], ...]
        'invalid_mask': invalid_mask,           # Shape: [T, G_max, n_bands, h, w]
        'img_bbox': tile['uavsar_img_bbox'],    # Bounding box
        'bands': tile['uavsar_bands']           # Band information
    }

##########################################
# Point Cloud and Attribute Normalization Functions
##########################################

def compute_point_attr_statistics(data_list):
    """
    Compute global mean and standard deviation for point attributes across the entire dataset.
    Handles NA, NaN, and Inf values by ignoring them in the computation.
    
    Inputs:
        data_list: List of sample dictionaries.
        
    Returns:
        means: Tensor of shape [n_attr] with mean values for each attribute.
        stds: Tensor of shape [n_attr] with standard deviation values for each attribute.
        invalid_counts: Dict with counts of invalid values per attribute.
    """
    # Check if dep_pnt_attr exists in the dataset
    if not any('dep_pnt_attr' in sample for sample in data_list):
        print("No point attributes found in the dataset.")
        return None, None, None
    
    # First find number of attributes from first valid sample
    n_attr = None
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            n_attr = sample['dep_pnt_attr'].shape[1]
            break
    
    if n_attr is None:
        print("Could not determine number of point attributes.")
        return None, None, None
    
    # Initialize statistics
    sum_attrs = torch.zeros(n_attr, dtype=torch.float64)
    count_valid_attrs = torch.zeros(n_attr, dtype=torch.int64)
    invalid_counts = {
        'nan': torch.zeros(n_attr, dtype=torch.int64),
        'inf': torch.zeros(n_attr, dtype=torch.int64)
    }
    
    # First pass: compute mean and count invalid values
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            attrs = sample['dep_pnt_attr']  # Shape: [n_points, n_attr]
            
            # Process each attribute separately to handle invalid values
            for attr_idx in range(n_attr):
                attr_data = attrs[:, attr_idx]  # Shape: [n_points]
                
                # Count NaN values
                nan_mask = torch.isnan(attr_data)
                invalid_counts['nan'][attr_idx] += nan_mask.sum().item()
                
                # Count Inf values
                inf_mask = torch.isinf(attr_data)
                invalid_counts['inf'][attr_idx] += inf_mask.sum().item()
                
                # Create combined mask of valid values
                valid_mask = ~(nan_mask | inf_mask)
                
                # Sum valid values and count them
                sum_attrs[attr_idx] += attr_data[valid_mask].sum().to(torch.float64)
                count_valid_attrs[attr_idx] += valid_mask.sum().item()
    
    # Print invalid value statistics
    total_attrs = sum(count_valid_attrs).item()
    for attr_idx in range(n_attr):
        total_invalid = invalid_counts['nan'][attr_idx] + invalid_counts['inf'][attr_idx]
        print(f"Point attribute {attr_idx}: {total_invalid} invalid values ({invalid_counts['nan'][attr_idx]} NaN, {invalid_counts['inf'][attr_idx]} Inf)")
    
    # Check if we have enough valid attributes
    if torch.any(count_valid_attrs == 0):
        print(f"Some point attributes have no valid values!")
        zero_attrs = torch.where(count_valid_attrs == 0)[0].tolist()
        print(f"Attributes with no valid values: {zero_attrs}")
        return None, None, invalid_counts
    
    # Calculate means
    means = sum_attrs / count_valid_attrs
    
    # Second pass: compute standard deviation
    sum_squared_diff = torch.zeros_like(means)
    
    for sample in data_list:
        if 'dep_pnt_attr' in sample and sample['dep_pnt_attr'] is not None:
            attrs = sample['dep_pnt_attr']
            
            # Process each attribute separately
            for attr_idx in range(n_attr):
                attr_data = attrs[:, attr_idx]
                
                # Create mask of valid values
                valid_mask = ~(torch.isnan(attr_data) | torch.isinf(attr_data))
                
                # Calculate squared differences for valid values only
                valid_data = attr_data[valid_mask]
                if valid_data.numel() > 0:  # Check if we have any valid data
                    diff = valid_data - means[attr_idx].to(valid_data.dtype)
                    sum_squared_diff[attr_idx] += (diff * diff).sum().to(torch.float64)
    
    # Calculate standard deviations
    stds = torch.sqrt(sum_squared_diff / count_valid_attrs)
    
    # Ensure no zero standard deviations
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    
    return means, stds, invalid_counts



# 1. Update normalize_point_clouds_with_bbox function to convert point clouds to specified dtype
def normalize_point_clouds_with_bbox(dep_points: torch.Tensor,
                                     uav_points: torch.Tensor,
                                     bbox: tuple,
                                     dtype: torch.dtype = torch.float16):
    """
    Normalizes 3DEP and UAV point clouds to a common coordinate system where:
    - x,y coordinates range from -5 to 5 (1 unit = 1 meter)
    - z coordinates are in meters, with minimum z value set to 0
    
    Inputs:
      dep_points: [N_dep, 3] tensor of 3DEP point coordinates.
      uav_points: [N_uav, 3] tensor of UAV point coordinates.
      bbox: Tuple (xmin, ymin, xmax, ymax) defining the spatial extent.
      dtype: PyTorch dtype to use for the output tensors.
      
    Returns:
      dep_points_norm: [N_dep, 3] normalized 3DEP points in specified dtype.
      uav_points_norm: [N_uav, 3] normalized UAV points in specified dtype.
      center: [1, 3] tensor representing the normalization center.
      scale: [1, 3] tensor with scale factors for x, y, z.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Combine both point clouds to find minimum z value
    z_min_dep = dep_points[:, 2].min()
    z_min_uav = uav_points[:, 2].min()
    z_min = min(z_min_dep, z_min_uav)

    # Calculate x,y centers from the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Create center tensor (for x,y we use bbox center, for z we use min value)
    center = torch.tensor([[center_x, center_y, z_min]], 
                         dtype=dep_points.dtype, device=dep_points.device)
    
    # Calculate x,y scales to map to [-5, 5] range
    # For a 10x10m box, scale factor = 10/10 = 1 (dividing by 1 doesn't change values)
    scale_x = (xmax - xmin) / 10
    scale_y = (ymax - ymin) / 10
    
    # For z, we use scale of 1 to keep values in meters
    scale_z = 1.0
    
    # Create scale tensor
    scale = torch.tensor([[scale_x, scale_y, scale_z]], 
                         dtype=dep_points.dtype, device=dep_points.device)
    
    # Apply normalization
    dep_points_norm = dep_points.clone()
    uav_points_norm = uav_points.clone()
    
    # Handle x,y coordinates (center at 0,0, scale to [-5,5] range)
    dep_points_norm[:, :2] -= center[:, :2]
    dep_points_norm[:, :2] /= scale[:, :2]
    uav_points_norm[:, :2] -= center[:, :2]
    uav_points_norm[:, :2] /= scale[:, :2]
    
    # Handle z coordinates (shift to make minimum = 0, keep in meters)
    dep_points_norm[:, 2] = dep_points[:, 2] - center[:, 2]  # Just subtract minimum z
    uav_points_norm[:, 2] = uav_points[:, 2] - center[:, 2]  # Just subtract minimum z
    
    # Convert to specified dtype for memory efficiency
    dep_points_norm = dep_points_norm.to(dtype)
    uav_points_norm = uav_points_norm.to(dtype)
    
    return dep_points_norm, uav_points_norm, center, scale


def voxel_downsample_masks(points, initial_voxel_size_cm, max_points_list):
    """
    Creates boolean masks for voxel grid downsampling for each value in max_points_list.
    For each maximum point count, the function adjusts the voxel size until the
    downsampled point cloud has no more than the specified number of points.
    
    Parameters:
        points (np.ndarray): (N, 3) array of point coordinates in meters.
        initial_voxel_size_cm (float): The starting voxel size in centimeters.
        max_points_list (List[int]): A list of maximum allowed number of points for the downsampled cloud.
    
    Returns:
        Tuple[List[np.ndarray], List[float]]:
            - masks: List of boolean masks, one for each maximum point threshold.
            - voxel_sizes: List of voxel sizes (in centimeters) used for each mask.
    """
    # Precompute values that remain constant for all iterations
    min_coords = np.min(points, axis=0)
    points_shifted = points - min_coords  # Precompute shifted points
    
    # Sort max_points_list in descending order to efficiently reuse voxel sizes
    sorted_indices = np.argsort(max_points_list)[::-1]
    sorted_max_points = [max_points_list[i] for i in sorted_indices]
    
    masks = [None] * len(max_points_list)
    voxel_sizes = [None] * len(max_points_list)
    
    # Track the previous voxel size for reuse
    prev_voxel_size_cm = initial_voxel_size_cm
    
    for i, max_points in enumerate(sorted_max_points):
        # Use binary search for optimal voxel size
        min_voxel_size = prev_voxel_size_cm  # Start from previous result
        max_voxel_size = prev_voxel_size_cm * 10  # Reasonable upper bound
        
        best_mask = None
        best_voxel_size = None
        best_count = float('inf')
        
        # Helper function to compute voxelized mask for a given size
        def compute_voxel_mask(voxel_size_cm):
            voxel_size = voxel_size_cm / 100.0  # Convert to meters
            
            # Compute voxel indices for each point
            voxel_indices = np.floor(points_shifted / voxel_size).astype(int)
            
            # Optimized hashing to avoid integer overflow
            # Use a tuple-based approach for smaller point clouds
            if len(points) < 1_000_000:
                # Dictionary-based approach for better memory performance
                voxel_dict = {}
                for i, (point, voxel_idx) in enumerate(zip(points, voxel_indices)):
                    voxel_key = tuple(voxel_idx)
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = []
                    voxel_dict[voxel_key].append(i)
                
                # Create mask
                mask = np.zeros(len(points), dtype=bool)
                for indices in voxel_dict.values():
                    if not indices:
                        continue
                    # Compute centroid of points in this voxel
                    voxel_points = points[indices]
                    centroid = np.mean(voxel_points, axis=0)
                    # Use squared distance (faster than np.linalg.norm)
                    distances = np.sum((voxel_points - centroid)**2, axis=1)
                    closest_idx = indices[np.argmin(distances)]
                    mask[closest_idx] = True
            else:
                # For very large point clouds, use a more memory-efficient approach
                # with vectorized operations where possible
                # Create a unique hash for each voxel using bit shifts
                hash_keys = (voxel_indices[:, 0].astype(np.int64) | 
                           (voxel_indices[:, 1].astype(np.int64) << 21) | 
                           (voxel_indices[:, 2].astype(np.int64) << 42))
                
                # Find unique voxels and group assignment
                unique_keys, inverse, counts = np.unique(hash_keys, return_inverse=True, return_counts=True)
                n_voxels = len(unique_keys)
                
                # Vectorized centroid calculation
                centroids = np.zeros((n_voxels, 3))
                for dim in range(3):
                    np.add.at(centroids[:, dim], inverse, points[:, dim])
                centroids /= counts[:, None]
                
                # Create mask - find closest point to each centroid
                mask = np.zeros(len(points), dtype=bool)
                for voxel_idx in range(n_voxels):
                    voxel_points_idx = np.where(inverse == voxel_idx)[0]
                    if len(voxel_points_idx) > 0:
                        # Squared Euclidean distance
                        distances = np.sum((points[voxel_points_idx] - centroids[voxel_idx])**2, axis=1)
                        closest_idx = voxel_points_idx[np.argmin(distances)]
                        mask[closest_idx] = True
            
            return mask, np.sum(mask)
        
        # Use binary search to find optimal voxel size
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while max_voxel_size - min_voxel_size > 0.5 and iterations < max_iterations:
            iterations += 1
            current_voxel_size_cm = (min_voxel_size + max_voxel_size) / 2
            
            mask, point_count = compute_voxel_mask(current_voxel_size_cm)
            
            if point_count <= max_points:
                # This voxel size works, try a smaller size
                if point_count > best_count or best_count > max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    best_count = point_count
                max_voxel_size = current_voxel_size_cm
            else:
                # Too many points, try a larger voxel size
                min_voxel_size = current_voxel_size_cm
        
        # If binary search didn't converge, use linear increase as fallback
        if best_mask is None or best_count > max_points:
            current_voxel_size_cm = min_voxel_size
            while True:
                mask, point_count = compute_voxel_mask(current_voxel_size_cm)
                if point_count <= max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    break
                # Adaptive increment to converge faster
                current_voxel_size_cm += 2 * (point_count / max_points)
        
        # Store the result in the original order
        original_idx = sorted_indices[i]
        masks[original_idx] = best_mask
        voxel_sizes[original_idx] = best_voxel_size
        
        # Save this voxel size for the next iteration
        prev_voxel_size_cm = best_voxel_size
    
    return masks, voxel_sizes



import numpy as np


##########################################
# voxel downsample masks
##########################################

def voxel_downsample_masks(points, initial_voxel_size_cm, max_points_list):
    """
    Creates boolean masks for voxel grid downsampling for each value in max_points_list.
    For each maximum point count, the function adjusts the voxel size until the
    downsampled point cloud has no more than the specified number of points.
    
    Parameters:
        points (np.ndarray): (N, 3) array of point coordinates in meters.
        initial_voxel_size_cm (float): The starting voxel size in centimeters.
        max_points_list (List[int]): A list of maximum allowed number of points for the downsampled cloud.
    
    Returns:
        Tuple[List[np.ndarray], List[float]]:
            - masks: List of boolean masks, one for each maximum point threshold.
            - voxel_sizes: List of voxel sizes (in centimeters) used for each mask.
    """
    # Precompute values that remain constant for all iterations
    min_coords = np.min(points, axis=0)
    points_shifted = points - min_coords  # Precompute shifted points
    
    # Sort max_points_list in descending order to efficiently reuse voxel sizes
    sorted_indices = np.argsort(max_points_list)[::-1]
    sorted_max_points = [max_points_list[i] for i in sorted_indices]
    
    masks = [None] * len(max_points_list)
    voxel_sizes = [None] * len(max_points_list)
    
    # Track the previous voxel size for reuse
    prev_voxel_size_cm = initial_voxel_size_cm
    
    for i, max_points in enumerate(sorted_max_points):
        # Use binary search for optimal voxel size
        min_voxel_size = prev_voxel_size_cm  # Start from previous result
        max_voxel_size = prev_voxel_size_cm * 10  # Reasonable upper bound
        
        best_mask = None
        best_voxel_size = None
        best_count = float('inf')
        
        # Helper function to compute voxelized mask for a given size
        def compute_voxel_mask(voxel_size_cm):
            voxel_size = voxel_size_cm / 100.0  # Convert to meters
            
            # Compute voxel indices for each point
            voxel_indices = np.floor(points_shifted / voxel_size).astype(int)
            
            # Optimized hashing to avoid integer overflow
            # Use a tuple-based approach for smaller point clouds
            if len(points) < 1_000_000:
                # Dictionary-based approach for better memory performance
                voxel_dict = {}
                for i, (point, voxel_idx) in enumerate(zip(points, voxel_indices)):
                    voxel_key = tuple(voxel_idx)
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = []
                    voxel_dict[voxel_key].append(i)
                
                # Create mask
                mask = np.zeros(len(points), dtype=bool)
                for indices in voxel_dict.values():
                    if not indices:
                        continue
                    # Compute centroid of points in this voxel
                    voxel_points = points[indices]
                    centroid = np.mean(voxel_points, axis=0)
                    # Use squared distance (faster than np.linalg.norm)
                    distances = np.sum((voxel_points - centroid)**2, axis=1)
                    closest_idx = indices[np.argmin(distances)]
                    mask[closest_idx] = True
            else:
                # For very large point clouds, use a more memory-efficient approach
                # with vectorized operations where possible
                # Create a unique hash for each voxel using bit shifts
                hash_keys = (voxel_indices[:, 0].astype(np.int64) | 
                           (voxel_indices[:, 1].astype(np.int64) << 21) | 
                           (voxel_indices[:, 2].astype(np.int64) << 42))
                
                # Find unique voxels and group assignment
                unique_keys, inverse, counts = np.unique(hash_keys, return_inverse=True, return_counts=True)
                n_voxels = len(unique_keys)
                
                # Vectorized centroid calculation
                centroids = np.zeros((n_voxels, 3))
                for dim in range(3):
                    np.add.at(centroids[:, dim], inverse, points[:, dim])
                centroids /= counts[:, None]
                
                # Create mask - find closest point to each centroid
                mask = np.zeros(len(points), dtype=bool)
                for voxel_idx in range(n_voxels):
                    voxel_points_idx = np.where(inverse == voxel_idx)[0]
                    if len(voxel_points_idx) > 0:
                        # Squared Euclidean distance
                        distances = np.sum((points[voxel_points_idx] - centroids[voxel_idx])**2, axis=1)
                        closest_idx = voxel_points_idx[np.argmin(distances)]
                        mask[closest_idx] = True
            
            return mask, np.sum(mask)
        
        # Use binary search to find optimal voxel size
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while max_voxel_size - min_voxel_size > 0.5 and iterations < max_iterations:
            iterations += 1
            current_voxel_size_cm = (min_voxel_size + max_voxel_size) / 2
            
            mask, point_count = compute_voxel_mask(current_voxel_size_cm)
            
            if point_count <= max_points:
                # This voxel size works, try a smaller size
                if point_count > best_count or best_count > max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    best_count = point_count
                max_voxel_size = current_voxel_size_cm
            else:
                # Too many points, try a larger voxel size
                min_voxel_size = current_voxel_size_cm
        
        # If binary search didn't converge, use linear increase as fallback
        if best_mask is None or best_count > max_points:
            current_voxel_size_cm = min_voxel_size
            while True:
                mask, point_count = compute_voxel_mask(current_voxel_size_cm)
                if point_count <= max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    break
                # Adaptive increment to converge faster
                current_voxel_size_cm += 2 * (point_count / max_points)
        
        # Store the result in the original order
        original_idx = sorted_indices[i]
        masks[original_idx] = best_mask
        voxel_sizes[original_idx] = best_voxel_size
        
        # Save this voxel size for the next iteration
        prev_voxel_size_cm = best_voxel_size
    
    return masks, voxel_sizes



import numpy as np



def anisotropic_voxel_downsample_masks(points, initial_voxel_size_cm, max_points_list, vertical_ratio=0.2):
    """
    Creates boolean masks for anisotropic voxel grid downsampling for each value in max_points_list.
    Uses different voxel sizes in horizontal (x,y) vs vertical (z) directions.
    
    Parameters:
        points (np.ndarray): (N, 3) array of point coordinates in meters.
        initial_voxel_size_cm (float): The starting voxel size in centimeters for horizontal dimensions.
        max_points_list (List[int]): A list of maximum allowed number of points for the downsampled cloud.
        vertical_ratio (float): Ratio of vertical to horizontal voxel size (default 0.2).
                               Lower values preserve more vertical detail.
    
    Returns:
        Tuple[List[np.ndarray], List[float]]:
            - masks: List of boolean masks, one for each maximum point threshold.
            - voxel_sizes: List of horizontal voxel sizes (in centimeters) used for each mask.
              Note: vertical voxel size = horizontal_size * vertical_ratio
    """
    # print("Using anisotropic voxel downsampling with vertical ratio:", vertical_ratio)
    # Precompute values that remain constant for all iterations
    min_coords = np.min(points, axis=0)
    points_shifted = points - min_coords  # Precompute shifted points
    
    # Sort max_points_list in descending order to efficiently reuse voxel sizes
    sorted_indices = np.argsort(max_points_list)[::-1]
    sorted_max_points = [max_points_list[i] for i in sorted_indices]
    
    masks = [None] * len(max_points_list)
    voxel_sizes = [None] * len(max_points_list)
    
    # Track the previous voxel size for reuse
    prev_voxel_size_cm = initial_voxel_size_cm
    
    # Cache computed masks to avoid redundant calculations
    voxel_size_cache = {}
    
    for i, max_points in enumerate(sorted_max_points):
        original_idx = sorted_indices[i]

        # # Check if the original point cloud already has fewer points than max_points
        # if len(points) <= max_points:
        #     # Return a mask of all True values
        #     masks[original_idx] = np.ones(len(points), dtype=bool)
        #     voxel_sizes[original_idx] = initial_voxel_size_cm
        #     continue  # Skip to the next threshold


        # Use binary search for optimal voxel size
        min_voxel_size = prev_voxel_size_cm  # Start from previous result
        max_voxel_size = prev_voxel_size_cm * 10  # Reasonable upper bound
        
        best_mask = None
        best_voxel_size = None
        best_count = float('inf')
        
        # Helper function to compute voxelized mask for a given size
        def compute_voxel_mask(horizontal_voxel_size_cm):
            # Check if we've already computed this voxel size
            if horizontal_voxel_size_cm in voxel_size_cache:
                return voxel_size_cache[horizontal_voxel_size_cm]
                
            # Convert to meters
            horizontal_voxel_size = horizontal_voxel_size_cm / 100.0
            vertical_voxel_size = horizontal_voxel_size * vertical_ratio
            
            # Create anisotropic voxel size array [x_size, y_size, z_size]
            voxel_sizes = np.array([horizontal_voxel_size, horizontal_voxel_size, vertical_voxel_size])
            
            # Compute voxel indices with different scales for each dimension
            voxel_indices = np.floor(points_shifted / voxel_sizes).astype(int)
            
            # Optimized hashing to avoid integer overflow
            # Use a tuple-based approach for smaller point clouds
            if len(points) < 1_000_000:
                # Dictionary-based approach for better memory performance
                voxel_dict = {}
                for idx, (point, voxel_idx) in enumerate(zip(points, voxel_indices)):
                    voxel_key = tuple(voxel_idx)
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = []
                    voxel_dict[voxel_key].append(idx)
                
                # Create mask
                mask = np.zeros(len(points), dtype=bool)
                for indices in voxel_dict.values():
                    if not indices:
                        continue
                    # Compute centroid of points in this voxel
                    voxel_points = points[indices]
                    centroid = np.mean(voxel_points, axis=0)
                    # Use squared distance (faster than np.linalg.norm)
                    distances = np.sum((voxel_points - centroid)**2, axis=1)
                    closest_idx = indices[np.argmin(distances)]
                    mask[closest_idx] = True
            else:
                # For very large point clouds, use a more memory-efficient approach
                # with vectorized operations where possible
                # Create a unique hash for each voxel using bit shifts
                hash_keys = (voxel_indices[:, 0].astype(np.int64) | 
                           (voxel_indices[:, 1].astype(np.int64) << 21) | 
                           (voxel_indices[:, 2].astype(np.int64) << 42))
                
                # Find unique voxels and group assignment
                unique_keys, inverse, counts = np.unique(hash_keys, return_inverse=True, return_counts=True)
                n_voxels = len(unique_keys)
                
                # Vectorized centroid calculation
                centroids = np.zeros((n_voxels, 3))
                for dim in range(3):
                    np.add.at(centroids[:, dim], inverse, points[:, dim])
                centroids /= counts[:, None]
                
                # Create mask - find closest point to each centroid
                mask = np.zeros(len(points), dtype=bool)
                for voxel_idx in range(n_voxels):
                    voxel_points_idx = np.where(inverse == voxel_idx)[0]
                    if len(voxel_points_idx) > 0:
                        # Squared Euclidean distance
                        distances = np.sum((points[voxel_points_idx] - centroids[voxel_idx])**2, axis=1)
                        closest_idx = voxel_points_idx[np.argmin(distances)]
                        mask[closest_idx] = True
            
            result = (mask, np.sum(mask))
            # Cache the result for this voxel size
            voxel_size_cache[horizontal_voxel_size_cm] = result
            return result
        
        # Use binary search to find optimal voxel size
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while max_voxel_size - min_voxel_size > 0.5 and iterations < max_iterations:
            iterations += 1
            current_voxel_size_cm = (min_voxel_size + max_voxel_size) / 2
            
            mask, point_count = compute_voxel_mask(current_voxel_size_cm)
            
            if point_count <= max_points:
                # This voxel size works, try a smaller size
                if point_count > best_count or best_count > max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    best_count = point_count
                max_voxel_size = current_voxel_size_cm
            else:
                # Too many points, try a larger voxel size
                min_voxel_size = current_voxel_size_cm
        
        # If binary search didn't converge, use adaptive linear increase as fallback
        if best_mask is None or best_count > max_points:
            current_voxel_size_cm = min_voxel_size
            
            # Start with small step sizes and gradually increase if needed
            step_factor = 0.5
            max_step = 5.0
            
            while True:
                mask, point_count = compute_voxel_mask(current_voxel_size_cm)
                
                if point_count <= max_points:
                    best_mask = mask
                    best_voxel_size = current_voxel_size_cm
                    break
                    
                # Adaptive increment to converge faster
                step_size = min(max_step, step_factor * (point_count / max_points))
                current_voxel_size_cm += step_size
                step_factor *= 1.2  # Gradually increase step factor
        
        # Store the result in the original order
        original_idx = sorted_indices[i]
        masks[original_idx] = best_mask
        voxel_sizes[original_idx] = best_voxel_size
        
        # Save this voxel size for the next iteration
        prev_voxel_size_cm = best_voxel_size
    
    return masks, voxel_sizes

# Define these helper functions at the module level (outside any other function)
def compute_anisotropic_voxel_mask(horizontal_voxel_size_cm, points, points_shifted, vertical_ratio):
    """Compute anisotropic voxel mask for a given horizontal voxel size."""
    # print("Computing anisotropic voxel mask with horizontal voxel size:", horizontal_voxel_size_cm)
    # Convert to meters
    horizontal_voxel_size = horizontal_voxel_size_cm / 100.0
    vertical_voxel_size = horizontal_voxel_size * vertical_ratio
    
    # Create anisotropic voxel size array [x_size, y_size, z_size]
    voxel_sizes = np.array([horizontal_voxel_size, horizontal_voxel_size, vertical_voxel_size])
    
    # Compute voxel indices with different scales for each dimension
    voxel_indices = np.floor(points_shifted / voxel_sizes).astype(int)
    
    # Dictionary-based approach for better memory performance
    voxel_dict = {}
    for idx, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(idx)
    
    # Create mask
    mask = np.zeros(len(points_shifted), dtype=bool)
    for indices in voxel_dict.values():
        if not indices:
            continue
        # Compute centroid of points in this voxel
        voxel_points = points[indices]
        centroid = np.mean(voxel_points, axis=0)
        # Use squared distance (faster than np.linalg.norm)
        distances = np.sum((voxel_points - centroid)**2, axis=1)
        closest_idx = indices[np.argmin(distances)]
        mask[closest_idx] = True
    
    return mask, np.sum(mask)


def process_one_max_points_aniso(args):
    """Process one max_points threshold for anisotropic voxel downsampling."""
    max_points, points, points_shifted, initial_voxel_size_cm, vertical_ratio = args
    
    # Use binary search to find optimal voxel size
    min_voxel_size = initial_voxel_size_cm
    max_voxel_size = initial_voxel_size_cm * 10  # Reasonable upper bound
    
    best_mask = None
    best_voxel_size = None
    best_count = float('inf')
    
    iterations = 0
    max_iterations = 10  # Prevent infinite loops
    
    while max_voxel_size - min_voxel_size > 0.5 and iterations < max_iterations:
        iterations += 1
        current_voxel_size_cm = (min_voxel_size + max_voxel_size) / 2
        
        mask, point_count = compute_anisotropic_voxel_mask(
            current_voxel_size_cm, points, points_shifted, vertical_ratio
        )
        # print(f"Current voxel size: {current_voxel_size_cm} cm, Point count: {point_count}, Best count: {best_count}")
        if point_count <= max_points:
            # This voxel size works, try a smaller size
            if point_count > best_count or best_count > max_points:
                best_mask = mask
                best_voxel_size = current_voxel_size_cm
                best_count = point_count
            max_voxel_size = current_voxel_size_cm
        else:
            # Too many points, try a larger voxel size
            min_voxel_size = current_voxel_size_cm
    
    # print("Best mask found, point count:", best_count)
    # If binary search didn't converge, use linear increase as fallback
    if best_mask is None or best_count > max_points:
        current_voxel_size_cm = min_voxel_size
        # print(f"Using linear increase for voxel size {current_voxel_size_cm}")
        # Start with small step sizes and gradually increase if needed
        step_factor = 0.5
        max_step = 5.0
        
        while True:
            mask, point_count = compute_anisotropic_voxel_mask(
                current_voxel_size_cm, points, points_shifted, vertical_ratio
            )
            
            if point_count <= max_points:
                best_mask = mask
                best_voxel_size = current_voxel_size_cm
                break
                
            # Adaptive increment to converge faster
            step_size = min(max_step, step_factor * (point_count / max_points))
            current_voxel_size_cm += step_size
            step_factor *= 1.2  # Gradually increase step factor
    
    return max_points, best_mask, best_voxel_size

# Similar function for regular voxel downsampling
def compute_voxel_mask(voxel_size_cm, points, points_shifted):
    """Compute voxel mask for a given voxel size."""
    voxel_size = voxel_size_cm / 100.0  # Convert to meters
    
    # Compute voxel indices for each point
    voxel_indices = np.floor(points_shifted / voxel_size).astype(int)
    
    # Dictionary-based approach for better memory performance
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(i)
    
    # Create mask
    mask = np.zeros(len(points_shifted), dtype=bool)
    for indices in voxel_dict.values():
        if not indices:
            continue
        # Compute centroid of points in this voxel
        voxel_points = points[indices]
        centroid = np.mean(voxel_points, axis=0)
        # Use squared distance (faster than np.linalg.norm)
        distances = np.sum((voxel_points - centroid)**2, axis=1)
        closest_idx = indices[np.argmin(distances)]
        mask[closest_idx] = True
    
    return mask, np.sum(mask)

def process_one_max_points_voxel(args):
    """Process one max_points threshold for regular voxel downsampling."""
    max_points, points, points_shifted, initial_voxel_size_cm = args
    
    # Use binary search to find optimal voxel size
    min_voxel_size = initial_voxel_size_cm
    max_voxel_size = initial_voxel_size_cm * 10  # Reasonable upper bound
    
    best_mask = None
    best_voxel_size = None
    best_count = float('inf')
    
    iterations = 0
    max_iterations = 10  # Prevent infinite loops
    
    while max_voxel_size - min_voxel_size > 0.5 and iterations < max_iterations:
        iterations += 1
        current_voxel_size_cm = (min_voxel_size + max_voxel_size) / 2
        
        mask, point_count = compute_voxel_mask(
            current_voxel_size_cm, points, points_shifted
        )
        
        if point_count <= max_points:
            # This voxel size works, try a smaller size
            if point_count > best_count or best_count > max_points:
                best_mask = mask
                best_voxel_size = current_voxel_size_cm
                best_count = point_count
            max_voxel_size = current_voxel_size_cm
        else:
            # Too many points, try a larger voxel size
            min_voxel_size = current_voxel_size_cm
    
    # If binary search didn't converge, use linear increase as fallback
    if best_mask is None or best_count > max_points:
        current_voxel_size_cm = min_voxel_size
        while True:
            mask, point_count = compute_voxel_mask(
                current_voxel_size_cm, points, points_shifted
            )
            if point_count <= max_points:
                best_mask = mask
                best_voxel_size = current_voxel_size_cm
                break
            # Adaptive increment to converge faster
            current_voxel_size_cm += 2 * (point_count / max_points)
    
    return max_points, best_mask, best_voxel_size

def parallel_anisotropic_voxel_downsample_masks(points, initial_voxel_size_cm, max_points_list, vertical_ratio=0.2):
    """
    Parallel version of anisotropic_voxel_downsample_masks using all available CPUs.
    """
    print()
    # Precompute values that remain constant for all iterations
    min_coords = np.min(points, axis=0)
    points_shifted = points - min_coords  # Precompute shifted points
    
    # Create result arrays with the same length as max_points_list
    masks = [None] * len(max_points_list)
    voxel_sizes = [None] * len(max_points_list)
    
    # Use ProcessPoolExecutor for parallelization
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel anisotropic voxel downsampling")
    
    # Prepare arguments for parallel processing - CRITICAL CHANGE HERE
    args_list = [
        (max_points, points, points_shifted, initial_voxel_size_cm, vertical_ratio)
        for max_points in max_points_list
    ]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_one_max_points_aniso, args_list))
        
    # Process results
    for i, (max_points, mask, voxel_size) in enumerate(results):
        # Find the corresponding index in the original list
        idx = max_points_list.index(max_points)
        masks[idx] = mask
        voxel_sizes[idx] = voxel_size
    
    return masks, voxel_sizes

def parallel_voxel_downsample_masks(points, initial_voxel_size_cm, max_points_list):
    """
    Parallel version of voxel_downsample_masks using all available CPUs.
    """
    # Precompute values that remain constant for all iterations
    min_coords = np.min(points, axis=0)
    points_shifted = points - min_coords  # Precompute shifted points
    
    # Create result arrays with the same length as max_points_list
    masks = [None] * len(max_points_list)
    voxel_sizes = [None] * len(max_points_list)
    
    # Use ProcessPoolExecutor for parallelization
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel voxel downsampling")
    
    # Prepare arguments for parallel processing - CRITICAL CHANGE HERE
    args_list = [
        (max_points, points, points_shifted, initial_voxel_size_cm)
        for max_points in max_points_list
    ]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_one_max_points_voxel, args_list))
        
    # Process results
    for i, (max_points, mask, voxel_size) in enumerate(results):
        # Find the corresponding index in the original list
        idx = max_points_list.index(max_points)
        masks[idx] = mask
        voxel_sizes[idx] = voxel_size
    
    return masks, voxel_sizes


##########################################
# Precomputation Function
##########################################

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_single_tile(sample, naip_means=None, naip_stds=None, uavsar_means=None, uavsar_stds=None,
                       point_attr_means=None, point_attr_stds=None, normalization_type='bbox',
                       max_dep_points=10000, precision=32, downsample_method='voxel',
                       initial_voxel_size_cm=5, max_uav_points=100000, vertical_ratio=0.2,
                       use_existing_masks=True):
    """Process a single tile - this function will be run in parallel."""
    # Get PyTorch dtype
    print(f"Processing tile {sample['tile_id']}")
    dtype = get_torch_dtype(precision)
    k_values = [16]
    
    # Extract data
    dep_points = sample['dep_points']
    dep_pnt_attr = sample['dep_pnt_attr']
    uav_points = sample['uav_points']
    bbox = sample['bbox']
    tile_id = sample.get('tile_id', 'unknown')
    
    # Limit the number of 3DEP points if max_dep_points is specified
    if max_dep_points is not None and len(dep_points) > max_dep_points:
        indices = torch.randperm(len(dep_points))[:max_dep_points]
        dep_points = dep_points[indices]
        if dep_pnt_attr is not None:
            dep_pnt_attr = dep_pnt_attr[indices]
    # print(f"Tile {tile_id}: 3DEP points: {len(dep_points)}, UAV points: {len(uav_points)}")
    # Downsample UAV points - always use SERIAL method
    if downsample_method != 'none':
        mask = None
        # print(f"Tile {tile_id}: Downsampling UAV points with {downsample_method} method")
        # Only check for existing masks if use_existing_masks is True
        if use_existing_masks:
            if 'uav_downsample_mask' in sample and sample['uav_downsample_mask'] is not None:
                mask = sample['uav_downsample_mask']
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                mask_source = "'uav_downsample_mask'"
            elif 'uav_downsample_masks' in sample and len(sample['uav_downsample_masks']) > 0:
                mask = sample['uav_downsample_masks'][0]
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                mask_source = "'uav_downsample_masks[0]'"
        
        # Generate a new mask if we don't have one - ALWAYS SERIAL PROCESSING HERE
        if mask is None:
            # Convert points to numpy for the voxel downsampling functions
            uav_points_np = uav_points.detach().cpu().numpy()
            
            # Apply different downsampling methods - always use serial version
            if downsample_method == 'voxel':
                masks, voxel_sizes = voxel_downsample_masks(
                    uav_points_np, initial_voxel_size_cm, [max_uav_points]
                )
                mask = masks[0]
                voxel_info = f"voxel size: {voxel_sizes[0]:.2f} cm"
            elif downsample_method == 'anisotropic_voxel':
                masks, voxel_sizes = anisotropic_voxel_downsample_masks(
                    uav_points_np, initial_voxel_size_cm, [max_uav_points], vertical_ratio
                )
                mask = masks[0]
                voxel_info = f"horizontal voxel size: {voxel_sizes[0]:.2f} cm, vertical size: {voxel_sizes[0] * vertical_ratio:.2f} cm"
            
            # Convert numpy mask to torch
            mask = torch.from_numpy(mask).to(uav_points.device)
            mask_source = f"new {downsample_method} mask"
        
        # Apply the mask
        if mask is not None:
            original_count = len(uav_points)
            uav_points = uav_points[mask]
            
            # Log output will be captured in the process output
            processing_info = f"Tile {tile_id}: Using {mask_source}, {voxel_info if 'voxel_info' in locals() else ''}, downsampled from {len(sample['uav_points'])} to {len(uav_points)} points ({len(uav_points)/original_count*100:.1f}%)"

    # print("Normalizing point clouds...")
    # Normalize point clouds
    dep_points_norm, uav_points_norm, center, scale = normalize_point_clouds_with_bbox(
        dep_points, uav_points, bbox, dtype=dtype
    )
    
    # Normalize point attributes if statistics are provided
    dep_points_attr_norm = None
    if dep_pnt_attr is not None and point_attr_means is not None and point_attr_stds is not None:
        # Create a copy to avoid modifying the original
        dep_points_attr_norm = dep_pnt_attr.clone()
        
        # Handle invalid values
        invalid_mask = torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)
        
        # Apply normalization
        if invalid_mask.any():
            # Set invalid values to means temporarily
            for attr_idx in range(dep_points_attr_norm.shape[1]):
                attr_invalid_mask = invalid_mask[:, attr_idx]
                if attr_invalid_mask.any():
                    dep_points_attr_norm[:, attr_idx][attr_invalid_mask] = point_attr_means[attr_idx].to(dep_points_attr_norm.dtype)
        
        # Normalize using global stats
        means = point_attr_means.to(dep_points_attr_norm.dtype)
        stds = point_attr_stds.to(dep_points_attr_norm.dtype)
        dep_points_attr_norm = (dep_points_attr_norm - means) / stds
        
        # Set any remaining invalid values to 0 (normalized mean)
        dep_points_attr_norm[torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)] = 0.0
        
        # Convert to torch.float16 for memory usage
        dep_points_attr_norm = dep_points_attr_norm.to(torch.float16)

    # print("calculating KNN edge indices...")
    # --- KNN Edge Indices Computation ---
    knn_edge_indices = {}
    for k in k_values:
        edge_index_k = knn_graph(dep_points_norm, k=k, loop=False)
        edge_index_k = to_undirected(edge_index_k, num_nodes=dep_points_norm.size(0))
        knn_edge_indices[k] = edge_index_k
    
    # Filter z values greater than 95m for UAV points
    z_filter_mask = uav_points_norm[:, 2] <= 95
    if z_filter_mask.sum() < len(uav_points_norm):
        filter_info = f"Filtered {len(uav_points_norm) - z_filter_mask.sum()} UAV points with z > 95m"            
        if z_filter_mask.sum() < 0.8 * len(uav_points_norm):
            return None  # Skip this tile
        uav_points_norm = uav_points_norm[z_filter_mask]    

    # --- Imagery Preprocessing ---
    ref_date_str = sample['uav_meta']['datetime']
    ref_date = parse_date(ref_date_str)
    
    # print("Preprocessing imagery...")
    # Process NAIP and UAVSAR imagery if available
    naip_preprocessed = None
    uavsar_preprocessed = None
    
    if sample.get('has_naip', False) and 'naip_imgs' in sample:
        naip_preprocessed = preprocess_naip_imagery(sample, ref_date, naip_means, naip_stds, dtype=dtype)
        
    if sample.get('has_uavsar', False) and 'uavsar_imgs' in sample:
        uavsar_preprocessed = preprocess_uavsar_imagery(sample, ref_date, uavsar_means, uavsar_stds, dtype=dtype)
    
    # Create unified precomputed sample
    precomputed_sample = {
        'dep_points_norm': dep_points_norm,
        'uav_points_norm': uav_points_norm,
        'dep_points_attr_norm': dep_points_attr_norm,
        'center': center,
        'scale': scale,
        'knn_edge_indices': knn_edge_indices,
        'naip': naip_preprocessed,
        'uavsar': uavsar_preprocessed,
        'tile_id': sample.get('tile_id', None),
        'processing_info': processing_info if 'processing_info' in locals() else "",
        'filter_info': filter_info if 'filter_info' in locals() else ""
    }
    
    return precomputed_sample, tile_id


def precompute_dataset(data_list, naip_means=None, naip_stds=None, uavsar_means=None, uavsar_stds=None, 
                      point_attr_means=None, point_attr_stds=None, normalization_type: str = 'bbox',
                      max_dep_points=10000, precision: int = 32,
                      downsample_method='voxel', initial_voxel_size_cm=5, max_uav_points=100000,
                      vertical_ratio=0.2, use_existing_masks=True, use_parallel=False, num_workers=None):
    """
    Precompute all necessary features for each tile in the dataset in parallel.
    For downsampling within each tile, serial processing is used.
    
    Parameters:
        data_list: List of sample dictionaries.
        ...
        use_parallel: Whether to use parallel processing (backward compatibility)
        num_workers: Number of parallel workers. If None, uses all available cores.
    
    Returns:
        precomputed_data_list: List of processed dictionaries.
    """
    print(use_parallel)
    # Determine number of workers - support both parameter styles
    if num_workers is None:
        if use_parallel:
            num_workers = multiprocessing.cpu_count()-1  # Leave one core free
        else:
            num_workers = 1  # No parallelism
    
    print(f"Processing {len(data_list)} tiles in parallel using {num_workers} workers")
    
    # If no parallelism is requested, process serially
    if num_workers == 1:
        print("Using serial processing for tiles")
        precomputed_data_list = []
        for idx, sample in enumerate(data_list):
            try:
                result, tile_id = process_single_tile(
                    sample, naip_means, naip_stds, uavsar_means, uavsar_stds,
                    point_attr_means, point_attr_stds, normalization_type,
                    max_dep_points, precision, downsample_method,
                    initial_voxel_size_cm, max_uav_points, vertical_ratio,
                    use_existing_masks
                )
                if result is not None:
                    precomputed_data_list.append(result)
                    print(f"[{idx+1}/{len(data_list)}] {result['processing_info']}")
                    if result.get('filter_info'):
                        print(f"  {result['filter_info']}")
                else:
                    print(f"[{idx+1}/{len(data_list)}] Tile {tile_id}: Skipped due to filtering criteria")
            except Exception as e:
                print(f"[{idx+1}/{len(data_list)}] Error processing tile at index {idx}: {str(e)}")
        return precomputed_data_list
    
    # Prepare arguments for each tile
    tile_args = []
    for sample in data_list:
        args = (sample, naip_means, naip_stds, uavsar_means, uavsar_stds, 
                point_attr_means, point_attr_stds, normalization_type,
                max_dep_points, precision, downsample_method, 
                initial_voxel_size_cm, max_uav_points, vertical_ratio,
                use_existing_masks)
        tile_args.append(args)
    
    # Process tiles in parallel
    precomputed_data_list = []
    completed_count = 0
    # print(f"Using {num_workers} workers for parallel processing")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        # print("Submitting jobs for parallel processing...")
        future_to_idx = {executor.submit(process_single_tile, *args): i for i, args in enumerate(tile_args)}
        # print(f"Submitted {len(future_to_idx)} tasks for processing")
        # Process results as they complete
        for future in as_completed(future_to_idx):
            
            idx = future_to_idx[future]
            completed_count += 1
            
            try:
                result, tile_id = future.result()
                if result is not None:  # Only add valid results
                    precomputed_data_list.append(result)
                    # Print processing info with progress indicator
                    print(f"[{completed_count}/{len(data_list)}] {result['processing_info']}")
                    if result.get('filter_info'):
                        print(f"  {result['filter_info']}")
                else:
                    print(f"[{completed_count}/{len(data_list)}] Tile {tile_id}: Skipped due to filtering criteria")
            except Exception as e:
                print(f"[{completed_count}/{len(data_list)}] Error processing tile at index {idx}: {str(e)}")
    
    print(f"Completed processing. Successfully processed {len(precomputed_data_list)} tiles out of {len(data_list)}")
    return precomputed_data_list





def combined_pipeline():
    """Main function that combines split and precompute operations"""
    parser = argparse.ArgumentParser(description="Split dataset and precompute in a single operation.")
    
    # Add arguments from both scripts
    parser.add_argument('--pt-file', type=str, default='data/processed/model_data/combined_training_data.pt',
                        help='Path to the PyTorch file containing the tiles.')
    parser.add_argument('--geojson-file', type=str, default='data/processed/test_val_polygons.geojson',
                        help='Path to the GeoJSON file containing the test/validation polygons.')
    parser.add_argument('--output-dir', type=str, default='data/processed/model_data/split',
                        help='Directory where the split datasets will be saved.')
    parser.add_argument('--no-val-set', action='store_false', dest='create_val_set',
                        help='Do not create a validation set from the test set.')
    parser.add_argument('--test-val-ratio', type=float, default=0.5,
                        help='Ratio for splitting the test set into test and validation sets (0.5 means 50%% test, 50%% validation).')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--min-uav-points', type=int, default=10000,
                        help='Minimum number of UAV points required for a valid tile.')
    parser.add_argument('--min-dep-points', type=int, default=500,
                        help='Minimum number of DEP points required for a valid tile.')
    parser.add_argument('--min-coverage', type=float, default=100,
                        help='Minimum percentage of grid cells that must be covered in UAV point cloud.')
    parser.add_argument('--min-points-per-cell', type=int, default=10,
                        help='Minimum points per grid cell to consider it covered.') 
    parser.add_argument('--min-uav-to-dep-ratio', type=float, default=None,
                        help='If provided, validates that the number of UAV points is at least this many times greater than the number of DEP points.')
    parser.add_argument('--remove-duplicates', action='store_true', default=True,
                        help='Check for and remove duplicate tile_ids (default: True)')
    parser.add_argument('--keep-duplicates', action='store_false', dest='remove_duplicates',
                        help='Keep duplicate tile_ids instead of removing them')
    # Add precompute arguments
    parser.add_argument('--precision', type=int, choices=[16, 32, 64], default=32,
                       help='Numerical precision to use (16, 32, or 64 bit). Default: 32')
    parser.add_argument('--downsample-method', type=str, choices=['none', 'voxel', 'anisotropic_voxel'], 
                        default='anisotropic_voxel', help='Method to use for downsampling UAV points')
    parser.add_argument('--initial-voxel-size-cm', type=float, default=5.0,
                        help='Initial voxel size in centimeters for downsampling')
    parser.add_argument('--max-uav-points', type=int, default=40000,
                        help='Maximum number of UAV points to keep after downsampling')
    parser.add_argument('--vertical-ratio', type=float, default=0.2,
                        help='Ratio of vertical to horizontal voxel size (for anisotropic voxel downsampling)')
    parser.add_argument('--use-existing-masks', action='store_true', default=True,
                        help='Use existing downsampling masks if available (default: True)')
    parser.add_argument('--force-new-masks', action='store_false', dest='use_existing_masks',
                        help='Force generation of new downsampling masks even if existing ones are available')
    parser.add_argument('--parallel', action='store_true', default=True,
                      help='Use parallel processing for voxel downsampling (default: True)')
    parser.add_argument('--no-parallel', action='store_false', dest='parallel',
                      help='Disable parallel processing for voxel downsampling')       
    parser.add_argument('--skip-split', action='store_true', default=False,
                        help='Skip the splitting step and load pre-split data from disk')
    args = parser.parse_args()

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")


    # Either split the dataset or load pre-split data depending on the --skip-split flag
    if args.skip_split:
        print("Skipping split step and loading pre-split data from disk...")
        # Load training, testing, and validation tiles from disk
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, 
                            message="You are using `torch.load` with `weights_only=False`")
            try:
                training_path = os.path.join(args.output_dir, 'training_tiles.pt')
                training_tiles = torch.load(training_path, map_location='cpu')
                print(f"Loaded {len(training_tiles)} training tiles from {training_path}")
            except FileNotFoundError:
                print(f"Error: Training file not found at {training_path}")
                sys.exit(1)
                
            try:
                test_path = os.path.join(args.output_dir, 'test_tiles.pt')
                test_tiles = torch.load(test_path, map_location='cpu')
                print(f"Loaded {len(test_tiles)} test tiles from {test_path}")
            except FileNotFoundError:
                print(f"Error: Test file not found at {test_path}")
                sys.exit(1)
            
            validation_tiles = []
            if args.create_val_set:
                try:
                    validation_path = os.path.join(args.output_dir, 'validation_tiles.pt')
                    validation_tiles = torch.load(validation_path, map_location='cpu')
                    print(f"Loaded {len(validation_tiles)} validation tiles from {validation_path}")
                except FileNotFoundError:
                    print(f"Warning: Validation file not found at {validation_path}. Continuing without validation tiles.")
            
        # Create stats dictionary for reporting
        stats = {
            'total_tiles': len(training_tiles) + len(test_tiles) + len(validation_tiles),
            'training_tiles': len(training_tiles),
            'test_tiles': len(test_tiles),
            'validation_tiles': len(validation_tiles)
        }
    else:
        print("Step 1: Splitting dataset and checking for duplicates...")
        result = split_dataset(
            pt_file_path=args.pt_file,
            geojson_file_path=args.geojson_file,
            output_dir=args.output_dir,
            create_val_set=args.create_val_set,
            test_val_split_ratio=args.test_val_ratio,
            min_uav_points=args.min_uav_points,
            min_dep_points=args.min_dep_points,
            min_uav_coverage_pct=args.min_coverage,
            min_points_per_cell=args.min_points_per_cell,
            min_uav_to_dep_ratio=args.min_uav_to_dep_ratio,
            random_seed=args.random_seed,
            remove_duplicates=args.remove_duplicates 
        )
        
        # Extract tile data from the result
        training_tiles = result['training_tiles']
        test_tiles = result['test_tiles']
        validation_tiles = result['validation_tiles']
        stats = result['statistics']
    
    # Print split statistics
    print("\nSplit statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Step 2: Proceed with precomputation using the tiles already in memory
    print("\nStep 2: Precomputing datasets without reloading from disk...")
    
    # Combine all datasets for computing statistics
    all_tiles = training_tiles + test_tiles + validation_tiles
    
    # Compute global statistics
    print("Computing global statistics...")
    naip_means, naip_stds, naip_invalid_counts = compute_global_band_statistics(all_tiles, imagery_type='naip')
    uavsar_means, uavsar_stds, uavsar_invalid_counts = compute_global_band_statistics(all_tiles, imagery_type='uavsar')
    point_attr_means, point_attr_stds, point_attr_invalid_counts = compute_point_attr_statistics(all_tiles)
    
    print("NAIP band statistics:")
    if naip_means is not None and naip_stds is not None:
        for i, (mean, std) in enumerate(zip(naip_means, naip_stds)):
            print(f"  Band {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(naip_means)):
            nan_count = naip_invalid_counts['nan'][i].item() if naip_invalid_counts else 0
            inf_count = naip_invalid_counts['inf'][i].item() if naip_invalid_counts else 0
            print(f"    Band {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No NAIP imagery found.")
        
    print("UAVSAR band statistics:")
    if uavsar_means is not None and uavsar_stds is not None:
        for i, (mean, std) in enumerate(zip(uavsar_means, uavsar_stds)):
            print(f"  Band {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(uavsar_means)):
            nan_count = uavsar_invalid_counts['nan'][i].item() if uavsar_invalid_counts else 0
            inf_count = uavsar_invalid_counts['inf'][i].item() if uavsar_invalid_counts else 0
            print(f"    Band {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No UAVSAR imagery found.")
    
    print("Point attribute statistics:")
    if point_attr_means is not None and point_attr_stds is not None:
        for i, (mean, std) in enumerate(zip(point_attr_means, point_attr_stds)):
            print(f"  Attribute {i}: mean = {mean:.4f}, std = {std:.4f}")
        print("  Invalid value summary:")
        for i in range(len(point_attr_means)):
            nan_count = point_attr_invalid_counts['nan'][i].item() if point_attr_invalid_counts else 0
            inf_count = point_attr_invalid_counts['inf'][i].item() if point_attr_invalid_counts else 0
            print(f"    Attribute {i}: {nan_count} NaN values, {inf_count} Inf values")
    else:
        print("  No point attributes found.")
    
    # Get corresponding PyTorch dtype
    dtype = get_torch_dtype(args.precision)
    print(f"Using numerical precision: {args.precision}-bit ({dtype})")
    
    # Precompute all datasets
    print("Precomputing validation dataset...")
    precomputed_validation = precompute_dataset(
        validation_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox',
        precision=args.precision,
        downsample_method=args.downsample_method,
        initial_voxel_size_cm=args.initial_voxel_size_cm,
        max_uav_points=args.max_uav_points,
        vertical_ratio=args.vertical_ratio,
        use_existing_masks=args.use_existing_masks,
        use_parallel=args.parallel
    )
    
    print("Precomputing test dataset...")
    precomputed_test = precompute_dataset(
        test_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox',
        precision=args.precision,
        downsample_method=args.downsample_method,
        initial_voxel_size_cm=args.initial_voxel_size_cm,
        max_uav_points=args.max_uav_points,
        vertical_ratio=args.vertical_ratio,
        use_existing_masks=args.use_existing_masks,
        use_parallel=args.parallel
    )
    
    print("Precomputing training dataset...")
    precomputed_training = precompute_dataset(
        training_tiles, 
        naip_means, naip_stds, 
        uavsar_means, uavsar_stds, 
        point_attr_means, point_attr_stds, 
        normalization_type='bbox',
        precision=args.precision,
        downsample_method=args.downsample_method,
        initial_voxel_size_cm=args.initial_voxel_size_cm,
        max_uav_points=args.max_uav_points,
        vertical_ratio=args.vertical_ratio,
        use_existing_masks=args.use_existing_masks,
        use_parallel=args.parallel
    )
    
    # Save all results
    output_suffix = f"_{args.precision}bit"
    
    # Save the precomputed datasets
    torch.save(precomputed_training, f'{args.output_dir}/precomputed_training_tiles{output_suffix}.pt')
    torch.save(precomputed_validation, f'{args.output_dir}/precomputed_validation_tiles{output_suffix}.pt')
    torch.save(precomputed_test, f'{args.output_dir}/precomputed_test_tiles{output_suffix}.pt')
    
    # Save normalization statistics
    if naip_means is not None:
        torch.save({'means': naip_means, 'stds': naip_stds}, f'{args.output_dir}/naip_normalization_stats.pt')
    if uavsar_means is not None:
        torch.save({'means': uavsar_means, 'stds': uavsar_stds}, f'{args.output_dir}/uavsar_normalization_stats.pt')
    if point_attr_means is not None:
        torch.save({'means': point_attr_means, 'stds': point_attr_stds}, f'{args.output_dir}/point_attr_normalization_stats.pt')
    
    print(f"\nComplete pipeline finished successfully. All data processed and saved.")

if __name__ == "__main__":
    try:
        combined_pipeline()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

