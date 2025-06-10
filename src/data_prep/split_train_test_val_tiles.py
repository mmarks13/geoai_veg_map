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

import warnings



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
        tiles = torch.load(pt_file_path)
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
        'total_tiles': len(tiles),
        'duplicate_tiles_removed': duplicate_count if remove_duplicates else 0,
        'valid_tiles': len(valid_tiles),
        'invalid_tiles': len(invalid_tiles),
        'invalid_reasons': invalid_reasons,
        'training_tiles': len(training_tiles),
        'test_tiles': len(test_tiles),
        'validation_tiles': len(val_tiles) if val_tiles else 0
    }

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split geospatial data tiles into training, validation, and test sets.")
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
        
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    split_dataset(
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