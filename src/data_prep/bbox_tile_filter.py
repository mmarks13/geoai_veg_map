"""
Tile Filter by Bounding Box

This script filters precomputed point cloud tiles based on whether their centers
are within a specified bounding box, and saves the filtered tiles to new files.

Usage:
python tile_filter.py --data_files file1.pt file2.pt --bbox xmin ymin xmax ymax --suffix _filtered
"""

import torch
import os
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path


def extract_center_coordinates(tile, index=None):
    """
    Extract center coordinates from a tile with multiple fallback methods.
    
    Args:
        tile: The tile dictionary
        index: Optional tile index for debugging messages
        
    Returns:
        (center_x, center_y) tuple or None if center cannot be determined
    """
    try:
        # Method 1: Use 'center' key if available
        if 'center' in tile:
            center = tile['center']
            
            # Handle different center formats
            if isinstance(center, torch.Tensor):
                # Convert to float32 if center is float16
                if center.dtype == torch.float16:
                    center = center.to(torch.float32)
                
                if center.dim() > 1 and center.shape[0] == 1:
                    # Format is likely [1, 3] tensor
                    center_x = center[0, 0].item()
                    center_y = center[0, 1].item()
                else:
                    # Format might be a 1D tensor
                    center_x = center[0].item()
                    center_y = center[1].item()
            elif isinstance(center, (list, tuple)):
                center_x = float(center[0])
                center_y = float(center[1])
            else:
                if index is not None:
                    print(f"Warning: Tile {index} has unexpected center format: {type(center)}")
                return None
                
        # Method 2: Calculate from 'bbox' if available
        elif 'bbox' in tile:
            bbox = tile['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
        # Method 3: Calculate from point cloud data
        elif 'dep_points_norm' in tile and 'uav_points_norm' in tile:
            dep_points = tile['dep_points_norm']
            uav_points = tile['uav_points_norm']
            all_points = torch.cat([dep_points, uav_points], dim=0)
            min_coords = torch.min(all_points[:, :2], dim=0)[0]
            max_coords = torch.max(all_points[:, :2], dim=0)[0]
            center_x = (min_coords[0].item() + max_coords[0].item()) / 2
            center_y = (min_coords[1].item() + max_coords[1].item()) / 2
            
        else:
            if index is not None:
                print(f"Warning: Could not determine center for tile {index}")
            return None
            
        return (center_x, center_y)
        
    except Exception as e:
        if index is not None:
            print(f"Error extracting center for tile {index}: {str(e)}")
        return None


def is_point_in_bbox(point, bbox):
    """
    Check if a point is within a bounding box.
    
    Args:
        point: (x, y) tuple
        bbox: [xmin, ymin, xmax, ymax] list or tuple
        
    Returns:
        Boolean indicating if the point is within the bbox
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def filter_tiles_by_bbox(data_files, bbox, suffix="_filtered", debug=False):
    """
    Filter tiles from multiple data files based on whether their centers are within a bounding box.
    
    Args:
        data_files: List of input file paths
        bbox: [xmin, ymin, xmax, ymax] list or tuple
        suffix: Suffix to add to the filtered output files
        debug: Whether to print debug information
        
    Returns:
        List of output file paths
    """
    output_files = []
    
    for data_file in data_files:
        try:
            # Load data
            tiles = torch.load(data_file)
            original_count = len(tiles)
            
            if debug:
                print(f"Loaded {original_count} tiles from {data_file}")
            
            # Filter tiles
            filtered_tiles = []
            
            for i, tile in enumerate(tqdm(tiles, desc=f"Filtering {Path(data_file).name}")):
                center = extract_center_coordinates(tile, index=i if debug else None)
                
                if center is not None and is_point_in_bbox(center, bbox):
                    filtered_tiles.append(tile)
            
            # Create output file path
            file_path = Path(data_file)
            output_path = file_path.parent / f"{file_path.stem}{suffix}{file_path.suffix}"
            output_files.append(str(output_path))
            
            # Save filtered tiles
            torch.save(filtered_tiles, output_path)
            
            kept_count = len(filtered_tiles)
            print(f"Kept {kept_count}/{original_count} tiles ({kept_count/original_count:.1%}) from {data_file}")
            print(f"Saved filtered tiles to {output_path}")
            
        except Exception as e:
            print(f"Error processing {data_file}: {str(e)}")
            continue
    
    return output_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter tiles by bounding box")
    parser.add_argument("--data_files", nargs="+", required=True, help="Paths to input data files (.pt)")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, help="Bounding box: xmin ymin xmax ymax")
    parser.add_argument("--suffix", default="_filtered", help="Suffix to add to output files")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Input files: {args.data_files}")
    print(f"  Bounding box: {args.bbox}")
    print(f"  Output suffix: {args.suffix}")
    
    # Execute filtering
    output_files = filter_tiles_by_bbox(
        data_files=args.data_files,
        bbox=args.bbox,
        suffix=args.suffix,
        debug=args.debug
    )
    
    print("\nSummary:")
    print(f"Processed {len(args.data_files)} input files")
    print(f"Created {len(output_files)} output files")
    for output_file in output_files:
        print(f"  - {output_file}")


# Example usage as a module:
# from tile_filter import filter_tiles_by_bbox
# 
# data_files = [
#     "data/processed/model_data/precomputed_training_tiles.pt",
#     "data/processed/model_data/precomputed_validation_tiles.pt",
#     "data/processed/model_data/precomputed_test_tiles.pt",
# ]
# 
# bbox = [557000, 4082000, 560000, 4085000]  # xmin, ymin, xmax, ymax in EPSG:32611
# 
# output_files = filter_tiles_by_bbox(
#     data_files=data_files,
#     bbox=bbox,
#     suffix="_filtered"
# )