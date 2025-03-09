#!/usr/bin/env python3
"""
H5 Dataset Loader Script

This script combines multiple H5 files containing tile data into a single PyTorch dataset.

Usage:
    python h5_loader.py --input_dir /path/to/h5/files --output_path /path/to/output.pt [options]

Options:
    --max_files N        Maximum number of files to process
    --max_tiles N        Maximum number of tiles to process per file
    --no_convert_float32 Don't convert float64 to float32 to save memory
    --verbose            Show detailed progress information
"""

import os
import sys
import h5py
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm

def combine_h5_files(input_dir, max_files=None, max_tiles_per_file=None, convert_float32=True, verbose=False):
    """
    Combines multiple H5 files into a single dataset.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing H5 files
    max_files : int, optional
        Maximum number of files to process
    max_tiles_per_file : int, optional
        Maximum number of tiles to process per file
    convert_float32 : bool
        Whether to convert float64 arrays to float32
    verbose : bool
        Whether to display detailed progress information
        
    Returns:
    --------
    list
        List of dictionaries, one per tile
    """
    combined_data = []
    
    # Get list of H5 files
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    if max_files is not None:
        h5_files = h5_files[:max_files]
    
    print(f"Processing {len(h5_files)} H5 files")
    
    # Process each file
    for file_idx, filename in enumerate(h5_files):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing file {file_idx+1}/{len(h5_files)}: {filename}")
        
        try:
            start_time = time.time()
            
            with h5py.File(filepath, 'r') as h5_file:
                # Find all tile groups
                tile_keys = [k for k in h5_file.keys() if k.startswith('tile_')]
                
                if max_tiles_per_file is not None:
                    tile_keys = tile_keys[:max_tiles_per_file]
                
                print(f"  Found {len(tile_keys)} tiles in file")
                
                # Process each tile
                for tile_idx, tile_key in enumerate(tqdm(tile_keys, desc=f"Tiles in {filename}", disable=not verbose)):
                    tile_group = h5_file[tile_key]
                    # Initialize tile dict with attributes
                    tile_dict = {}
                    
                    # Add basic attributes
                    for key, value in tile_group.attrs.items():
                        tile_dict[key] = value
                    
                    # Add tile identification
                    tile_dict['tile_id'] = tile_key
                    tile_dict['source_file'] = filename
                    
                    # Add top-level datasets directly
                    for key in tile_group.keys():
                        if isinstance(tile_group[key], h5py.Dataset):
                            data = tile_group[key][...]
                            if isinstance(data, np.ndarray):
                                if convert_float32 and data.dtype == np.float64:
                                    data = data.astype(np.float32)
                                tile_dict[key] = torch.from_numpy(data)
                            else:
                                tile_dict[key] = data
                    
                    # Process point cloud data - extract directly to top level
                    if 'point_clouds' in tile_group:
                        pc_group = tile_group['point_clouds']
                        for key in pc_group.keys():
                            data = pc_group[key][...]
                            if convert_float32 and data.dtype == np.float64:
                                data = data.astype(np.float32)
                            tile_dict[key] = torch.from_numpy(data)
                    
                    # Process downsample masks
                    if 'downsample_masks' in tile_group:
                        masks_group = tile_group['downsample_masks']
                        masks = []
                        for mask_key in sorted(masks_group.keys()):
                            data = masks_group[mask_key][...]
                            if convert_float32 and data.dtype == np.float64:
                                data = data.astype(np.float32)
                            masks.append(torch.from_numpy(data))
                        
                        tile_dict['downsample_masks'] = masks
                        # Set first mask as default if available
                        if masks:
                            tile_dict['uav_downsample_mask'] = masks[0]
                    
                    # Process NAIP imagery if present
                    if 'naip' in tile_group:
                        naip_group = tile_group['naip']
                        
                        # Add attributes
                        for key, value in naip_group.attrs.items():
                            tile_dict[f'naip_{key}'] = value
                        
                        # Process images
                        if 'images' in naip_group:
                            naip_imgs = []
                            naip_imgs_meta = []
                            
                            imgs_group = naip_group['images']
                            meta_group = naip_group['metadata'] if 'metadata' in naip_group else None
                            
                            for img_key in sorted(imgs_group.keys()):
                                # Load image
                                data = imgs_group[img_key][...]
                                naip_imgs.append(torch.from_numpy(data))
                                
                                # Load metadata if available
                                if meta_group and img_key in meta_group:
                                    img_meta = {}
                                    img_meta_group = meta_group[img_key]
                                    
                                    # Add attributes
                                    for k, v in img_meta_group.attrs.items():
                                        img_meta[k] = v
                                    
                                    # Add datasets
                                    for k in img_meta_group.keys():
                                        img_meta[k] = img_meta_group[k][...]
                                    
                                    naip_imgs_meta.append(img_meta)
                            
                            tile_dict['naip_imgs'] = naip_imgs
                            if naip_imgs_meta:
                                tile_dict['naip_imgs_meta'] = naip_imgs_meta
                        
                        # Process stacked images
                        if 'stacked_imgs' in naip_group:
                            data = naip_group['stacked_imgs'][...]
                            tile_dict['naip_imgs_array'] = torch.from_numpy(data)
                            
                            # Update flags
                            tile_dict['has_naip'] = True
                            tile_dict['has_imagery'] = True
                    
                    # Process UAVSAR imagery if present (similar approach as NAIP)
                    if 'uavsar' in tile_group:
                        uavsar_group = tile_group['uavsar']
                        
                        # Add attributes
                        for key, value in uavsar_group.attrs.items():
                            tile_dict[f'uavsar_{key}'] = value
                        
                        # Process images
                        if 'images' in uavsar_group:
                            uavsar_imgs = []
                            uavsar_imgs_meta = []
                            
                            imgs_group = uavsar_group['images']
                            meta_group = uavsar_group['metadata'] if 'metadata' in uavsar_group else None
                            
                            for img_key in sorted(imgs_group.keys()):
                                # Load image
                                data = imgs_group[img_key][...]
                                if convert_float32 and data.dtype == np.float64:
                                    data = data.astype(np.float32)
                                uavsar_imgs.append(torch.from_numpy(data))
                                
                                # Load metadata if available
                                if meta_group and img_key in meta_group:
                                    img_meta = {}
                                    img_meta_group = meta_group[img_key]
                                    
                                    # Add attributes
                                    for k, v in img_meta_group.attrs.items():
                                        img_meta[k] = v
                                    
                                    # Add datasets
                                    for k in img_meta_group.keys():
                                        img_meta[k] = img_meta_group[k][...]
                                    
                                    uavsar_imgs_meta.append(img_meta)
                            
                            tile_dict['uavsar_imgs'] = uavsar_imgs
                            if uavsar_imgs_meta:
                                tile_dict['uavsar_imgs_meta'] = uavsar_imgs_meta
                        
                        # Process stacked images
                        if 'stacked_imgs' in uavsar_group:
                            data = uavsar_group['stacked_imgs'][...]
                            if convert_float32 and data.dtype == np.float64:
                                data = data.astype(np.float32)
                            tile_dict['uavsar_imgs_array'] = torch.from_numpy(data)
                            
                            # Update flags
                            tile_dict['has_uavsar'] = True
                            tile_dict['has_imagery'] = True
                    
                    # Add to combined data
                    combined_data.append(tile_dict)
            
            elapsed = time.time() - start_time
            print(f"  Processed {len(tile_keys)} tiles in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    print(f"Total tiles processed: {len(combined_data)}")
    return combined_data

def estimate_size(data_list):
    """
    Estimate the size of the dataset in memory.
    
    Parameters:
    -----------
    data_list : list
        List of dictionaries containing tile data
        
    Returns:
    --------
    str
        Human-readable size estimate
    """
    total_bytes = 0
    for tile in data_list:
        for key, value in tile.items():
            if isinstance(value, torch.Tensor):
                total_bytes += value.element_size() * value.nelement()
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_bytes < 1024 or unit == 'TB':
            return f"{total_bytes:.2f} {unit}"
        total_bytes /= 1024

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Combine H5 tile files into a single PyTorch dataset')
    
    parser.add_argument('--input_dir', required=True, 
                        help='Directory containing H5 files')
    parser.add_argument('--output_path', required=True, 
                        help='Path to save the combined dataset')
    parser.add_argument('--max_files', type=int, default=None, 
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--max_tiles', type=int, default=None, 
                        help='Maximum number of tiles per file to process (default: all)')
    parser.add_argument('--no_convert_float32', action='store_true', 
                        help='Do not convert float64 to float32 (saves memory but uses more space)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Show detailed progress information')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Max files: {args.max_files if args.max_files is not None else 'All'}")
    print(f"Max tiles per file: {args.max_tiles if args.max_tiles is not None else 'All'}")
    print(f"Convert float64 to float32: {not args.no_convert_float32}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine H5 files
    combined_data = combine_h5_files(
        input_dir=args.input_dir,
        max_files=args.max_files,
        max_tiles_per_file=args.max_tiles,
        convert_float32=not args.no_convert_float32,
        verbose=args.verbose
    )
    
    # Estimate size
    size_estimate = estimate_size(combined_data)
    print(f"Estimated size of combined data: {size_estimate}")
    
    # Save the data
    print(f"Saving combined data to {args.output_path}...")
    try:
        torch.save(combined_data, args.output_path)
        print("Save completed successfully!")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()