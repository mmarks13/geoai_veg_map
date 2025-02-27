#!/usr/bin/env python3
"""
LiDAR Point Cloud Processing Script

This script processes LiDAR point cloud data (.las files) by:
1. Classifying points based on height above ground (HAG)
2. Creating Digital Terrain Models (DTM) and Digital Surface Models (DSM)

The script processes data from Sedgwick Reserve and Volcan Mountain locations.

Usage:
    python process_uav_lidar.py
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from src.utils.point_cloud_utils import process_and_classify_las
from src.utils.point_cloud_utils import create_dem

# Define input LAS file paths
sedgwick_las_paths = [
    'data/raw/uavlidar/Sedgwick/T01-T09_LiDAR_20230928_Pre_LAS.las',
    'data/raw/uavlidar/Sedgwick/T01-T09_LiDAR_20231024_Post_LAS.las',
    'data/raw/uavlidar/Sedgwick/T03-T13_LIDAR_20231025_Pre_LAS.las',
    'data/raw/uavlidar/Sedgwick/T06-T14_LIDAR_20231025_Pre_LAS.las',
    'data/raw/uavlidar/Sedgwick/T06-T14_LIDAR_20231204_Post_LAS.las',
    'data/raw/uavlidar/Sedgwick/TREX_LIDAR_20230630_Pre_LAS.las',
    'data/raw/uavlidar/Sedgwick/TREX_LIDAR_20231127_Post_LAS.las',
    'data/raw/uavlidar/Sedgwick/T10-T12_LIDAR_20231024_Post_LAS.las'
]

volcan_las_path = 'data/raw/uavlidar/full_volcan_mtn_las/VolcanMt_20231025_LAS.las'

# Define output directories
sedgwick_output_dir = 'data/processed/uavlidar/sedgwick'
volcan_output_dir = 'data/processed/uavlidar/volcan'
sedgwick_dem_output_dir = 'data/processed/dems/sedgwick'
volcan_dem_output_dir = 'data/processed/dems/volcan'


def process_and_classify_las_file(input_las, output_dir, min_hag, max_hag, filter_noise):
    """
    Process and classify a LAS file using PDAL.
    
    Args:
        input_las (str): Path to the input LAS file
        output_dir (str): Directory to save the processed LAS file
        min_hag (float): Minimum height above ground for classification
        max_hag (float): Maximum height above ground for classification
        filter_noise (bool): Whether to filter noise points
    """
    # Extract the filename from the input LAS path
    input_filename = os.path.basename(input_las)
    
    # Create the output LAS path
    output_las = os.path.join(output_dir, input_filename.replace('.las', '_classified.las'))
    
    # Create and execute the PDAL pipeline
    las_pipeline = process_and_classify_las(
        input_las=input_las,
        output_las=output_las,
        min_hag=min_hag,
        max_hag=max_hag,
        filter_noise=filter_noise
    )
    las_pipeline.execute()
    print(f"Processed and classified {input_las} to {output_las}")


def create_dems_for_las_files(las_files, output_dir, dem_type='dtm', resolution=1.0, window_size=None):
    """
    Create Digital Elevation Models (DEMs) for a list of LAS files.
    
    Args:
        las_files (list): List of paths to LAS files
        output_dir (str): Directory to save the DEMs
        dem_type (str): Type of DEM to create ('dtm' or 'dsm')
        resolution (float): Resolution of the DEM in meters
        window_size (int, optional): Window size for DEM creation
    """
    for las_file in las_files:
        # Extract the filename without extension
        base_filename = os.path.splitext(os.path.basename(las_file))[0]
        
        # Define the output TIF filename
        output_tif = os.path.join(output_dir, f"{base_filename}_{dem_type}_{resolution}m.tif")
        
        # Create the DEM
        print(f"Creating {dem_type.upper()} for {las_file}")
        create_dem(input_las=las_file, output_tif=output_tif, dem_type=dem_type, resolution=resolution, window_size=window_size).execute()
        print(f"Created {dem_type.upper()} at {output_tif}")


def main():
    """Main execution function for the LiDAR processing pipeline."""
    # Create output directories if they don't exist
    os.makedirs(sedgwick_output_dir, exist_ok=True)
    os.makedirs(volcan_output_dir, exist_ok=True)
    os.makedirs(sedgwick_dem_output_dir, exist_ok=True)
    os.makedirs(volcan_dem_output_dir, exist_ok=True)
    
    # # Step 1: Process and classify Sedgwick LAS files
    # print("\n--- Processing Sedgwick LAS files ---")
    # for las_path in sedgwick_las_paths:
    #     process_and_classify_las_file(
    #         input_las=las_path,
    #         output_dir=sedgwick_output_dir,
    #         min_hag=0,
    #         max_hag=25,
    #         filter_noise=True
    #     )
    
    # # Step 2: Process and classify Volcan Mountain LAS file
    # print("\n--- Processing Volcan Mountain LAS file ---")
    # process_and_classify_las_file(
    #     input_las=volcan_las_path,
    #     output_dir=volcan_output_dir,
    #     min_hag=0,
    #     max_hag=70,
    #     filter_noise=True
    # )
    
    # Step 3: Create DEMs for processed Sedgwick LAS files
    print("\n--- Creating DEMs for Sedgwick ---")
    processed_las_paths = [os.path.join(sedgwick_output_dir, f) for f in os.listdir(sedgwick_output_dir) if f.endswith('.las')]
    create_dems_for_las_files(processed_las_paths, sedgwick_dem_output_dir, dem_type='dtm', resolution=1.0)
    create_dems_for_las_files(processed_las_paths, sedgwick_dem_output_dir, dem_type='dsm', resolution=1.0)
    
    # Step 4: Create DEMs for processed Volcan Mountain LAS files
    print("\n--- Creating DEMs for Volcan Mountain ---")
    processed_las_paths = [os.path.join(volcan_output_dir, f) for f in os.listdir(volcan_output_dir) if f.endswith('.las')]
    create_dems_for_las_files(processed_las_paths, volcan_dem_output_dir, dem_type='dtm', resolution=1.0)
    create_dems_for_las_files(processed_las_paths, volcan_dem_output_dir, dem_type='dsm', resolution=1.0)
    
    print(f"\nLiDAR processing completed.")


# Run the script when executed directly
if __name__ == "__main__":
    main()