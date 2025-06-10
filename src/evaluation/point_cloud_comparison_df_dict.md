# Point Cloud Upsampling Comparison DataFrame Data Dictionary

## Overview

This dataset contains results from evaluating multiple terrain point cloud upsampling models designed to enhance low-resolution Digital Elevation Program (3DEP) point clouds from 2014-2016 to match high-resolution Unmanned Aerial Vehicle (UAV) LiDAR scans from 2023-2024. 

The primary goal is to synthetically enhance the resolution and accuracy of publicly available 3DEP terrain data through deep learning techniques. Four different model variants are compared:

1. **Baseline Model**: Uses only 3DEP point clouds with transformer-based upsampling (no additional data sources)
2. **NAIP Model**: Incorporates NAIP satellite imagery with 3DEP point clouds
3. **UAVSAR Model**: Incorporates UAVSAR radar data with 3DEP point clouds
4. **Combined Model**: Uses all data sources (3DEP points + NAIP imagery + UAVSAR radar)

Each model is evaluated using standard Chamfer Distance (CD) and InfoCD metrics against ground truth UAV LiDAR data. The dataset includes the original 3DEP input point clouds, ground truth UAV point clouds, and model predictions for each sample tile. Additionally, canopy height metrics are calculated by rasterizing point clouds to a 2m×2m grid and extracting the 95th percentile z-values.

The dataset features geospatial metadata for each tile including UTM Zone 11N polygon boundaries, along with source imagery (NAIP and UAVSAR) used for prediction. Each sample (row) represents a specific geographic tile with a unique identifier linking to its source LiDAR data.

This data can be used to assess model performance, analyze canopy height changes between time periods, and evaluate the contribution of different remote sensing data sources to point cloud upsampling quality.

## Metadata Columns

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| sample_idx | int | Index of the sample in the test dataset |
| tile_id | string | Unique identifier for the tile (e.g., 'tile_28798') |
| filename | string | Identifier for the source LiDAR tile from GeoJSON file (e.g., 'T01-T09_LiDAR_20230928_Pre_LAS') |

## Point Cloud Statistics

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| input_point_count | int | Number of points in the input (3DEP) point cloud |
| ground_truth_point_count | int | Number of points in the ground truth (UAV) point cloud |
| pred_point_count | int | Number of points in each predicted point cloud (same for all models) |

## Point Cloud Data (3D Coordinates)

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| input_points | numpy.ndarray (N×3) | Input (3DEP 2014-2016) point cloud coordinates (x,y,z) |
| ground_truth_points | numpy.ndarray (M×3) | Ground truth (UAV 2023-2024) point cloud coordinates (x,y,z) |
| combined_pred_points | numpy.ndarray (P×3) | Predicted point cloud from combined model (NAIP+UAVSAR) |
| naip_pred_points | numpy.ndarray (P×3) | Predicted point cloud from NAIP-only model |
| uavsar_pred_points | numpy.ndarray (P×3) | Predicted point cloud from UAVSAR-only model |
| baseline_pred_points | numpy.ndarray (P×3) | Predicted point cloud from baseline model (no imagery) |

## Distance Metrics

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| input_chamfer_distance | float | Chamfer distance between input and ground truth point clouds |
| input_infocd | float | InfoCD (Chamfer + contrastive) distance between input and ground truth |
| combined_chamfer_distance | float | Chamfer distance between combined model prediction and ground truth |
| combined_infocd | float | InfoCD distance between combined model prediction and ground truth |
| naip_chamfer_distance | float | Chamfer distance between NAIP model prediction and ground truth |
| naip_infocd | float | InfoCD distance between NAIP model prediction and ground truth |
| uavsar_chamfer_distance | float | Chamfer distance between UAVSAR model prediction and ground truth |
| uavsar_infocd | float | InfoCD distance between UAVSAR model prediction and ground truth |
| baseline_chamfer_distance | float | Chamfer distance between baseline model prediction and ground truth |
| baseline_infocd | float | InfoCD distance between baseline model prediction and ground truth |

## Canopy Height Metrics

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| input_mean_pct95_z | float | Mean of 95th percentile z-values across 2m×2m grid cells for input point cloud |
| gt_mean_pct95_z | float | Mean of 95th percentile z-values across 2m×2m grid cells for ground truth point cloud |
| net_canopy_height_change | float | Difference between ground truth and input mean 95th percentile z (gt - input) |

## Remote Sensing Imagery

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| img_bbox | numpy.ndarray (4,) | Bounding box coordinates for imagery [minx, miny, maxx, maxy] |
| naip_images | numpy.ndarray (n_images, 4, h, w) | NAIP satellite imagery with 4 spectral bands scaled 0-1 |
| naip_dates | list[str] | NAIP imagery acquisition dates (e.g., '2018-07-15') |
| uavsar_images | numpy.ndarray (n_images, l, k, h, w) | UAVSAR radar imagery with k bands and l look angles |
| uavsar_dates | list[str] | UAVSAR imagery acquisition dates |

## Spatial Data

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| geometry | shapely.geometry.Polygon | Polygon geometry representing the tile boundary in UTM Zone 11N (EPSG:32611) coordinate system |