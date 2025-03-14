# Precomputed Data Structure

This document describes the structure of the precomputed data files:
- `precomputed_training_tiles.pt`
- `precomputed_validation_tiles.pt`
- `precomputed_test_tiles.pt`

## Overview

Each file contains a list of dictionaries, where each dictionary represents a precomputed "tile" with normalized point cloud data, grid indices, KNN edge indices, and preprocessed imagery data.

Each tile represents a 10x10m section of the earth's surface in EPSG:32611 coordinate system. These tiles serve as training samples for a PyTorch model designed to upsample 3DEP point clouds based on the UAV point cloud ground truth.

## Tile Structure

Each tile dictionary contains the following top-level keys:

| Key | Data Type | Shape | Description |
|-----|-----------|-------|-------------|
| `dep_points_norm` | Tensor | [N_dep, 3] | Normalized 3DEP (3D Elevation Program) points. Point densities vary. Typically 1000-5000 points, sometimes as high as 10000 |
| `uav_points_norm` | Tensor | [N_uav, 3] | Normalized UAV (Unmanned Aerial Vehicle) points (downsampled). Point densities vary, but each one is downsampled to 20000 points or fewer.  |
| `dep_points_attr` | Tensor | [N_dep, 3] | Attributes associated with 3DEP points. ['Intensity', 'ReturnNumber', 'NumberOfReturns'] |
| `uav_points_attr` | Tensor or None | [N_uav, 3] | Attributes associated with UAV points. ['Intensity', 'ReturnNumber', 'NumberOfReturns']  |
| `center` | Tensor | [1, 3] | Normalization center used for point cloud normalization |
| `scale` | Scalar Tensor | [] | Normalization scale factor |
| `dep_grid_indices` | Tensor | [N_dep] | Grid indices for 3DEP points (values 0 to grid_size²-1) |
| `uav_grid_indices` | Tensor | [N_uav] | Grid indices for UAV points (values 0 to grid_size²-1) |
| `grid_coords` | Tensor | [grid_size, grid_size, 2] | Normalized grid coordinates for positional encoding |
| `knn_edge_indices` | Dict | - | KNN edge indices for different k values |
| `naip` | Dict or None | - | Preprocessed NAIP imagery data |
| `uavsar` | Dict or None | - | Preprocessed UAVSAR imagery data |
| `tile_id` | String or None | - | Optional tile identifier |
| `bbox` | Tuple | [4] | Original bounding box [xmin, ymin, xmax, ymax] in EPSG:32611 |

### KNN Edge Indices

The `knn_edge_indices` dictionary maps k-values to edge indices:

| Key | Data Type | Shape | Description |
|-----|-----------|-------|-------------|
| k (integer) | Tensor | [2, E] | Edge indices for KNN graph with k neighbors |

Where k is one of: 10, 15, 20, 30, 40, 50, 60 and E is the number of edges.

### NAIP Dictionary

The `naip` dicionary contains the available NAIP imagery for a given tile between the 3DEP and UAV Lidar acquisition dates. It could contain anywhere from 2 to 6 images. The `naip` dictionary contains:

| Key | Data Type | Shape | Description |
|-----|-----------|-------|-------------|
| `images` | Tensor | [n_images, 4, h, w] | NAIP imagery tensor with 4 spectral bands. Resampled to 0.5m resolution so image is 40x40 height and width.  |
| `ids` | List[str] | [n_images] | List of NAIP image IDs |
| `dates` | List[str] | [n_images] | List of NAIP acquisition date strings |
| `relative_dates` | Tensor | [n_images, 1] | Relative days from UAV acquisition date |
| `img_bbox` | List or Tuple | [4] | NAIP imagery bounding box [minx, miny, maxx, maxy], 20x20m sharing centroid with main bbox |
| `bands` | List | - | NAIP band information |


### UAVSAR Dictionary

The `uavsar` dicionary contains the available UAVSAR imagery for a given tile between the 3DEP and UAV Lidar acquisition dates. It could contain anywhere from 4 to 30 images. The `uavsar` dictionary contains:

| Key | Data Type | Shape | Description |
|-----|-----------|-------|-------------|
| `images` | Tensor | [n_images, n_bands, h, w] | UAVSAR imagery tensor. Should be six 'bands' (i.e. polarizations) and resampled to a standard 4x4 pixels height and width.|
| `ids` | List[str] | [n_images] | List of UAVSAR image IDs |
| `dates` | List[str] | [n_images] | List of UAVSAR acquisition date strings |
| `relative_dates` | Tensor | [n_images, 1] | Relative days from UAV acquisition date |
| `img_bbox` | List or Tuple | [4] | UAVSAR imagery bounding box [minx, miny, maxx, maxy], 20x20m sharing centroid with main bbox |
| `bands` | List | - | UAVSAR band information |


## Notes

- **Point Normalization**: Points are normalized using a bounding box ('bbox') normalization where x and y coordinates are normalized based on the bounding box and z coordinates are normalized using data statistics.
- **Grid Indices**: Each point is assigned to a grid cell in a grid_size × grid_size grid (default grid_size=20).
- **Relative Dates**: Date differences are computed in days relative to the UAV LiDAR acquisition date.
- **KNN Graphs**: K-nearest neighbor graphs are computed for the 3DEP points for each k value and converted to undirected graphs.
- **Bounding Boxes**: The main tile bounding box is 10x10m, while NAIP and UAVSAR imagery use a 20x20m bounding box that shares the same centroid. Both are in EPSG:32611 coordinate system.