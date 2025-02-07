import torch
from concurrent.futures import ProcessPoolExecutor
import torch
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from pyproj import Transformer
import gc
import psutil
import os
import psutil
from datetime import datetime
import pystac
import json
import pdal
import numpy as np
from pystac_client import Client
import planetary_computer
from shapely.geometry import mapping, box
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pdal
import json
from shapely.geometry import box
from planetary_computer import sign_inplace
import matplotlib.pyplot as plt


import numpy as np
from numpy.lib import recfunctions as rfn

from scipy.interpolate import griddata

def aggregate_grid(points, resolution, agg_func, field='Z',
                   remove_outliers=False, outlier_threshold=3,
                   bounding_box=None,
                   interpolate=True, nan_value=0):
    """
    Aggregate a point cloud into a 2D grid using an arbitrary aggregation function.
    Optionally remove extreme outliers in each grid cell before aggregation and then
    interpolate (or fill) any missing (NaN) grid cells.
    
    Parameters
    ----------
    points : np.ndarray
        Structured numpy array with at least the fields 'X', 'Y', and the specified field.
    resolution : float
        Grid cell size.
    agg_func : function
        A function that takes a 1D numpy array and returns a scalar.
    field : str, optional
        Name of the field to aggregate (default is 'Z').
    remove_outliers : bool, optional
        Whether to remove extreme outliers before aggregation (default False).
    outlier_threshold : float, optional
        Threshold factor for outlier removal (default 3).
    bounding_box : list or tuple of float, optional
        Bounding box in the form [xmin, ymin, xmax, ymax] (default None). When provided,
        the grid will be constructed for this spatial extent and only points within the
        bounding box will be used for aggregation.
    interpolate : bool, optional
        If True, interpolate missing (NaN) grid cells (default True). If False, no interpolation
        is performed.
    nan_value : float, optional
        The value to fill in for NaN grid cells if interpolation is off (default 0). This value
        should be chosen to penalize missing data in the loss computation.
    
    Returns
    -------
    grid : np.ndarray
        2D numpy array of aggregated values. If interpolate=True, NaNs are filled by interpolation;
        otherwise, NaNs are replaced by default_value.
    x_min : float
        The minimum x coordinate of the grid.
    y_min : float
        The minimum y coordinate of the grid.
    resolution : float
        The grid cell size.
    """
    # Extract coordinate and value arrays.
    x = points['X']
    y = points['Y']
    values = points[field]

    # Apply bounding box if provided.
    if bounding_box is not None:
        x_min, y_min, x_max, y_max = bounding_box
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x = x[mask]
        y = y[mask]
        values = values[mask]
    else:
        x_min, y_min = x.min(), y.min()
        x_max, y_max = x.max(), y.max()

    # Determine number of columns and rows.
    ncols = int(np.ceil((x_max - x_min) / resolution))
    nrows = int(np.ceil((y_max - y_min) / resolution))

    # Compute grid cell indices.
    col_indices = np.minimum(((x - x_min) / resolution).astype(np.int32), ncols - 1)
    row_indices = np.minimum(((y - y_min) / resolution).astype(np.int32), nrows - 1)

    # Flatten the 2D cell indices into a 1D index.
    flat_indices = row_indices * ncols + col_indices

    # Sort the points by grid cell.
    order = np.argsort(flat_indices)
    flat_indices_sorted = flat_indices[order]
    values_sorted = values[order]

    # Group sorted points by unique grid cell.
    unique_bins, start_idx, counts = np.unique(flat_indices_sorted, return_index=True, return_counts=True)

    # Initialize a flat grid with NaN (for cells with no data).
    grid_flat = np.full(nrows * ncols, np.nan, dtype=values.dtype)

    # Aggregate values for each unique grid cell.
    for flat_ind, start, count in zip(unique_bins, start_idx, counts):
        cell_vals = values_sorted[start:start+count]
        if remove_outliers and cell_vals.size > 0:
            median_val = np.median(cell_vals)
            mad = np.median(np.abs(cell_vals - median_val))
            if mad > 0:
                mask = np.abs(cell_vals - median_val) <= outlier_threshold * mad
            else:
                mask = np.ones_like(cell_vals, dtype=bool)
            cell_vals = cell_vals[mask]
        grid_flat[flat_ind] = agg_func(cell_vals) if cell_vals.size > 0 else np.nan

    # Reshape the flat grid into a 2D array.
    grid = grid_flat.reshape(nrows, ncols)

    if interpolate:
        # --- Interpolate NaN values ---
        valid_mask = ~np.isnan(grid)
        if np.any(valid_mask):
            grid_x, grid_y = np.meshgrid(np.arange(ncols), np.arange(nrows))
            points_valid = np.column_stack((grid_x[valid_mask], grid_y[valid_mask]))
            values_valid = grid[valid_mask]
            points_nan = np.column_stack((grid_x[~valid_mask], grid_y[~valid_mask]))
            
            # Use nearest if points are degenerate.
            if points_valid.shape[0] < 4 or np.unique(points_valid, axis=0).shape[0] < 3:
                interpolated_values = griddata(points_valid, values_valid, points_nan, method='nearest')
            else:
                try:
                    interpolated_values = griddata(points_valid, values_valid, points_nan, method='linear')
                except Exception:
                    interpolated_values = griddata(points_valid, values_valid, points_nan, method='nearest')
                nan_linear = np.isnan(interpolated_values)
                if np.any(nan_linear):
                    interpolated_values[nan_linear] = griddata(points_valid, values_valid,
                                                               points_nan[nan_linear], method='nearest')
            grid[~valid_mask] = interpolated_values
    else:
        # If interpolation is off, replace any NaNs with the default nan_value.
        grid = np.nan_to_num(grid, nan=nan_value)

    return grid, x_min, y_min, resolution



# def aggregate_grid(points, resolution, agg_func, field='Z',
#                    remove_outliers=False, outlier_threshold=3,
#                    x_min=None, y_min=None, x_max=None, y_max=None):
#     """
#     Aggregate a point cloud into a 2D grid using an arbitrary aggregation function.
#     Optionally remove extreme outliers in each grid cell before aggregation and then
#     interpolate any missing (NaN) grid cells.

#     If x_min, y_min, x_max, and y_max are provided, these define the grid's boundaries.
#     """
#     # Extract coordinate and value arrays.
#     x = points['X']
#     y = points['Y']
#     values = points[field]

#     # Determine grid boundaries.
#     if x_min is None:
#         x_min = x.min()
#     if y_min is None:
#         y_min = y.min()
#     if x_max is None:
#         x_max = x.max()
#     if y_max is None:
#         y_max = y.max()

#     # Determine number of columns and rows.
#     ncols = int(np.ceil((x_max - x_min) / resolution))
#     nrows = int(np.ceil((y_max - y_min) / resolution))

#     # Compute grid cell indices.
#     col_indices = ((x - x_min) / resolution).astype(np.int32)
#     row_indices = ((y - y_min) / resolution).astype(np.int32)

#     # For points outside the boundaries, assign the boundary index.
#     col_indices = np.where(col_indices < 0, 0, col_indices)
#     col_indices = np.where(col_indices >= ncols, ncols - 1, col_indices)
#     row_indices = np.where(row_indices < 0, 0, row_indices)
#     row_indices = np.where(row_indices >= nrows, nrows - 1, row_indices)

#     # Flatten the 2D indices.
#     flat_indices = row_indices * ncols + col_indices

#     # Sort points by grid cell.
#     order = np.argsort(flat_indices)
#     flat_indices_sorted = flat_indices[order]
#     values_sorted = values[order]

#     # Group by unique grid cell.
#     unique_bins, start_idx, counts = np.unique(flat_indices_sorted, return_index=True, return_counts=True)

#     # Initialize grid.
#     grid_flat = np.full(nrows * ncols, np.nan, dtype=values.dtype)

#     # Aggregate values per cell.
#     for flat_ind, start, count in zip(unique_bins, start_idx, counts):
#         cell_vals = values_sorted[start:start+count]
#         if remove_outliers and cell_vals.size > 0:
#             median_val = np.median(cell_vals)
#             mad = np.median(np.abs(cell_vals - median_val))
#             if mad > 0:
#                 mask = np.abs(cell_vals - median_val) <= outlier_threshold * mad
#             else:
#                 mask = np.ones_like(cell_vals, dtype=bool)
#             cell_vals = cell_vals[mask]
#         grid_flat[flat_ind] = agg_func(cell_vals) if cell_vals.size > 0 else np.nan

#     # Reshape to 2D.
#     grid = grid_flat.reshape(nrows, ncols)

#     # --- Interpolate NaN values ---
#     valid_mask = ~np.isnan(grid)
#     if np.any(valid_mask):
#         grid_x, grid_y = np.meshgrid(np.arange(ncols), np.arange(nrows))
#         points_valid = np.column_stack((grid_x[valid_mask], grid_y[valid_mask]))
#         values_valid = grid[valid_mask]
#         points_nan = np.column_stack((grid_x[~valid_mask], grid_y[~valid_mask]))
        
#         # Check if valid points are degenerate (not full-dimensional).
#         if points_valid.shape[0] < 4 or np.unique(points_valid, axis=0).shape[0] < 3:
#             # Use nearest-neighbor if not enough unique points for linear interpolation.
#             interpolated_values = griddata(points_valid, values_valid, points_nan, method='nearest')
#         else:
#             try:
#                 interpolated_values = griddata(points_valid, values_valid, points_nan, method='linear')
#             except Exception as e:
#                 # Fall back to nearest if linear interpolation fails.
#                 interpolated_values = griddata(points_valid, values_valid, points_nan, method='nearest')
#             nan_linear = np.isnan(interpolated_values)
#             if np.any(nan_linear):
#                 interpolated_values[nan_linear] = griddata(points_valid, values_valid,
#                                                            points_nan[nan_linear], method='nearest')
#         grid[~valid_mask] = interpolated_values
#     # --- End interpolation ---
#     return grid, x_min, y_min, resolution






def assign_aggregated_values(points, grid, x_origin, y_origin, resolution, new_field='agg_val'):
    """
    For each point in the point cloud, compute its grid cell index and assign the corresponding
    aggregated value from the grid. A new field is appended to the point cloud with these values.
    
    Parameters
    ----------
    points : np.ndarray
        Structured numpy array with at least 'X' and 'Y' fields.
    grid : np.ndarray
        2D numpy array returned from aggregate_grid.
    x_origin : float
        Minimum X value (origin of the grid).
    y_origin : float
        Minimum Y value (origin of the grid).
    resolution : float
        Grid cell size.
    new_field : str, optional
        Name of the new field to add to the point cloud.
    
    Returns
    -------
    points_new : np.ndarray
        New structured array with the added field containing aggregated values.
    """
    # Compute grid indices for each point.
    col_indices = ((points['X'] - x_origin) / resolution).astype(np.int32)
    row_indices = ((points['Y'] - y_origin) / resolution).astype(np.int32)

    # Ensure indices fall within valid range.
    nrows, ncols = grid.shape
    col_indices = np.clip(col_indices, 0, ncols - 1)
    row_indices = np.clip(row_indices, 0, nrows - 1)

    # Retrieve the aggregated value from the grid for each point.
    aggregated_values = grid[row_indices, col_indices]

    # Append the new field to the structured array.
    points_new = rfn.append_fields(points, new_field, aggregated_values, usemask=False)
    return points_new


def scale_grid_values(values, out_min, out_max, in_min, in_max):
    """
    Scale the input grid values linearly between out_min and out_max based on the input range.

    Parameters
    ----------
    values : np.ndarray
        Array of grid values to be scaled.
    out_min : float
        The minimum value of the scaled output (e.g., 0.1).
    out_max : float
        The maximum value of the scaled output (e.g., 1).
    in_min : float
        The lower bound of the input range. Values at or below this will be set to out_min.
    in_max : float
        The upper bound of the input range. Values at or above this will be set to out_max.

    Returns
    -------
    scaled_values : np.ndarray
        Array of scaled values.
    """
    # Convert input to a numpy array if it isn't one already.
    values = np.asarray(values)

    # Compute the scaling factor
    scale = (out_max - out_min) / (in_max - in_min) if in_max != in_min else 0.0

    # Apply linear scaling: For a value v in the range in_min to in_max,
    # the corresponding value is:
    #   out_min + (v - in_min) * scale
    scaled = out_min + (values - in_min) * scale

    # Clip the results so that values below out_min become out_min and
    # values above out_max become out_max.
    scaled = np.clip(scaled, out_min, out_max)

    return scaled




def create_pointcloud_stack(bbox, start_date, end_date, stac_source, collection, threads=4,
                            bbox_crs="EPSG:4326", target_crs=None):
    """
    Create a stack of point clouds from STAC items filtered by collection, date, and bounding box.
    In addition, compute an approximate DSM (max height) and DTM (min height) grid and derive the
    approximate canopy height. Two new fields ('approx_ch_raw' and 'approx_ch_scaled') are appended to
    the point cloud. The function returns the updated point cloud stack along with the DSM and DTM grids.
    
    Parameters
    ----------
    bbox : list or tuple
        The bounding box in the form [xmin, ymin, xmax, ymax].
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    stac_source : str
        Path or URL to the STAC catalog.
    collection : str
        The collection id to filter items.
    threads : int, optional
        Number of threads for PDAL processing (default is 4).
    bbox_crs : str, optional
        CRS of the bbox (default "EPSG:4326").
    target_crs : str, optional
        Target CRS for reprojection (if different from bbox_crs).

    Returns
    -------
    point_clouds : list of np.ndarray
        List of structured numpy arrays (point clouds) with the additional fields.
    uav_approx_dsm : np.ndarray
        2D grid representing the approximate digital surface model.
    uav_approx_dtm : np.ndarray
        2D grid representing the approximate digital terrain model.
    """
    # Read the STAC catalog and filter items.
    catalog = pystac.read_file(stac_source)
    items = []
    for item in catalog.get_all_items():
        if item.collection_id == collection:
            item_date = item.datetime.date()
            if start_date <= str(item_date) <= end_date:
                items.append(item)

    if not items:
        raise ValueError("No items found for the specified parameters.")

    point_clouds = []
    for item in items:
        if 'copc' not in item.assets:
            continue

        pc_file = item.assets['copc'].href

        pipeline_dict = {
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": pc_file,
                    "threads": threads,
                    "bounds": f"([{bbox[0]},{bbox[2]}],[{bbox[1]},{bbox[3]}])"
                }
            ]
        }

        if target_crs and target_crs != bbox_crs:
            pipeline_dict["pipeline"].append({
                "type": "filters.reprojection",
                "out_srs": target_crs
            })

        pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
        try:
            pipeline.execute()
            arrays = pipeline.arrays
            if len(arrays) > 0:
                point_clouds.append(arrays[0])
        except Exception as e:
            print(f"Pipeline execution error: {e}")
            continue
        finally:
            del pipeline
            gc.collect()
    try:

        # If at least one point cloud was created, compute additional fields.
        if point_clouds:
            for i, pc in enumerate(point_clouds):
                # Process the current point cloud.
                uav_point_cloud = pc
                            
                #add a constant to convert from ellipsoid height. Constant found from here:
                #https://geographiclib.sourceforge.io/cgi-bin/GeoidEval?input=33.1332887778855%2C+-116.60471670447386&option=Submit
                #NOTE: THIS CONSTANT ONLY APPLIES TO VOLCAN MTN. IT WILL NEED TO BE ADJUSTED FOR SEDGWICK.
                uav_point_cloud['Z'] = uav_point_cloud['Z'] + 31.8684
                
                # Compute the approximate DSM (max of Z) and DTM (min of Z) at 0.5m resolution.
                dsm_50cm, _, _, _ = aggregate_grid(uav_point_cloud, resolution=0.5,
                                                     agg_func=np.max, field='Z',
                                                     remove_outliers=True, outlier_threshold=3, bounding_box=bbox)
                
                dtm_50cm, x_orig, y_orig, res = aggregate_grid(uav_point_cloud, resolution=0.5,
                                                                 agg_func=np.min, field='Z',
                                                                 remove_outliers=True, outlier_threshold=3, bounding_box=bbox)
                # Compute approximate canopy height (CHM)
                ch_50cm = dsm_50cm - dtm_50cm
                
                # Assign the raw approximate canopy height to a new field.
                uav_point_cloud = assign_aggregated_values(uav_point_cloud, ch_50cm,
                                                             x_orig, y_orig, res, new_field='ch_50cm')
            
                # Compute the approximate DSM (max of Z) and DTM (min of Z) at 1m resolution.
                dsm_1m, _, _, _ = aggregate_grid(uav_point_cloud, resolution=1,
                                                   agg_func=np.max, field='Z',
                                                   remove_outliers=True, outlier_threshold=3, bounding_box=bbox)
                
                dtm_1m, x_orig, y_orig, res = aggregate_grid(uav_point_cloud, resolution=1,
                                                               agg_func=np.min, field='Z',
                                                               remove_outliers=True, outlier_threshold=3, bounding_box=bbox)
                # Compute approximate canopy height (CHM)
                ch_1m = dsm_1m - dtm_1m
                
                # Assign the raw approximate canopy height to a new field.
                uav_point_cloud = assign_aggregated_values(uav_point_cloud, ch_1m,
                                                             x_orig, y_orig, res, new_field='ch_1m')
                
                # Replace the processed point cloud back into the list.
                point_clouds[i] = uav_point_cloud
                        
    except Exception as e:
        print(f"dtm, dsm or chm error: {e}")
    
    # Clean up catalog items (if desired)
    del items, catalog
    gc.collect()

    return point_clouds, dsm_50cm, dtm_50cm, dsm_1m, dtm_1m

# def create_pointcloud_stack(bbox, start_date, end_date, stac_source, collection, threads = 4, bbox_crs="EPSG:4326", target_crs=None):

#     catalog = pystac.read_file(stac_source)
#     items = []
#     for item in catalog.get_all_items():
#         if item.collection_id == collection:
#             item_date = item.datetime.date()
#             if start_date <= str(item_date) <= end_date:
#                 items.append(item)

#     if not items:
#         raise ValueError("No items found for the specified parameters.")

#     point_clouds = []
#     for item in items:
#         if 'copc' not in item.assets:
#             continue

#         pc_file = item.assets['copc'].href

#         pipeline = {
#             "pipeline": [
#                 {
#                     "type": "readers.copc",
#                     "filename": pc_file,
#                     "threads": threads,  # Use the threads argument here
#                     "bounds": f"([{bbox[0]},{bbox[2]}],[{bbox[1]},{bbox[3]}])"
#                 }
#             ]
#         }

#         if target_crs and target_crs != bbox_crs:
#             pipeline["pipeline"].append({
#                 "type": "filters.reprojection",
#                 "out_srs": target_crs
#             })

#         pipeline = pdal.Pipeline(json.dumps(pipeline))
#         try:
#             pipeline.execute()
#             arrays = pipeline.arrays
#             if len(arrays) > 0:
#                 point_clouds.append(arrays[0])
#         except Exception as e:
#             print(f"Pipeline execution error: {e}")
            
#             continue
        
#         del pipeline
#         del arrays
#         del items 
#         del catalog
#         gc.collect()
#     return point_clouds


def visualize_pointclouds(point_clouds, elev=45, azim=45):
    """
    Visualize a list of point clouds in a 3D scatter plot with adjustable viewing angles.

    :param point_clouds: list of NumPy structured arrays, each containing point data (X, Y, Z, etc.).
    :param elev: elevation angle in degrees (default=30).
    :param azim: azimuth angle in degrees (default=45).
    """
    if not point_clouds:
        print("No point clouds to visualize.")
        return

    # Get the common fields across all point clouds
    common_fields = set(point_clouds[0].dtype.names)
    for pc in point_clouds:
        common_fields.intersection_update(pc.dtype.names)

    # Normalize all arrays to have only the common fields
    normalized_point_clouds = []
    for pc in point_clouds:
        # Create a new array with only the common fields
        normalized_pc = np.zeros(pc.shape, dtype=[(field, pc.dtype[field]) for field in common_fields])
        for field in common_fields:
            normalized_pc[field] = pc[field]
        normalized_point_clouds.append(normalized_pc)

    # Concatenate all normalized point clouds into one array
    all_points = np.concatenate(normalized_point_clouds)

    # Extract X, Y, Z
    x = all_points['X']
    y = all_points['Y']
    z = all_points['Z']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    sc = ax.scatter(x, y, z, c=z, s=0.1, cmap='viridis')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Optionally fix aspect ratio so the axes scales are consistent:
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    plt.title("Point Cloud Visualization")
    plt.show()




def bounding_box_to_geojson(bbox):
    """
    Converts a bounding box to a GeoJSON polygon.
    """
    return json.dumps(mapping(box(bbox[0], bbox[1], bbox[2], bbox[3])))

def create_3dep_stack(bbox, start_date, end_date, threads = 4, target_crs=None):
    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=sign_inplace
    )

    try:
        search = client.search(
            collections=["3dep-lidar-copc"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}"
        )
        items = list(search.items())
        if not items:
            raise ValueError("No items found for the specified parameters.")
    finally:
        del client  # Explicitly delete or close client if applicable
        del search
        gc.collect()
    
    
    polygon = bounding_box_to_geojson(bbox)
    point_clouds = []
    for tile in items:
        try:
            url = planetary_computer.sign(tile.assets["data"].href)
            
            pipeline_dict = {
                "pipeline": [
                    {
                        "type": "readers.copc",
                        "filename": url,
                        "polygon": polygon,
                        "threads": threads  # Use the threads argument
                    }
                ]
            }
            
            if target_crs:
                pipeline_dict["pipeline"].append({
                    "type": "filters.reprojection",
                    "out_srs": target_crs
                })
            
            pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
            pipeline.execute()
            
            arrays = pipeline.arrays
            if len(arrays) > 0 and arrays[0].size > 0:
                point_clouds.append(arrays[0])
            del pipeline
        except Exception as e:
            print(f"Error processing tile {url}: {str(e)}")
            continue
    
    if not point_clouds:
        raise ValueError("No point cloud data was successfully retrieved")
    # Find the common fields across all arrays
    common_fields = set(point_clouds[0].dtype.names)
    for pc in point_clouds[1:]:
        common_fields.intersection_update(pc.dtype.names)

    # Keep only the common fields and return as a list of structured arrays
    common_fields = list(common_fields)
    filtered_point_clouds = [
        pc[common_fields].copy() for pc in point_clouds
    ]
    del point_clouds
    del items
    del common_fields
    gc.collect()
    return filtered_point_clouds





def get_stac_catalog_bbox(catalog_path):
    """
    Retrieve the bounding box of the given STAC catalog.

    Parameters:
        catalog_path (str): Path to the STAC catalog.

    Returns:
        list: Bounding box [min_x, min_y, max_x, max_y] of the catalog.
    """

    
    catalog = pystac.Catalog.from_file(catalog_path)
    collections = list(catalog.get_collections())
    
    if not collections:
        raise ValueError("No collections found in catalog")
    
    # Assuming the catalog's bounding box is the union of its collections' bounding boxes
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    for collection in collections:
        bbox = collection.extent.spatial.bboxes[0]
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    del collections
    return [min_x, min_y, max_x, max_y]


import random

def sample_bounding_boxes_within_catalog(catalog_path, n, box_width, box_height, shrink_percentage=0):
    """
    Sample bounding boxes of specified dimensions that lie entirely within the STAC catalog bounding box,
    with an optional shrinkage percentage to focus sampling closer to the center.

    Parameters:
        catalog_path (str): Path to the STAC catalog.
        n (int): Number of bounding boxes to sample.
        box_width (float): Width of each sampled bounding box.
        box_height (float): Height of each sampled bounding box.
        shrink_percentage (float): Percentage to shrink the bounding box from each side (0-100).

    Returns:
        list: A list of sampled bounding boxes as [min_x, min_y, max_x, max_y].
    """
    # Get the catalog's bounding box
    catalog_bbox = get_stac_catalog_bbox(catalog_path)

    min_x, min_y, max_x, max_y = catalog_bbox
    
    bbox_x_width = (max_x - min_x)
    bbox_y_width = (max_y - min_y)
    print(f"catalog_bbox = {catalog_bbox}")
    print(f"bbox_x_width = {bbox_x_width}")
    print(f"bbox_y_width = {bbox_y_width}")
    
    # Calculate shrinkage
    shrink_factor = shrink_percentage / 100
    width_shrink = bbox_x_width * shrink_factor / 2
    height_shrink = bbox_y_width * shrink_factor / 2

    # Shrink the bounding box
    min_x += width_shrink
    max_x -= width_shrink
    min_y += height_shrink
    max_y -= height_shrink

    print(  [min_x, min_y, max_x, max_y])
    print(f"shrink bbox_x_width = {max_x - min_x}")
    print(f"shrink bbox_y_width = {max_y - min_y}")
    # Ensure the catalog bounding box is large enough for the specified dimensions
    if max_x - min_x < box_width or max_y - min_y < box_height:
        raise ValueError("Shrunk bounding box is too small to fit the specified dimensions.")

    # Sample bounding boxes
    sampled_bboxes = []
    for _ in range(n):
        random_min_x = random.uniform(min_x, max_x - box_width)
        random_min_y = random.uniform(min_y, max_y - box_height)
        sampled_bboxes.append([random_min_x, random_min_y, random_min_x + box_width, random_min_y + box_height])

    return sampled_bboxes



def process_bbox(args):
    
    i, bbox, start_date, end_date, stac_source, collection, bbox_crs = args
    # print(f"Start bounding box {i}")
    try:
        # UAV LiDAR point clouds, DTM, and DSM
        uav_pc, dsm_50cm, dtm_50cm, dsm_1m, dtm_1m= create_pointcloud_stack(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            stac_source=stac_source,
            collection=collection,
            bbox_crs=bbox_crs,
        )
        
        # print(f"bounding box {i}: done uav_pc")
        if not uav_pc or len(uav_pc) == 0:
            print(f"No UAV points found for bounding box {i}. Skipping.")
            return None

        xyz_uav = np.array([(p['X'], p['Y'], p['Z'] ) for p in uav_pc], dtype=np.float32)

        ch_50cm = np.array([(p['ch_50cm']) for p in uav_pc], dtype=np.float32)

        ch_1m = np.array([(p['ch_1m']) for p in uav_pc], dtype=np.float32)
        # 3DEP LiDAR point clouds
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        bbox_wgs84 = transformer.transform_bounds(*bbox)

        dep_pc = create_3dep_stack(
            bbox=bbox_wgs84,
            start_date=start_date,
            end_date=end_date
        )
        
        if not dep_pc or len(dep_pc) == 0:
            print(f"No 3DEP points found for bounding box {i}. Skipping.")
            return None
        
        xyz_dep = np.vstack([np.column_stack((p['X'], p['Y'], p['Z'])) for p in dep_pc]).astype(np.float32)
        print(f"Bounding box {i}: {bbox}: UAV: {xyz_uav.shape}; 3DEP: {xyz_dep.shape} ---> Memory usage: {psutil.virtual_memory().percent}%, Open file descriptors: {psutil.Process().num_fds()}")
        
        del transformer
        del bbox_wgs84
        del uav_pc
        del dep_pc
        gc.collect
        return {
            'dep_points': torch.tensor(xyz_dep, dtype=torch.float32),
            'uav_points': torch.tensor(xyz_uav.squeeze(), dtype=torch.float32).T,
        
            'dsm_50cm': torch.tensor(dsm_50cm, dtype=torch.float32),
            'dtm_50cm': torch.tensor(dtm_50cm, dtype=torch.float32),
            'dsm_1m': torch.tensor(dsm_1m, dtype=torch.float32),
            'dtm_1m': torch.tensor(dtm_1m, dtype=torch.float32),
        
            'ch_50cm': torch.tensor(ch_50cm, dtype=torch.float32),
            'ch_1m': torch.tensor(ch_1m, dtype=torch.float32),
            'bbox':bbox
        }
    except Exception as e:
        error_msg = f"Error processing bounding box {i}: {e}"
        return None

import math

def chunk_data(data, chunk_size):
    """Split data into smaller chunks of size `chunk_size`."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def process_and_store_training_data(
    catalog_path, 
    n, 
    box_width, 
    box_height, 
    start_date, 
    end_date, 
    stac_source, 
    collection, 
    bbox_crs="EPSG:32611", 
    target_crs="EPSG:32611",
    shrink_percentage=0, 
    max_threads=2,
    chunk_size=100,
    output_dir="training_data_chunks"
):
    sampled_bboxes = sample_bounding_boxes_within_catalog(catalog_path, n, box_width, box_height, shrink_percentage)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split sampled bounding boxes into chunks
    chunks = list(chunk_data(sampled_bboxes, chunk_size))
    print(f"Processing {len(chunks)} chunks of bounding boxes, with up to {chunk_size} per chunk.")

    for chunk_index, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {chunk_index}/{len(chunks)}...")
        
        # Prepare arguments for parallel processing
        args_list = [
            (i, bbox, start_date, end_date, stac_source, collection, bbox_crs)
            for i, bbox in enumerate(chunk, start=1)
        ]
        
        chunk_results = []
        try:
            with ProcessPoolExecutor(max_workers=max_threads) as executor:
                for result in executor.map(process_bbox, args_list):
                    if result is not None:
                        chunk_results.append(result)
        except Exception as e:
            print(f"Error during parallel processing of chunk {chunk_index}: {e}")
        
        # Write chunk results to disk
        if chunk_results:
            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_file = os.path.join(output_dir, f"chunk_{timestamp}.pt")
            torch.save(chunk_results, chunk_file)
            print(f"Saved chunk {chunk_index} with {len(chunk_results)} results to {chunk_file}")
        else:
            print(f"No results for chunk {chunk_index}. Skipping saving.")

        # Clean up memory
        del args_list
        del chunk_results
        gc.collect()

    print("Completed processing all chunks.")



if __name__ == "__main__":
    catalog_path = "uavls_stac/catalog.json"
    n =4000 # Number of bounding boxes to sample
    box_width = 10  # Width of each bounding box
    box_height = 10  # Height of each bounding box
    start_date = "2014-01-01"
    end_date = "2025-01-27"
    stac_source = "uavls_stac/catalog.json"
    collection = "pointcloud_20241025_205750"
    
    n_target_points = 1000  # Target points for training data
    shrink_percentage =10
    # Run the function
    training_data = process_and_store_training_data(
        catalog_path=catalog_path,
        n=n,
        box_width=box_width,
        box_height=box_height,
        start_date=start_date,
        end_date=end_date,
        stac_source=stac_source,
        collection=collection,
        shrink_percentage = shrink_percentage,
        max_threads = 15,
        chunk_size=100,
        output_dir="training_data_chunks/point_clouds/"
    )