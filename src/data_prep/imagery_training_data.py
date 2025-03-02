import os
import math
import gc
import pystac
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box, mapping
from rasterio.enums import Resampling
from stackstac import stack


def create_singleband_stac_stack(bbox, start_date, end_date, stac_source, assets, out_resolution,
                     bbox_crs="EPSG:4326", target_crs=None, resampling_method=Resampling.cubic):
    """
    Creates a data stack from a local STAC catalog within a specified bounding box.
    
    Parameters:
    ----------
    bbox : list
        Bounding box in the format [xmin, ymin, xmax, ymax]
    start_date : str
        Start date for filtering in YYYY-MM-DD format
    end_date : str
        End date for filtering in YYYY-MM-DD format
    stac_source : str
        Path to the local STAC catalog
    assets : list
        List of asset names to include in the stack
    out_resolution : float
        Output resolution in meters
    bbox_crs : str, optional
        CRS of the input bounding box. Default is "EPSG:4326"
    target_crs : str, optional
        Target CRS for the output. If None, will use the CRS from the first item.
    resampling_method : Resampling, optional
        Resampling method to use. Default is cubic.
        
    Returns:
    -------
    tuple:
        - xarray.DataArray: The computed data stack
        - dict: Metadata dictionary for the most recent item
    """
    # Helper function to round bounds to the nearest resolution
    def round_bounds(bounds, resolution):
        minx, miny, maxx, maxy = bounds
        minx = math.floor(minx / resolution) * resolution
        miny = math.floor(miny / resolution) * resolution
        maxx = math.ceil(maxx / resolution) * resolution
        maxy = math.ceil(maxy / resolution) * resolution
        return (minx, miny, maxx, maxy)
    
    # Define bbox intersection function
    def bboxes_intersect(bbox1, bbox2):
        # Return True if bbox1 intersects bbox2 (both in [xmin, ymin, xmax, ymax] format)
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
    
    # Read the STAC catalog
    catalog = None
    items = []
    item_metadata = []
    
    try:
        # Reproject bbox to EPSG:4326 if needed for filtering
        if bbox_crs != "EPSG:4326":
            transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
            bbox_4326 = transformer.transform_bounds(*bbox)
        else:
            bbox_4326 = bbox
        
        # Read the local STAC catalog
        catalog = pystac.read_file(stac_source)
        
        # Filter items by date and spatial intersection
        for item in catalog.get_all_items():
            # Check the date
            item_date = item.datetime.date()
            if not (start_date <= str(item_date) <= end_date):
                continue
                
            # Check horizontal bounding box intersection
            if item.bbox:
                if bboxes_intersect(bbox_4326, item.bbox):
                    items.append(item)
                    
                    # Create metadata for this item - extract only id and datetime
                    meta = {
                        'id': item.id,
                        'datetime': item.datetime.isoformat() if item.datetime else None,
                    }
                    item_metadata.append(meta)
        
        if not items:
            raise ValueError("No items found for the specified parameters.")
            
        print(f"Found {len(items)} items between {start_date} and {end_date}")
        
        # Determine the target CRS if not provided
        if target_crs is None:
            if 'proj:epsg' in items[0].properties:
                target_crs = f"EPSG:{items[0].properties['proj:epsg']}"
            else:
                raise ValueError("No target_crs provided and no 'proj:epsg' property found in items.")
        
        print(f"Using target CRS: {target_crs}")
        
        # Reproject the given bbox from bbox_crs to target_crs
        gdf = gpd.GeoDataFrame(
            geometry=[box(*bbox)],
            crs=bbox_crs
        )
        gdf_target = gdf.to_crs(target_crs)
        
        # Get reprojected bounds and round to resolution
        reproj_bounds = gdf_target.geometry.iloc[0].bounds
        reproj_bounds = round_bounds(reproj_bounds, out_resolution)
        
        # Stack the imagery
        stack_data = stack(
            items,
            bounds=reproj_bounds,
            snap_bounds=False,
            epsg=int(target_crs.split(":")[1]),
            resolution=out_resolution,
            fill_value=0,
            assets=assets,
            resampling=resampling_method,
            rescale=False
        )
        
        computed_stack = stack_data.compute()
        
        # Get metadata from the most recent item
        latest_metadata = {}
        if item_metadata:
            # Sort by datetime if available
            item_metadata.sort(key=lambda x: x['datetime'], reverse=True)
            latest_metadata = item_metadata[0]
        
        # Print stats about the output
        print(f"Stack dimensions: {computed_stack.sizes}")
        print(f"Resolution: {out_resolution}m")
        
        return computed_stack, latest_metadata
        
    except Exception as e:
        print(f"Error in create_data_stack: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
        
    finally:
        # Clean up resources
        if items:
            del items
            
        if catalog:
            del catalog
            
        gc.collect()



import os
import math
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import pystac
import gc

def bboxes_intersect(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.
    
    Parameters:
      bbox1 (tuple): (minx, miny, maxx, maxy)
      bbox2 (tuple): (minx, miny, maxx, maxy)
      
    Returns:
      bool: True if bounding boxes intersect, False otherwise
    """
    # Extract coordinates
    minx1, miny1, maxx1, maxy1 = bbox1
    minx2, miny2, maxx2, maxy2 = bbox2
    
    # Check if one box is to the left of the other
    if maxx1 < minx2 or maxx2 < minx1:
        return False
    
    # Check if one box is above the other
    if maxy1 < miny2 or maxy2 < miny1:
        return False
    
    # If neither of the above is true, the boxes must intersect
    return True

def round_bounds(bounds, resolution):
    """
    Round bounds to the nearest resolution.
    
    Parameters:
      bounds (tuple): (minx, miny, maxx, maxy)
      resolution (float): Resolution to round to
      
    Returns:
      tuple: Rounded bounds (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / resolution) * resolution
    miny = math.floor(miny / resolution) * resolution
    maxx = math.ceil(maxx / resolution) * resolution
    maxy = math.ceil(maxy / resolution) * resolution
    return (minx, miny, maxx, maxy)

import os
import math
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import pystac
import gc

# Note: This module provides functionality for working with multiband imagery STAC catalogs
# as an alternative to stackstac which only supports single-band imagery
# Note: This module works with multiband imagery STAC catalogs
# as an alternative to stackstac which only works with single-band imagery

def bboxes_intersect(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.
    
    Parameters:
      bbox1 (tuple): (minx, miny, maxx, maxy)
      bbox2 (tuple): (minx, miny, maxx, maxy)
      
    Returns:
      bool: True if bounding boxes intersect, False otherwise
    """
    # Extract coordinates
    minx1, miny1, maxx1, maxy1 = bbox1
    minx2, miny2, maxx2, maxy2 = bbox2
    
    # Check if one box is to the left of the other
    if maxx1 < minx2 or maxx2 < minx1:
        return False
    
    # Check if one box is above the other
    if maxy1 < miny2 or maxy2 < miny1:
        return False
    
    # If neither of the above is true, the boxes must intersect
    return True

def round_bounds(bounds, resolution):
    """
    Round bounds to the nearest resolution.
    
    Parameters:
      bounds (tuple): (minx, miny, maxx, maxy)
      resolution (float): Resolution to round to
      
    Returns:
      tuple: Rounded bounds (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / resolution) * resolution
    miny = math.floor(miny / resolution) * resolution
    maxx = math.ceil(maxx / resolution) * resolution
    maxy = math.ceil(maxy / resolution) * resolution
    return (minx, miny, maxx, maxy)

def create_multiband_stac_stack(bbox, start_date, end_date, local_catalog_path,
                             asset="image", out_resolution=0.5, bbox_crs="EPSG:4326",
                             target_crs=None, resampling_method=Resampling.cubic,
                             fill_value=0):
    """
    Creates a data stack from a local STAC catalog containing MULTIBAND imagery (e.g., NAIP).
    This function is designed as an alternative to stackstac, which only works with single-band
    imagery. When using stackstac with multiband imagery, you'll encounter errors like:
    "Assets must have exactly 1 band, but file has 4. We can't currently handle multi-band rasters."
    
    This function handles multiband GeoTIFFs properly, reprojecting and resampling each image 
    to a common grid defined by the reprojected bounding box and out_resolution, then stacking
    them along the time dimension.
    
    Parameters:
      bbox (tuple): (minx, miny, maxx, maxy) in bbox_crs coordinates.
      start_date (str): e.g., "2018-01-01"
      end_date (str): e.g., "2024-12-30"
      local_catalog_path (str): Path to the local STAC catalog file.
      asset (str): The asset key to use from each item. (Default: "image")
      out_resolution (float): Output resolution in target_crs units.
      bbox_crs (str): CRS of the input bbox (default "EPSG:4326").
      target_crs (str): Output CRS. If None, inferred from the first item's "proj:epsg".
      resampling_method: Rasterio resampling method. Default: Resampling.cubic.
      fill_value: Value to use for areas with no data.
      
    Returns:
      tuple:
        - xarray.DataArray: DataArray with dims ("time", "band", "y", "x")
        - dict: Metadata dictionary for the most recent item
    """
    catalog = None
    items = []
    item_metadata = []
    computed_da = None
    
    try:
        # -- Reproject bbox to EPSG:4326 if needed for the STAC search --
        if bbox_crs != "EPSG:4326":
            transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
            minx, miny, maxx, maxy = bbox
            minx, miny = transformer.transform(minx, miny)
            maxx, maxy = transformer.transform(maxx, maxy)
            bbox_4326 = (minx, miny, maxx, maxy)
        else:
            bbox_4326 = bbox

        # -- Query local STAC catalog --
        print("Opening local STAC catalog...")
        catalog = pystac.read_file(local_catalog_path)
        
        # Filter items by date and spatial intersection
        for item in catalog.get_all_items():
            # Check the date
            item_date = item.datetime.date()
            if not (start_date <= str(item_date) <= end_date):
                continue
                
            # Check horizontal bounding box intersection
            if item.bbox:
                if bboxes_intersect(bbox_4326, item.bbox):
                    items.append(item)
                    
                    # Create metadata for this item
                    meta = {
                        'id': item.id,
                        'datetime': item.datetime.isoformat() if item.datetime else None,
                        'bbox': item.bbox,
                        'collection': item.collection_id
                    }
                    
                    # Add additional properties if available
                    for key in ['eo:cloud_cover', 'platform', 'instrument', 'gsd', 'proj:epsg']:
                        if key in item.properties:
                            meta[key] = item.properties[key]
                    
                    item_metadata.append(meta)
        
        if not items:
            raise ValueError("No items found for the specified date range and bounding box.")
        print(f"Found {len(items)} items between {start_date} and {end_date} that intersect with the bounding box")

        # -- Determine target CRS --
        if target_crs is None:
            if 'proj:epsg' in items[0].properties:
                target_crs = f"EPSG:{items[0].properties['proj:epsg']}"
            else:
                raise ValueError("target_crs not provided and cannot be inferred.")
        print(f"Target CRS: {target_crs}")

        # -- Reproject input bbox to target_crs to define common grid --
        gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=bbox_crs)
        gdf_target = gdf.to_crs(target_crs)
        reproj_bounds = gdf_target.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)
        
        # Round bounds to the nearest resolution
        reproj_bounds = round_bounds(reproj_bounds, out_resolution)
        print(f"Reprojected Bounding Box (rounded): {reproj_bounds}")

        minx_t, miny_t, maxx_t, maxy_t = reproj_bounds
        width = math.ceil((maxx_t - minx_t) / out_resolution)
        height = math.ceil((maxy_t - miny_t) / out_resolution)
        # Use rasterio's from_origin; note that the "origin" is the top-left corner.
        transform = from_origin(minx_t, maxy_t, out_resolution, out_resolution)
        print(f"Output grid: {width} x {height} pixels")

        # -- Get number of bands from the first item --
        with rasterio.open(items[0].assets[asset].href) as src:
            nbands = src.count
            dtype = src.dtypes[0]
        print(f"Number of bands: {nbands}")
        print(f"Data type: {dtype}")
        
        # -- Function to reproject an item's asset onto the common grid --
        def process_item(item):
            asset_obj = item.assets[asset]
            print(f"Processing: {asset_obj.href}")

            with rasterio.open(asset_obj.href) as src:
                print(f"CRS: {src.crs}, Shape: {src.shape}")
                src_nbands = src.count
                # Prepare destination array with shape (bands, height, width)
                dest = np.full((src_nbands, height, width), fill_value, dtype=dtype)
                reproject(
                    source=src.read(),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method,
                    dst_nodata=fill_value
                )
                return dest

        # -- Create a delayed dask array for each item --
        delayed_arrays = []
        times = []
        
        # Sort items by datetime
        items.sort(key=lambda x: x.datetime, reverse=False)
        
        for item in items:
            # Create a delayed version of the processed image
            darr = delayed(process_item)(item)
            # Wrap with dask.array.from_delayed; shape is (nbands, height, width)
            darr = da.from_delayed(darr, shape=(nbands, height, width), dtype=dtype)
            delayed_arrays.append(darr)
            times.append(np.datetime64(item.datetime))  # use numpy datetime64 for coordinates

        # Stack along a new "time" axis (resulting shape: (time, band, y, x))
        data = da.stack(delayed_arrays, axis=0)
        print(f"Stacked data shape: {data.shape}")
        
        # -- Define spatial coordinate arrays based on the affine transform --
        # The top-left corner is at (transform.c, transform.f)
        x_coords = transform.c + np.arange(width) * out_resolution
        y_coords = transform.f - np.arange(height) * out_resolution

        # Create the DataArray
        da_stack = xr.DataArray(
            data,
            dims=("time", "band", "y", "x"),
            coords={
                "time": times,
                "band": np.arange(1, nbands + 1),
                "x": x_coords,
                "y": y_coords
            },
            attrs={
                "crs": target_crs,
                "transform": transform.to_gdal(),  # GDAL-style transform tuple
                "resolution": out_resolution,
                "created": np.datetime64('now').astype(str)
            }
        )

        # Get metadata from the most recent item (last in the time series)
        latest_metadata = {}
        if item_metadata:
            # Sort by datetime
            item_metadata.sort(key=lambda x: x['datetime'], reverse=True)
            latest_metadata = item_metadata[0]
        
        # Compute the data array
        print("Computing data stack...")
        computed_da = da_stack.compute(scheduler="threads")
        
        # Print stats about the output
        print(f"Final stack dimensions: {computed_da.sizes}")
        print(f"Final memory usage: ~{computed_da.nbytes / (1024**2):.2f} MB")
        print(f"Resolution: {out_resolution}m")
        print(f"Time range: {computed_da.time.min().values} to {computed_da.time.max().values}")
        
        return computed_da, latest_metadata
        
    except Exception as e:
        print(f"Error in create_local_naip_data_stack: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
        
    finally:
        # Clean up resources
        if items:
            del items
            
        if catalog:
            del catalog
            
        gc.collect()

    
import numpy as np
import pandas as pd
from datetime import datetime

def convert_stack_to_tensors(data_stack, metadata, bbox, resolution):
    """
    Convert xarray DataArray from create_data_stack() to a dictionary of torch tensors with metadata.
    
    Parameters:
    -----------
    data_stack : xarray.DataArray
        The stacked imagery from create_data_stack()
    metadata : dict
        The metadata dictionary from create_data_stack()
    bbox : list or tuple
        The bounding box used in the create_data_stack() call [xmin, ymin, xmax, ymax]
    resolution : float
        The output resolution used in the create_data_stack() call
        
    Returns:
    --------
    dict
        A dictionary containing the imagery as torch tensors and associated metadata
    """
    if data_stack is None:
        return None
        
    # Get the date and id attributes from the xarray
    times = data_stack.time.values
    ids = data_stack.id.values if hasattr(data_stack, 'id') else [metadata['id']] * len(times)
    bands = data_stack.band.values
    
    # Create a dictionary to hold the tensors and metadata
    result = {
        'imgs': [],
        'imgs_meta': [],
        'bbox': bbox,
        'resolution': resolution
    }
    
    # Convert each time slice to a torch tensor and capture metadata
    for i, (time, item_id) in enumerate(zip(times, ids)):
        # Extract data for this time
        img_data = data_stack.sel(time=time).values
        
        # Convert to torch tensor - shape is (band, y, x)
        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        
        # Add to imgs list
        result['imgs'].append(img_tensor)
        
        # Parse datetime
        if isinstance(time, np.datetime64):
            date_str = pd.Timestamp(time).strftime('%Y-%m-%d')
        else:
            date_str = time
            
        # Create metadata entry
        img_metadata = {
            'id': item_id,
            'date': date_str,
            'datetime': pd.Timestamp(time).isoformat(),
            'bbox': bbox,
            'resolution': resolution,
            'bands': list(bands),
            'shape': img_tensor.shape,
            'index': i
        }
        
        # Add additional metadata from the metadata dictionary if available
        if metadata and isinstance(metadata, dict):
            # If we have specific metadata for this item (by id), use it
            if 'href' in metadata:
                img_metadata['href'] = metadata['href']
                
        # Add to imgs_meta list
        result['imgs_meta'].append(img_metadata)
    
    # Consolidate tensors into a single batch tensor if all have the same shape
    if all(tensor.shape == result['imgs'][0].shape for tensor in result['imgs']):
        result['imgs_tensor'] = torch.stack(result['imgs'])
    
    return result


def process_imagery_data(bbox, start_date, end_date, stac_source, assets, out_resolution, 
                        bbox_crs="EPSG:4326", target_crs=None, resampling_method=None):
    """
    Helper function to process imagery data using create_data_stack and convert to tensors.
    
    Parameters are the same as create_data_stack.
    
    Returns:
    --------
    dict
        A dictionary containing the imagery as torch tensors and associated metadata
    """
    # Import here to avoid circular imports
    from rasterio.enums import Resampling
    
    # Set default resampling method if none provided
    if resampling_method is None:
        resampling_method = Resampling.cubic
    
    # Call create_data_stack to get the data
    data_stack, metadata = create_data_stack(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        stac_source=stac_source,
        assets=assets,
        out_resolution=out_resolution,
        bbox_crs=bbox_crs,
        target_crs=target_crs,
        resampling_method=resampling_method
    )
    
    # Convert to tensors and return
    return convert_stack_to_tensors(data_stack, metadata, bbox, out_resolution)



    import torch
import copy

def combine_imagery_pointcloud(imagery_data, pointcloud_data):
    """
    Combine imagery tensor dictionary with point cloud tensor dictionary.
    
    Parameters:
    -----------
    imagery_data : dict
        Dictionary containing imagery tensors and metadata from convert_stack_to_tensors()
    pointcloud_data : dict
        Dictionary containing point cloud tensors and metadata from process_bbox()
        
    Returns:
    --------
    dict
        A combined dictionary with both imagery and point cloud data
    """
    if imagery_data is None or pointcloud_data is None:
        print("Error: Either imagery data or point cloud data is None.")
        return None
    
    # Create a deep copy to avoid modifying the originals
    combined_data = copy.deepcopy(pointcloud_data)
    
    # Add imagery data
    combined_data['imgs'] = imagery_data['imgs']
    combined_data['imgs_meta'] = imagery_data['imgs_meta']
    combined_data['img_resolution'] = imagery_data['resolution']
    
    # If we have a stacked tensor, add it
    if 'imgs_tensor' in imagery_data:
        combined_data['imgs_tensor'] = imagery_data['imgs_tensor']
    
    # Add validation flag - True if both point cloud and imagery data are present
    combined_data['has_imagery'] = True
    combined_data['has_pointcloud'] = True
    
    return combined_data


def process_tile_with_imagery(tile_id, bbox, stac_source, lidar_stac_source, 
                             start_date, end_date, assets, out_resolution, 
                             bbox_crs="EPSG:32611", target_crs="EPSG:32611"):
    """
    Process a tile to get both imagery and point cloud data.
    
    Parameters:
    -----------
    tile_id : str
        Identifier for the tile
    bbox : list or tuple
        Bounding box [xmin, ymin, xmax, ymax]
    stac_source : str
        Path to imagery STAC catalog
    lidar_stac_source : str
        Path to LiDAR STAC catalog
    start_date : str
        Start date for filtering
    end_date : str
        End date for filtering
    assets : list
        List of imagery assets to include
    out_resolution : float
        Output resolution for imagery
    bbox_crs : str
        CRS of the input bounding box
    target_crs : str
        Target CRS for the output
        
    Returns:
    --------
    dict
        Combined dictionary with both imagery and point cloud data
    """
    from rasterio.enums import Resampling
    
    # Get imagery data
    print(f"Processing imagery for tile {tile_id}...")
    imagery_data = process_imagery_data(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        stac_source=stac_source,
        assets=assets,
        out_resolution=out_resolution,
        bbox_crs=bbox_crs,
        target_crs=target_crs,
        resampling_method=Resampling.cubic
    )
    
    # Get point cloud data
    print(f"Processing point cloud for tile {tile_id}...")
    pointcloud_data = process_bbox(
        (0, tile_id, bbox, start_date, end_date, lidar_stac_source, bbox_crs)
    )
    
    # Combine the data
    if imagery_data and pointcloud_data:
        return combine_imagery_pointcloud(imagery_data, pointcloud_data)
    else:
        if not imagery_data:
            print(f"No imagery data found for tile {tile_id}.")
        if not pointcloud_data:
            print(f"No point cloud data found for tile {tile_id}.")
        return None