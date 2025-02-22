import pystac_client
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import stackstac
import numpy as np
from rasterio.enums import Resampling
from shapely.geometry import box
import os
import requests
import pystac_client
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import stackstac
import numpy as np
from rasterio.enums import Resampling
import folium
from pyproj import Transformer
import rasterio
from stackstac import stack
import pystac
from pyproj import Transformer
import planetary_computer


import os
import copy
from shapely.geometry import box
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from pyproj import Transformer
import pystac
import pystac_client
import planetary_computer
from stackstac import stack

import os
import math
import copy
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
import pystac_client
import planetary_computer

# from dask.distributed import Client
import os




def create_naip_data_stack(bbox, start_date, end_date, stac_source, collection,
                           asset="image", out_resolution=0.5, bbox_crs="EPSG:4326",
                           target_crs=None, resampling_method=Resampling.cubic,
                           fill_value=0):
    """
    Queries a NAIP STAC catalog and returns an xarray.DataArray with dimensions
    ("time", "band", "y", "x"). Each image (a multi-band GeoTIFF) is reprojected
    and resampled to a common grid defined by the reprojected bounding box and
    out_resolution.
    
    Parameters:
      bbox (tuple): (minx, miny, maxx, maxy) in bbox_crs coordinates.
      start_date (str): e.g., "2018-01-01"
      end_date (str): e.g., "2024-12-30"
      stac_source (str): URL to a STAC API (or local catalog file path).
      collection (str): The collection ID (e.g. "naip").
      asset (str): The asset key to use from each item. (Default: "image")
      out_resolution (float): Output resolution in target_crs units.
      bbox_crs (str): CRS of the input bbox (default "EPSG:4326").
      target_crs (str): Output CRS. If None, inferred from the first item’s "proj:epsg".
      resampling_method: Rasterio resampling method. Default: Resampling.cubic.
      fill_value: Value to use for areas with no data.
      
    Returns:
      xarray.DataArray: DataArray with dims ("time", "band", "y", "x") and lazy Dask backing.
    """
    # -- Reproject bbox to EPSG:4326 if needed for the STAC search --
    if bbox_crs != "EPSG:4326":
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)
        bbox_4326 = (minx, miny, maxx, maxy)
    else:
        bbox_4326 = bbox

    # -- Query STAC items --
    if os.path.exists(stac_source):
        # print("Opening local STAC catalog...")
        catalog = pystac.read_file(stac_source)
        items = []
        for item in catalog.get_all_items():
            if item.collection_id == collection:
                item_date = item.datetime.date()
                if start_date <= str(item_date) <= end_date:
                    items.append(item)
    else:
        # print("Connecting to STAC API...")
        if "planetarycomputer" in stac_source:
            client = pystac_client.Client.open(
                stac_source,
                modifier=planetary_computer.sign_inplace
            )
        else:
            client = pystac_client.Client.open(stac_source)
        search = client.search(
            collections=[collection],
            datetime=f"{start_date}/{end_date}",
            bbox=bbox_4326,
            max_items=100,
        )
        # Using item_collection() (the modern alternative to get_all_items())
        items = list(search.item_collection())
        # Optionally, reduce to one item per date:
        unique = {}
        for item in items:
            d = item.datetime.date()
            if d not in unique:
                unique[d] = item
        items = list(unique.values())

    if not items:
        raise ValueError("No items found for the specified parameters.")
    # print(f"Found {len(items)} items in the collection '{collection}'")

    # -- Determine target CRS --
    if target_crs is None:
        if 'proj:epsg' in items[0].properties:
            target_crs = f"EPSG:{items[0].properties['proj:epsg']}"
        else:
            raise ValueError("target_crs not provided and cannot be inferred.")
    # print(f"Target CRS: {target_crs}")

    # -- Reproject input bbox to target_crs to define common grid --
    gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=bbox_crs)
    gdf_target = gdf.to_crs(target_crs)
    reproj_bounds = gdf_target.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)
    # print(f"Reprojected Bounding Box: {reproj_bounds}")

    minx_t, miny_t, maxx_t, maxy_t = reproj_bounds
    width = math.ceil((maxx_t - minx_t) / out_resolution)
    height = math.ceil((maxy_t - miny_t) / out_resolution)
    # Use rasterio's from_origin; note that the "origin" is the top-left corner.
    transform = from_origin(minx_t, maxy_t, out_resolution, out_resolution)
    # print(f"Output grid: {width} x {height} pixels.")

    # -- Get number of bands from the first item --
    with rasterio.open(items[0].assets[asset].href) as src:
        nbands = src.count
        dtype = src.dtypes[0]
    # print(f"Number of bands: {nbands}")
    # print(f"Data type: {dtype}")
    # -- Function to reproject an item’s asset onto the common grid --
    def process_item(item):
        asset_obj = item.assets[asset]
        print(asset_obj.href)

        with rasterio.open(asset_obj.href) as src:
            # print(f"Metadata for {asset_obj.href}:")
            print(src.meta)
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
    for item in items:
        # print(f"Processing item: {item.datetime}")
        # Create a delayed version of the processed image
        darr = delayed(process_item)(item)
        # Wrap with dask.array.from_delayed; shape is (nbands, height, width)
        darr = da.from_delayed(darr, shape=(nbands, height, width), dtype=dtype)
        delayed_arrays.append(darr)
        times.append(np.datetime64(item.datetime))  # use numpy datetime64 for coordinates

    # Stack along a new "time" axis (resulting shape: (time, band, y, x))
    data = da.stack(delayed_arrays, axis=0)
    # print(f"Stacked data shape: {data.shape}")
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
            "transform": transform.to_gdal()  # GDAL-style transform tuple
        }
    )

    computed_da = da_stack.compute(scheduler="threads")

    # print("DataArray created.")
    return computed_da  

def round_bounds(bounds, res):
    # Round each coordinate to the nearest multiple of 'res'
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / res) * res
    miny = math.floor(miny / res) * res
    maxx = math.ceil(maxx / res) * res
    maxy = math.ceil(maxy / res) * res
    return (minx, miny, maxx, maxy)



def create_data_stack(bbox, start_date, end_date, stac_source, collection, assets, out_resolution, bbox_crs="EPSG:4326", target_crs=None, resampling_method = Resampling.cubic):
    """
    Creates a data stack from a STAC API or local STAC catalog within a specified bounding box.

    If the collection is 'sentinel-1-rtc', a SAS token is fetched and appended to asset URLs for authenticated access.

    Parameters:
        bbox (tuple): Bounding box as (minx, miny, maxx, maxy) in bbox_crs units.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        stac_source (str): URL of the STAC API or path to a local STAC catalog file.
        collection (str): Collection name for the data.
        assets (list): List of assets to include in the stack.
        out_resolution (float): Ground sampling distance (GSD) in meters for the output resolution.
        bbox_crs (str, optional): CRS of the input bounding box. Default is "EPSG:4326".
        target_crs (str, optional): CRS for the resulting image. If not provided, inferred from the first item.
        resampling_method (rasterio.enums.Resampling, optional): Resampling method to use. Default is Resampling.cubic.

    Returns:
        xarray.DataArray: The computed data stack with resampled dimensions.
    """
    # Reproject bbox to EPSG:4326 if necessary for pystac_client search
    if bbox_crs != "EPSG:4326":
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)
        bbox_4326 = (minx, miny, maxx, maxy)
    else:
        bbox_4326 = bbox

    # Step 1: Open the STAC source (API or local catalog)
    if os.path.exists(stac_source):
        print("Opening local STAC catalog...")
        catalog = pystac.read_file(stac_source)
        items = []
        for item in catalog.get_all_items():
            if item.collection_id == collection:
                item_date = item.datetime.date()
                if start_date <= str(item_date) <= end_date:
                    items.append(item)
    else:
        print("Connecting to STAC API...")
        if "planetarycomputer" in stac_source:
            client = pystac_client.Client.open(
            stac_source,
            modifier=planetary_computer.sign_inplace
            )
        else:
            client = pystac_client.Client.open(stac_source)
        client = pystac_client.Client.open(
            stac_source,
            modifier=planetary_computer.sign_inplace)
        search = client.search(
            collections=[collection],
            datetime=f"{start_date}/{end_date}",
            bbox=bbox_4326,
            max_items=100,
        )
        all_items = search.get_all_items()

        # Reduce to one item per date. (Might want to remove this later)
        items = []
        dates = []
        for item in all_items:
            if item.datetime.date() not in dates:
                items.append(item)
                dates.append(item.datetime.date())

    if not items:
        raise ValueError("No items found for the specified parameters.")

    print(f"Found {len(items)} items in the collection '{collection}'")


    # Print original gsd for the first asset of the first item    
    with rasterio.open(items[0].assets[assets[0]].href) as src:
        # orig_width = src.width
        # orig_height = src.height
        
        orig_resolution = src.res
        print(f"Original '{assets[0]}' Image Resolution: {orig_resolution[0]} x {orig_resolution[1]} gsd.")# EPSG:{items[0].properties['proj:epsg']}")
        

    # Determine the target CRS if not provided
    if target_crs is None:
        if 'proj:epsg' in items[0].properties:
            target_crs = f"EPSG:{items[0].properties['proj:epsg']}"
        else:
            raise ValueError("No target_crs provided and no 'proj:epsg' property found in items.")
    print(f"Target CRS: {target_crs}")
    # Reproject the given bbox from bbox_crs to target_crs
    gdf = gpd.GeoDataFrame(
        geometry=[box(*bbox)],
        crs=bbox_crs
    )
    gdf_target = gdf.to_crs(target_crs)


# In your create_data_stack function, after computing reproj_bounds:
    reproj_bounds = gdf_target.geometry.iloc[0].bounds  # in target_crs
    reproj_bounds = round_bounds(reproj_bounds, out_resolution)
    print(f"Rounded Reprojected Bounding Box: {reproj_bounds}")

    # Retrieve the pixel values for the bounding box exactly as reprojected
    stack_data = stack(
        items,
        bounds=reproj_bounds,
        snap_bounds=False,  # do not snap to pixel boundaries
        epsg=int(target_crs.split(":")[1]),
        resolution=out_resolution,
        fill_value=0,
        assets=assets,
        resampling=resampling_method, #https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
        rescale=False
    )

    computed_stack = stack_data.compute()

    # # After stacking and resampling, print resampled dimensions
    print(f"Resampled Dimensions: {computed_stack.sizes.get('x', 'N/A')} x {computed_stack.sizes.get('y', 'N/A')} pixels (at {out_resolution}x{out_resolution} gsd). Stack {target_crs}")
    return computed_stack



def plot_data_stack_bounding_boxes(data_stacks, map_center=None, zoom_start=12):
    """
    Plots bounding boxes of data stacks on a Folium map.

    Parameters:
        data_stacks (list of xarray.DataArray): List of data stacks with bounding box metadata.
        map_center (tuple, optional): Latitude and longitude for centering the map. Default is None (auto-calculated).
        zoom_start (int, optional): Initial zoom level for the map. Default is 12.

    Returns:
        folium.Map: A Folium map with the bounding boxes plotted.
    """
    if not data_stacks:
        raise ValueError("The data_stacks list cannot be empty.")

    # Initialize map center if not provided
    if map_center is None:
        # Compute average center of all bounding boxes after converting to WGS84
        centers = []
        for stack in data_stacks:
            # Extract bounding box and CRS
            bbox = stack.attrs["spec"].bounds
            crs = stack.attrs["crs"].upper()  # Ensure CRS format is 'EPSG:XXXX'
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            x_min, y_min = transformer.transform(bbox[0], bbox[1])
            x_max, y_max = transformer.transform(bbox[2], bbox[3])
            centers.append(((y_min + y_max) / 2, (x_min + x_max) / 2))
        avg_lat = sum(lat for lat, lon in centers) / len(centers)
        avg_lon = sum(lon for lat, lon in centers) / len(centers)
        map_center = [avg_lat, avg_lon]

    # Create the map
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Add bounding boxes to the map
    for stack in data_stacks:
        # Extract bounding box and CRS
        bbox = stack.attrs["spec"].bounds
        crs = stack.attrs["crs"].upper()  # Ensure CRS format is 'EPSG:XXXX'
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x_min, y_min = transformer.transform(bbox[0], bbox[1])
        x_max, y_max = transformer.transform(bbox[2], bbox[3])

        # Add bounding box to map
        folium.Rectangle(
            bounds=[(y_min, x_min), (y_max, x_max)],
            color="blue",
            weight=2,
            fill=True,
            fill_color="blue",
            fill_opacity=0.2,
        ).add_to(m)

    return m

import numpy as np
import torch
import math
from box import Box
import yaml
from torchvision.transforms import Compose, Normalize


def prepare_data_stack(stack, platform, metadata_path, device, lat, lon):
    """
    Prepares a data stack for input into a model by normalizing pixel values,
    and generating embeddings for time, latitude, longitude, and other metadata.

    Parameters:
        stack (xarray.DataArray): The data stack generated from a STAC API query.
        platform (str): Name of the platform (e.g., 'sentinel-1-rtc').
        metadata_path (str): Path to the metadata YAML configuration file.
        device (torch.device): Device to which tensors should be moved (e.g., 'cuda' or 'cpu').
        lat (float): Latitude of the point of interest.
        lon (float): Longitude of the point of interest.

    Returns:
        dict: A dictionary containing prepared data and embeddings for model input.
    """
    # Load metadata
    metadata = Box(yaml.safe_load(open(metadata_path)))

    # Normalize bands using metadata
    mean, std, waves = [], [], []
    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    transform = Compose([Normalize(mean=mean, std=std)])

    # Normalize timestamps
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Normalize lat/lon
    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixel values
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare additional information
    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.resolution, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    return datacube




def create_and_prepare_data_stack(
    bbox, start, end, stac_source, collection, metadata_path, device, out_resolution, 
    bbox_crs="EPSG:4326", target_crs=None, assets=None, resampling_method=Resampling.cubic
):
    """
    Combines the creation and preparation of a data stack in one function.
    Calls `create_data_stack` and `prepare_data_stack` in sequence.

    Parameters:
        bbox (tuple): Bounding box as (minx, miny, maxx, maxy) in bbox_crs units.
        start, end (str): Date range for the data query.
        stac_source (str): URL of the STAC API or path to a local STAC catalog file.
        collection (str): Collection name for the data.
        metadata_path (str): Path to the metadata YAML configuration file.
        device (torch.device): Device to which tensors should be moved (e.g., 'cuda' or 'cpu').
        out_resolution (float): Desired output resolution in meters (GSD).
        bbox_crs (str, optional): CRS of the input bounding box. Default: "EPSG:4326".
        target_crs (str, optional): CRS for aligning bounding boxes (e.g., "EPSG:32610"). Default is None (uses native CRS).
        assets (list, optional): List of assets to include in the stack.
        resampling_method (Resampling, optional): Resampling method to use. Default is Resampling.cubic.

    Returns:
        dict: A dictionary containing prepared data and embeddings for model input.
    """
    # Step 1: Create the data stack
    stack = create_data_stack(
        bbox=bbox,
        start_date=start,
        end_date=end,
        stac_source=stac_source,
        collection=collection,
        assets=assets,
        out_resolution=out_resolution,
        bbox_crs=bbox_crs,
        target_crs=target_crs,
        resampling_method=resampling_method,
    )

    # Extract lat and lon as the top-right corner of the bounding box
    _, miny, maxx, maxy = bbox
    lat, lon = maxy, maxx  # Top-right corner

    # Step 2: Prepare the data stack for model input
    datacube = prepare_data_stack(
        stack=stack,
        platform=collection,
        metadata_path=metadata_path,
        device=device,
        lat=lat,
        lon=lon,
    )

    # Convert datetimes to a list of Python datetime objects
    original_times = stack.time.values.astype("datetime64[s]").tolist()

    # Add original timestamps to the datacube
    datacube["original_time"] = original_times

    return datacube

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# # Sentinel-1 Stack
# sentinel1_cube = create_and_prepare_data_stack(
#     bbox = bounds
#     , start = start_end[0]
#     , end = start_end[1]
#     ,stac_source="https://planetarycomputer.microsoft.com/api/stac/v1"
#     ,collection="sentinel-1-rtc"
#     ,assets=["vv", "vh"]
#     , bbox_crs="EPSG:32610"
#     , target_crs="EPSG:32610"
#     , out_resolution = 10
#     , resampling_method = Resampling.cubic
#     , metadata_path = "/home/jovyan/geoai_veg_map/clay_model/configs/metadata.yaml"
#     , device = device
# )

# sentinel2_cube = create_and_prepare_data_stack(
#     bbox = bounds
#     , start = start_end[0]
#     , end = start_end[1]
#     ,stac_source="https://earth-search.aws.element84.com/v1"
#     ,collection="sentinel-2-l2a"
#     ,assets=["blue", "green", "red", "nir"]
#     , bbox_crs="EPSG:32610"
#     , target_crs="EPSG:32610"
#     , out_resolution = 10
#     , resampling_method = Resampling.cubic
#     , metadata_path = "/home/jovyan/geoai_veg_map/clay_model/configs/metadata.yaml"
#     , device = device
# )



# # Landsat Datacube
# landsat_cube = create_and_prepare_data_stack(
#     lat=poi_latlon[0],
#     lon=poi_latlon[1],
#     start='2024-06-14',
#     end=start_end[1],
#     stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
#     collection="landsat-c2-l2",
#     size=256,
#     gsd=10,
#     assets=["blue", "green", "red", "nir08"],
#     target_crs=common_crs,  # Force alignment to common CRS
#     metadata_path = "/home/jovyan/geoai_veg_map/clay_model/configs/metadata.yaml",
#     device = device
# )
# landsat_cube