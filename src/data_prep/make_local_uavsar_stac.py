#!/usr/bin/env python
"""
This script accepts one or more bounding boxes (each specified as 4 floats: minLon, minLat, maxLon, maxLat),
a start date, an end date, and an output directory. For each bounding box, it:
  - Queries the ASF UAVSAR API for scenes,
  - Downloads and extracts TIFF files (via UavsarScene) into a temporary folder,
  - For each scene, crops each polarization GeoTIFF to the input bounding box,
    converting the cropped image to a Cloud Optimized GeoTIFF (COG),
  - Uses the cropped HHHH image to update the scene’s geometry/bbox, and
  - Creates (or updates) a local STAC catalog with a unique item per scene per bbox.
  
Authentication is handled using the environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD.
The .netrc file is automatically created (if it doesn’t exist) in the required format.

Script Arguments: 
--bbox <minLon> <minLat> <maxLon> <maxLat> One or more bounding boxes defined by four float values in EPSG:4326. 
        Multiple --bbox arguments can be provided to process different areas. 
--start <YYYY-MM-DD> The start date for the search timeframe. 
--end <YYYY-MM-DD> The end date for the search timeframe. 
--output <output_directory> Directory to store the final STAC catalog and the processed COG files. 
--temp <temporary_directory> Temporary directory to store raw UAVSAR downloads and TIFF files.

Example: python src/data_prep/make_local_uavsar_stac.py
--bbox -116.674113 33.096041 -116.567707 33.170647
--bbox -120.127900 34.649349 -119.938771 34.775782
--start 2014-01-01
--end 2024-12-31
--output /home/jovyan/geoai_veg_map/data/stac/uavsar/
--temp data/raw/uavsar
"""

import os
import argparse
import subprocess
import logging
from datetime import datetime

import rasterio
import rioxarray
import pyproj
from shapely.geometry import box, mapping
from shapely.ops import transform

import pystac
from pystac import Catalog, CatalogType, Item, Asset, Collection, SpatialExtent, TemporalExtent, Extent

import asf_search as asf
from uavsar_pytools.uavsar_tools import create_netrc  # for netrc generation
from uavsar_pytools import UavsarScene

# Configure logging
logging.basicConfig(level=print, format="%(levelname)s: %(message)s")

# Fixed polarization keys (order is fixed)
REQUIRED_POLS = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]

def load_or_create_catalog(output_dir):
    """
    If a catalog already exists in the output directory (catalog.json), load and return it.
    Otherwise, create and return a new STAC catalog.
    """
    catalog_file = os.path.join(output_dir, "catalog.json")
    if os.path.exists(catalog_file):
        print(f"Existing catalog found; loading catalog from {catalog_file}")
        return pystac.read_file(catalog_file)
    else:
        print("No existing catalog found; creating a new catalog.")
        return Catalog(
            id="uavsar-catalog",
            description="Local STAC catalog for cropped UAVSAR COGs",
            title="UAVSAR Imagery Catalog"
        )

def ensure_netrc():
    """
    Ensure a .netrc file exists for UAVSAR authentication.
    If not found, attempt to create one using EARTHDATA_USERNAME and EARTHDATA_PASSWORD
    from the environment. The file is created with the following format:
    
    machine urs.earthdata.nasa.gov
    login <username>
    password <password>
    """
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        username = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if username and password:
            with open(netrc_path, "w") as f:
                f.write("machine urs.earthdata.nasa.gov\n")
                f.write(f"login {username}\n")
                f.write(f"password {password}\n")
            os.chmod(netrc_path, 0o600)
            print(f"Created {netrc_path} using EARTHDATA_USERNAME and EARTHDATA_PASSWORD.")
        else:
            logging.error("No .netrc file found and EARTHDATA_USERNAME/EARTHDATA_PASSWORD are not set. "
                          "Please set them or run 'create_netrc()' manually.")
            exit(1)
    else:
        print(f".netrc file found at {netrc_path}.")

def bbox_to_wkt(bbox_str):
    """
    Convert a comma-separated bounding box string (minLon,minLat,maxLon,maxLat) to a WKT polygon.
    """
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_str.split(','))
    return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

def convert_to_cog(input_tif, output_tif):
    """
    Convert a GeoTIFF to a Cloud Optimized GeoTIFF (COG) using gdal_translate.
    """
    num_threads = os.cpu_count() - 2
    if num_threads < 1:
        num_threads = 1

    cmd = [
        "gdal_translate",
        "-of", "COG",
        "-r", "NEAREST",  # Use NEAREST resampling for complex data
        "-co", "COMPRESS=LZW",
        "-co", f"NUM_THREADS={num_threads}",
        "-co", "OVERVIEW_RESAMPLING=NEAREST",
        input_tif,
        output_tif,
    ]
    print("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_tif

def crop_to_bbox(input_tif, bbox_geom, temp_cropped_path):
    """
    Opens the input_tif with rioxarray, crops it to the provided bbox_geom (a shapely geometry in EPSG:4326),
    and writes the cropped result to temp_cropped_path.
    Returns the cropped dataset bounds.
    """
    ds = rioxarray.open_rasterio(input_tif)
    dataset_crs = ds.rio.crs
    # If dataset is not in EPSG:4326, transform bbox_geom from EPSG:4326 to dataset CRS.
    if dataset_crs.to_string() != "EPSG:4326":
        transformer = pyproj.Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
        transformed_bbox = transform(transformer.transform, bbox_geom)
    else:
        transformed_bbox = bbox_geom

    cropped_ds = ds.rio.clip([mapping(transformed_bbox)], crs=dataset_crs, drop=True)
    # Write the cropped dataset to disk (as an intermediate file).
    cropped_ds.rio.to_raster(temp_cropped_path, compress="deflate", tiled=True)
    bounds = cropped_ds.rio.bounds()
    return bounds

def search_and_download(aoi_bbox_str, start_date, end_date, temp_dir, session):
    """
    For the given bounding box string, search for UAVSAR scenes using asf_search,
    and for each returned GRD zip URL, extract TIFF files via UavsarScene into temp_dir.
    """
    wkt = bbox_to_wkt(aoi_bbox_str)
    print(f"Searching for UAVSAR scenes in area: {wkt}")
    results = asf.geo_search(
        platform=[asf.PLATFORM.UAVSAR],
        intersectsWith=wkt,
        start=start_date,
        end=end_date,
        #
        beamMode=asf.BEAMMODE.POL, #https://github.com/asfadmin/Discovery-asf_search/blob/master/asf_search/constants/BEAMMODE.py
        processingLevel=asf.PRODUCT_TYPE.PROJECTED, #see https://github.com/asfadmin/Discovery-asf_search/blob/master/asf_search/constants/PRODUCT_TYPE.py
        maxResults=300  # adjust if needed
    )
    results = list(results)
    if not results:
        print("No UAVSAR scenes found for this bounding box.")
        return

    print(f"Found {len(results)} UAVSAR scene(s) for this area.")

    grd_urls = [res.properties.get('url') for res in results]

    # Process GRD files (unzipping as before):
    if grd_urls:
        print(f"Starting download: {len(grd_urls)} GRD file(s) will be processed.")
        for i, zip_url in enumerate(grd_urls, 1):
            print(f"Processing GRD file {i} of {len(grd_urls)}: {zip_url}")
            print(f"Processing GRD URL: {zip_url}")
            scene = UavsarScene(url=zip_url, work_dir=temp_dir)
            try:
                scene.url_to_tiffs()
            except Exception as e:
                logging.error(f"Error processing {zip_url}: {e}")
                



def process_scene_directories(temp_dir, output_dir, input_bbox_str, catalog):
    """
    Process each scene directory in temp_dir. For each scene:
      - Crop the asset(s) to the input bounding box (given as a comma-separated string),
      - Use the cropped HHHH asset to set the item's geometry and bbox,
      - For each required polarization, crop and convert to a COG,
      - Create a STAC item with a unique ID (sceneID_bbox) and add assets,
      - Only add the item if it does not already exist in the catalog.
    """
    # Parse the input bounding box string into a shapely geometry (EPSG:4326)
    min_lon, min_lat, max_lon, max_lat = map(float, input_bbox_str.split(','))
    aoi_polygon = box(min_lon, min_lat, max_lon, max_lat)
    # Create a simple identifier for the bbox to append to item IDs.
    bbox_identifier = input_bbox_str.replace(',', '_')

    for scene_dir in os.listdir(temp_dir):
        scene_path = os.path.join(temp_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue

        # Create a unique item ID for this scene and bbox.
        item_id = f"{scene_dir}_{bbox_identifier}"

        # Attempt to parse acquisition date from the directory name.
        tokens = scene_dir.split('_')
        acq_datetime = None
        if len(tokens) >= 5 and len(tokens[4]) == 6:
            date_str = tokens[4]
            try:
                year = "20" + date_str[:2]
                month = date_str[2:4]
                day = date_str[4:6]
                acq_datetime_str = f"{year}-{month}-{day}T00:00:00Z"
                acq_datetime = datetime.strptime(acq_datetime_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception as e:
                logging.error(f"Error parsing date for {item_id}: {e}")
        if acq_datetime is None:
            logging.warning(f"No acquisition date found for {item_id}. Using current datetime as fallback.")
            acq_datetime = datetime.now()

        # Use the HHHH asset for geometry: find a TIFF file with 'hhhh' in its name.
        hhhh_file = None
        for file in os.listdir(scene_path):
            if file.lower().endswith('.tiff') and "hhhh" in file.lower():
                hhhh_file = os.path.join(scene_path, file)
                break
        if not hhhh_file:
            logging.warning(f"Skipping {item_id} because no GeoTIFF with 'HHHH' polarization was found for geometry.")
            continue

        # Crop the HHHH asset to the input bounding box and get its new bounds.
        temp_cropped_hhhh = os.path.join(output_dir, f"temp_cropped_{scene_dir}_HHHH.tif")
        try:
            new_bounds = crop_to_bbox(hhhh_file, aoi_polygon, temp_cropped_hhhh)
        except Exception as e:
            logging.error(f"Error cropping {hhhh_file}: {e}")
            continue

        # Create new geometry and bbox from the cropped bounds.
        cropped_box = box(*new_bounds)
        new_geom = mapping(cropped_box)
        new_bbox = list(cropped_box.bounds)
        # Remove temporary cropped HHHH file.
        if os.path.exists(temp_cropped_hhhh):
            os.remove(temp_cropped_hhhh)

        # Create a STAC Item.
        item = Item(
            id=item_id,
            geometry=new_geom,
            bbox=new_bbox,
            datetime=acq_datetime,
            properties={}
        )
        # Create a Collection for the item.
        spatial_extent = SpatialExtent(bboxes=[new_bbox])
        temporal_extent = TemporalExtent(intervals=[[acq_datetime.isoformat(), acq_datetime.isoformat()]])
        extent = Extent(spatial=spatial_extent, temporal=temporal_extent)
        collection = Collection(
            id="uavsar",
            description="UAVSAR COG collection",
            extent=extent
        )
        item.collection = collection

        # Process each required polarization asset.
        for pol in REQUIRED_POLS:
            asset_file = None
            for file in os.listdir(scene_path):
                if file.lower().endswith('.tiff') and pol.lower() in file.lower():
                    asset_file = os.path.join(scene_path, file)
                    break
            if asset_file:
                # Crop the asset to the input bounding box.
                temp_cropped = os.path.join(output_dir, f"temp_cropped_{scene_dir}_{pol}.tif")
                try:
                    crop_to_bbox(asset_file, aoi_polygon, temp_cropped)
                except Exception as e:
                    logging.error(f"Error cropping {asset_file}: {e}")
                    continue
                # Define final output COG path.
                output_filename = f"{item_id}_{pol}_COG.tiff"
                final_output = os.path.join(output_dir, output_filename)
                try:
                    convert_to_cog(temp_cropped, final_output)
                except Exception as e:
                    logging.error(f"Error converting {temp_cropped} to COG: {e}")
                    if os.path.exists(temp_cropped):
                        os.remove(temp_cropped)
                    continue
                # Clean up temporary cropped file.
                if os.path.exists(temp_cropped):
                    os.remove(temp_cropped)
                asset = Asset(
                    href=final_output,
                    media_type=pystac.MediaType.GEOTIFF,
                    roles=["data"],
                    title=f"{item_id} {pol} image (COG)"
                )
                item.add_asset(pol, asset)
            else:
                logging.warning(f"{item_id} does not contain an asset for polarization {pol}.")

        # Only add the item if it doesn't already exist in the catalog.
        if catalog.get_item(item.id) is None:
            catalog.add_item(item)
            print(f"Added STAC item {item.id} to catalog.")
        else:
            print(f"Item {item.id} already exists in catalog; skipping addition.")

    return catalog

def main():
    parser = argparse.ArgumentParser(
        description="Query UAVSAR imagery for bounding boxes and a date range, download, crop to each bbox, convert to COGs, and build (or append to) a local STAC catalog."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("minLon", "minLat", "maxLon", "maxLat"),
        action="append",
        required=True,
        help="Bounding box coordinates in EPSG:4326. Use multiple --bbox for multiple areas."
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD) for the search timeframe."
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD) for the search timeframe."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to store the final STAC catalog and COG files."
    )
    parser.add_argument(
        "--temp",
        type=str,
        default="data/raw/uavsar",
        help="Temporary base directory to store raw UAVSAR downloads and TIFFs."
    )
    args = parser.parse_args()

    # Ensure output and temporary base directories exist.
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.temp, exist_ok=True)

    # Ensure .netrc file exists for UAVSAR API authentication.
    ensure_netrc()

    # Set up ASF session using environment variables (EARTHDATA_USERNAME and EARTHDATA_PASSWORD).
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        logging.error("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set for authentication.")
        exit(1)
    session = asf.ASFSession()
    session.auth_with_creds(username, password)

    # Load or create the STAC catalog from the output directory.
    catalog = load_or_create_catalog(args.output)

    # Process each bounding box individually.
    bboxes = args.bbox  # each is a list of 4 floats
    for i, bbox in enumerate(bboxes):
        # Convert the bbox list into a comma-separated string.
        bbox_str = ",".join(map(str, bbox))
        print(f"Processing bounding box: {bbox_str}")
        # Create a dedicated temporary subdirectory for this bbox.
        temp_subdir = os.path.join(args.temp, f"bbox_{i}")
        os.makedirs(temp_subdir, exist_ok=True)
        # Search and download scenes for this bounding box.
        search_and_download(bbox_str, args.start, args.end, temp_subdir, session)
        # Process the downloaded scenes (crop to the bbox, convert to COG, add to catalog).
        catalog = process_scene_directories(temp_subdir, args.output, bbox_str, catalog)

    # Save the final STAC catalog (self-contained) in the output directory.
    catalog.normalize_and_save(root_href=args.output, catalog_type=CatalogType.SELF_CONTAINED)
    print(f"Local STAC catalog saved to {args.output}")

if __name__ == "__main__":
    main()
