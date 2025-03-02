#!/usr/bin/env python
"""
This script takes one or more bounding boxes (in EPSG:4326) and a date range,
queries the Planetary Computer STAC API for NAIP imagery,
downloads each full-resolution cloud optimized GeoTIFF (all bands together) for every item found,
crops each image to the original bounding box,
and then creates (or appends to) a local STAC catalog referencing the downloaded files.
The saved STAC item geometry and bbox reflect the cropped image.
"""

import argparse
import os
import time
import logging
from shapely.geometry import box, shape, mapping
from shapely.ops import transform
import pystac_client
import planetary_computer
import rioxarray
import pystac
from pystac import Catalog, CatalogType, Item, Asset
import pyproj

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_or_create_catalog(output_dir):
    """
    Loads an existing STAC catalog from the output directory if it exists,
    otherwise creates a new STAC catalog.
    """
    catalog_file = os.path.join(output_dir, "catalog.json")
    if os.path.exists(catalog_file):
        logger.info(f"Existing catalog found; loading catalog from {catalog_file}")
        return pystac.read_file(catalog_file)
    else:
        logger.info("No existing catalog found; creating a new catalog.")
        return Catalog(
            id="naip_local_catalog",
            description="Local STAC catalog for NAIP imagery with full-resolution COGs cropped to the AOI",
            title="NAIP Local Catalog"
        )

def process_item(item, aoi_polygon, output_dir, stac_catalog, retry_count=3, retry_delay=5):
    """
    Process a single STAC item: download, crop, and add to the catalog.
    Includes retry logic for resilience against temporary failures.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"\nProcessing item: {item.id}")
    
    # Skip if this item already exists in the catalog
    if stac_catalog.get_item(item.id) is not None:
        logger.info(f"Item {item.id} already exists in the catalog; skipping processing.")
        return True
    
    # Define output file path
    out_file = os.path.join(output_dir, f"{item.id}.tif")
    
    # Check if output file already exists
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        logger.info(f"Output file {out_file} already exists; using existing file.")
        # Create a STAC item for the existing file
        try:
            # Open the existing file to get its bounds
            ds = rioxarray.open_rasterio(out_file)
            dataset_crs = ds.rio.crs
            bounds = ds.rio.bounds()
            bounds_box = box(*bounds)
            
            # Transform bounds to EPSG:4326 if necessary
            if dataset_crs.to_string() != "EPSG:4326":
                transformer = pyproj.Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)
                bounds_box = transform(transformer.transform, bounds_box)
            
            new_geom = mapping(bounds_box)
            new_bbox = list(bounds_box.bounds)
            
            stac_item = Item(
                id=item.id,
                geometry=new_geom,
                bbox=new_bbox,
                datetime=item.datetime,
                properties=item.properties,
            )
            
            asset = Asset(
                href=out_file,
                media_type="image/tiff",
                roles=["data"],
                title="NAIP Cropped Image"
            )
            stac_item.add_asset(key="image", asset=asset)
            stac_catalog.add_item(stac_item)
            logger.info(f"Item {stac_item.id} added to catalog from existing file.")
            return True
        except Exception as e:
            logger.warning(f"Error processing existing file {out_file}: {str(e)}")
            # Continue with download attempt if we can't use the existing file
    
    # Attempt to download and process with retries
    for attempt in range(retry_count):
        try:
            # Get the signed URL for the raw 'image' asset
            image_href = item.assets["image"].href
            
            # Open the cloud-optimized GeoTIFF using rioxarray
            ds = rioxarray.open_rasterio(image_href)
            logger.info(f"Dataset dimensions: {ds.dims} | Bands: {ds.band.values}")
            
            # --- Crop the dataset to the original bounding box ---
            # If the dataset is not in EPSG:4326, transform the AOI polygon
            dataset_crs = ds.rio.crs
            if dataset_crs.to_string() != "EPSG:4326":
                transformer = pyproj.Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
                transformed_aoi = transform(transformer.transform, aoi_polygon)
            else:
                transformed_aoi = aoi_polygon

            cropped_ds = ds.rio.clip([transformed_aoi], crs=dataset_crs, drop=True)
            
            # --- Determine new geometry and bbox from cropped dataset ---
            # Get bounds of the cropped dataset (in its CRS)
            cropped_bounds = cropped_ds.rio.bounds()  # (minx, miny, maxx, maxy)
            cropped_box = box(*cropped_bounds)
            # Transform the cropped bounds back to EPSG:4326 if necessary
            if dataset_crs.to_string() != "EPSG:4326":
                transformer_back = pyproj.Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)
                cropped_box = transform(transformer_back.transform, cropped_box)
            new_geom = mapping(cropped_box)
            new_bbox = list(cropped_box.bounds)
            
            # Write one single GeoTIFF (with all bands) that is cropped to the AOI
            cropped_ds.rio.to_raster(out_file, compress="deflate", tiled=True)
            logger.info(f"Written cropped image to {out_file}")
            
            # Create a STAC item using the cropped geometry and bbox
            stac_item = Item(
                id=item.id,
                geometry=new_geom,
                bbox=new_bbox,
                datetime=item.datetime,
                properties=item.properties,
            )
            
            # Add the full (cropped) image as an asset
            asset = Asset(
                href=out_file,
                media_type="image/tiff",
                roles=["data"],
                title="NAIP Cropped Image"
            )
            stac_item.add_asset(key="image", asset=asset)
            
            # Add item to catalog
            stac_catalog.add_item(stac_item)
            logger.info(f"Item {stac_item.id} added to catalog.")
            
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{retry_count} failed for item {item.id}: {str(e)}")
            
            # If this is not the last attempt, wait before retrying
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to process item {item.id} after {retry_count} attempts.")
                return False

def process_bbox(aoi_coords, date_range, output_dir, stac_catalog, catalog_client):
    """
    Processes a single bounding box by:
      - Constructing the AOI polygon (in EPSG:4326).
      - Searching for NAIP imagery within the AOI and specified date range.
      - Iterating over every returned item.
      - Downloading a single cloud-optimized GeoTIFF (all bands) for each item.
      - Cropping the image to the original bounding box.
      - Updating the STAC item geometry and bbox to reflect the cropped image.
      - Adding the new STAC item to the local catalog.
    """
    # Construct the AOI polygon in EPSG:4326.
    aoi_polygon = box(aoi_coords[0], aoi_coords[1], aoi_coords[2], aoi_coords[3])
    logger.info(f"\nSearching for NAIP imagery intersecting bounding box {aoi_coords} and date range {date_range}...")

    # Query the API for NAIP imagery intersecting the AOI and date range.
    search = catalog_client.search(
        collections=["naip"],
        intersects={
            "type": "Polygon",
            "coordinates": [list(aoi_polygon.exterior.coords)]
        },
        datetime=date_range,
        limit=100  # adjust if needed
    )
    items = list(search.items())
    if not items:
        logger.info("No NAIP imagery found for this bounding box and date range.")
        return

    # Process each returned item, continuing even if some fail
    successful_items = 0
    failed_items = 0
    
    for item in items:
        if process_item(item, aoi_polygon, output_dir, stac_catalog):
            successful_items += 1
        else:
            failed_items += 1
    
    logger.info(f"Processed {len(items)} items: {successful_items} successful, {failed_items} failed.")

def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Query NAIP imagery for one or more bounding boxes and a date range, "
                    "and build (or append to) a local STAC catalog with full-resolution COGs cropped to the AOI."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("minx", "miny", "maxx", "maxy"),
        required=True,
        action="append",
        help="Bounding box coordinates (minx miny maxx maxy) in EPSG:4326. Use multiple --bbox arguments for multiple areas."
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD) for the date range."
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD) for the date range."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/jovyan/geoai_veg_map/data/raw/naip/",
        help="Directory to write the local STAC catalog and GeoTIFFs."
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retry attempts for failed downloads."
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay in seconds between retry attempts."
    )
    args = parser.parse_args()

    # Ensure output directory exists.
    os.makedirs(args.output, exist_ok=True)

    # Create the ISO8601 date range string.
    date_range = f"{args.start}/{args.end}"

    # Open the Planetary Computer STAC API client.
    catalog_client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Load an existing catalog or create a new one.
    stac_catalog = load_or_create_catalog(args.output)

    # Process each bounding box.
    for bbox_coords in args.bbox:
        process_bbox(bbox_coords, date_range, args.output, stac_catalog, catalog_client)

    # Save the updated catalog.
    stac_catalog.normalize_and_save(args.output, catalog_type=CatalogType.SELF_CONTAINED)
    logger.info(f"\nLocal STAC catalog saved to directory: {args.output}")

if __name__ == "__main__":
    main()