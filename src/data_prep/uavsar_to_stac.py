#!/usr/bin/env python3
"""
uavsar_to_stac.py

This script creates a STAC catalog for UAVSAR products.
It assumes that each product is in its own subdirectory under
~/geoai_veg_map/UAVSAR/images, and that each product directory
contains the original GeoTIFFs. The corresponding Cloud Optimized
GeoTIFFs (COGs) for each polarization (HHHH, HHHV, VVVV, HVVV, HVHV, HHVV)
have been generated and stored in a subdirectory named "COG" under
the UAVSAR images directory.
"""

import os
from datetime import datetime
from pathlib import Path

import rasterio
from shapely.geometry import box, mapping
import geopandas as gpd

import pystac
from pystac import (
    Catalog,
    Collection,
    Item,
    Asset,
    MediaType,
    CatalogType,
    Extent,
    SpatialExtent,
    TemporalExtent,
)


# List of required polarization asset keys (in desired order)
REQUIRED_POLS = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]

# Directories
ROOT_DIR = os.path.expanduser('~/geoai_veg_map/UAVSAR/images')
COG_DIR = os.path.join(ROOT_DIR, "COG")
os.makedirs(COG_DIR, exist_ok=True)


def create_stac_catalog(catalog_id, collection_id, collection_description):
    """Create a STAC Catalog and a child Collection."""
    catalog = Catalog(id=catalog_id, description=f"Local {catalog_id} Catalog")
    # Initialize collection with a dummy extent; will be updated later.
    dummy_extent = Extent(
        spatial=SpatialExtent([[0, 0, 0, 0]]),
        temporal=TemporalExtent(intervals=[[datetime.utcnow(), datetime.utcnow()]])
    )
    collection = Collection(
        id=collection_id,
        description=collection_description,
        extent=dummy_extent,
        license="proprietary"
    )
    catalog.add_child(collection)
    return catalog, collection


def update_collection_extent(collection, all_bboxes, all_times):
    """Update the collection's spatial and temporal extents based on processed items."""
    if all_bboxes and all_times:
        minx = min(b[0] for b in all_bboxes)
        miny = min(b[1] for b in all_bboxes)
        maxx = max(b[2] for b in all_bboxes)
        maxy = max(b[3] for b in all_bboxes)
        temporal_start = min(all_times)
        temporal_end = max(all_times)
        collection.extent = Extent(
            spatial=SpatialExtent([[minx, miny, maxx, maxy]]),
            temporal=TemporalExtent(intervals=[[temporal_start, temporal_end]])
        )
    else:
        now = datetime.utcnow()
        collection.extent = Extent(
            spatial=SpatialExtent([[0, 0, 0, 0]]),
            temporal=TemporalExtent(intervals=[[now, now]])
        )


import re
from datetime import datetime
import os
import rasterio
from shapely.geometry import box, mapping
from pystac import Item, Asset, MediaType

def process_uav_product(product_dir, root_dir, cog_dir, required_pols, exclude_patterns=['^COG$',"_ML3X3","ML5X5"]):
    """
    Process one UAVSAR product directory:
      - Parse acquisition datetime from the directory name.
      - Extract geometry (bounding box) from a GeoTIFF containing "HHHH".
      - Create a STAC Item and add COG assets for each required polarization,
        excluding assets whose filenames match any regex in exclude_patterns.
    Returns:
      (item, bbox, acq_datetime) or (None, None, None) if geometry cannot be determined.
    """
    item_path = os.path.join(root_dir, product_dir)
    item_id = product_dir

    # Parse acquisition date from directory name.
    tokens = product_dir.split('_')
    if len(tokens) >= 5 and len(tokens[4]) == 6:
        date_str = tokens[4]
        year = "20" + date_str[:2]
        month = date_str[2:4]
        day = date_str[4:6]
        acq_datetime_str = f"{year}-{month}-{day}T00:00:00Z"
        try:
            acq_datetime = datetime.strptime(acq_datetime_str, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(f"Error parsing date for {item_id}: {e}")
            acq_datetime = datetime.now()
    else:
        print(f"No valid acquisition date in {item_id}; using current datetime as fallback.")
        acq_datetime = datetime.now()

    # Extract geometry from a GeoTIFF file with "HHHH"
    geometry = None
    bbox = None
    for file in os.listdir(item_path):
        if file.lower().endswith('.tiff') and "hhhh" in file.lower():
            tiff_path = os.path.join(item_path, file)
            try:
                with rasterio.open(tiff_path) as src:
                    bounds = src.bounds  # (left, bottom, right, top)
                    geometry = mapping(box(bounds.left, bounds.bottom, bounds.right, bounds.top))
                    bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            except Exception as e:
                print(f"Error reading geometry from {tiff_path}: {e}")
            break

    if geometry is None:
        print(f"Skipping {item_id}: unable to extract geometry.")
        return None, None, None

    # Create the STAC Item.
    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=acq_datetime,
        properties={}
    )

    # For each required polarization, add a corresponding COG asset.
    # Expected COG file name: {product_dir}_{pol}_COG.tiff in the COG directory.
    for pol in required_pols:
        cog_filename = f"{product_dir}_{pol}_COG.tiff"
        
        # If exclude_patterns is provided (a list of regex patterns), skip asset if any pattern matches.
        if exclude_patterns is not None:
            matched = False
            for pattern in exclude_patterns:
                if re.search(pattern, cog_filename, re.IGNORECASE):
                    print(f"Excluding asset {cog_filename} because it matches pattern '{pattern}'.")
                    matched = True
                    break
            if matched:
                continue

        cog_path = os.path.join(cog_dir, cog_filename)
        if os.path.exists(cog_path):
            asset = Asset(
                href=os.path.abspath(cog_path),  # use absolute path
                media_type=MediaType.GEOTIFF,
                roles=["data"],
                title=f"{pol} polarization"
            )
            item.add_asset(pol, asset)
        else:
            print(f"Warning: {item_id} missing asset for polarization {pol} (expected {cog_filename}).")
    return item, bbox, acq_datetime




def main():
    stac_output = os.path.abspath("uavsar_stac")
    os.makedirs(stac_output, exist_ok=True)

    # Create the catalog and collection.
    catalog, collection = create_stac_catalog(
        catalog_id="uavsar-catalog",
        collection_id="uavsar",
        collection_description="UAVSAR COG Imagery Collection"
    )

    items = []
    all_bboxes = []
    all_times = []

    # Process each product directory under ROOT_DIR.
    for product_dir in os.listdir(ROOT_DIR):
        product_path = os.path.join(ROOT_DIR, product_dir)
        if not os.path.isdir(product_path):
            continue
        item, bbox, acq_datetime = process_uav_product(product_dir, ROOT_DIR, COG_DIR, REQUIRED_POLS)
        if item is not None:
            collection.add_item(item)
            items.append(item)
            all_bboxes.append(bbox)
            all_times.append(acq_datetime)

    if not items:
        raise ValueError("No UAVSAR items were added to the catalog.")

    update_collection_extent(collection, all_bboxes, all_times)

    # Normalize HREFs and save the catalog.
    catalog.normalize_hrefs(stac_output)
    catalog.save(catalog_type=CatalogType.SELF_CONTAINED)
    print(f"STAC catalog saved to {stac_output}")


if __name__ == '__main__':
    main()
