#!/usr/bin/env python
"""
This script takes one or more bounding boxes and a date range, queries the Planetary Computer STAC API for
3DEP Lidar COPC point clouds, and processes each returned item individually as follows:
  - For each bounding box, it queries for items intersecting the area.
  - For each item that contains a point cloud asset under the key "data", it builds a PDAL pipeline that:
      * Reads the asset,
      * Crops it to the provided bounding box,
      * Optionally reprojects it,
      * Writes the processed point cloud to a separate local COPC file.
  - A new STAC item referencing each output COPC file is created (or appended to an existing local catalog).
  
Usage:
  python this_script.py --bbox minx miny maxx maxy [--bbox ...] --start YYYY-MM-DD --end YYYY-MM-DD [--output /path/to/output] [--threads N] [--target-crs EPSG:XXXX]
"""

import argparse
import os
import json
import gc
from datetime import datetime

import pdal
import numpy as np
from shapely.geometry import mapping, box

import pystac
import pystac_client
import planetary_computer
from pystac import Catalog, CatalogType, Item, Asset


def bounding_box_to_geojson(bbox):
    """Convert [minx, miny, maxx, maxy] to a GeoJSON Polygon."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
        ]]
    }


def load_or_create_catalog(output_dir):
    """
    Loads an existing STAC catalog from the output directory if it exists,
    otherwise creates a new one.
    """
    catalog_file = os.path.join(output_dir, "catalog.json")
    if os.path.exists(catalog_file):
        print("Existing catalog found; loading catalog from", catalog_file)
        return pystac.read_file(catalog_file)
    else:
        print("No existing catalog found; creating a new catalog.")
        return Catalog(
            id="3dep_lidar_copc_local_catalog",
            description="Local STAC catalog for processed 3DEP Lidar COPC point clouds",
            title="3DEP Lidar COPC Local Catalog"
        )


def process_bbox(bbox, date_range, output_dir, stac_catalog, catalog_client, threads, target_crs):
    """
    For a given bounding box and date range:
      - Query the Planetary Computer for COPC point cloud items intersecting the area.
      - For each item that has a "data" asset, build and execute a PDAL pipeline that:
            * Reads the asset,
            * Crops to the bounding box,
            * Optionally reprojects to a target CRS,
            * Writes the processed point cloud to a local COPC file.
      - Create a new STAC item referencing the processed file.
    """
    # Create GeoJSON polygon from bbox.
    polygon = bounding_box_to_geojson(bbox)
    print(f"\nSearching for 3DEP Lidar COPC point clouds intersecting bbox {bbox} and date range {date_range}...")
    
    # Query the STAC API.
    search = catalog_client.search(
        collections=["3dep-lidar-copc"],
        intersects=polygon,
        datetime=date_range,
        limit=100
    )
    items = list(search.items())
    if not items:
        print("No 3DEP Lidar COPC point clouds found for this bbox and date range.")
        return

    # Process each item separately.
    for item in items:
        if 'data' not in item.assets:
            print(f"Item {item.id} does not have a 'data' asset; skipping.")
            continue
        
        asset_url = item.assets['data'].href
        print(f"\nProcessing item {item.id} with asset: {asset_url}")
        
        # Build the PDAL pipeline for this item.
        pipeline_stages = [
            {
                "type": "readers.copc",
                "filename": asset_url,
                "threads": threads,
                "polygon": json.dumps(polygon)
            },
            {
                "type": "filters.crop",
                "polygon": json.dumps(polygon)
            }
        ]
        
        if target_crs:
            pipeline_stages.append({
                "type": "filters.reprojection",
                "out_srs": target_crs
            })
        
        # Define the output filename using the item ID.
        output_filename = os.path.join(output_dir, f"{item.id}.copc")
        pipeline_stages.append({
            "type": "writers.copc",
            "filename": output_filename
        })
        
        pipeline_dict = {"pipeline": pipeline_stages}
        pipeline_json = json.dumps(pipeline_dict, indent=2)
        print("PDAL pipeline:")
        print(pipeline_json)
        
        try:
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            print(f"Processed COPC file written to {output_filename}")
        except Exception as e:
            print(f"Error processing point cloud for item {item.id}: {e}")
            continue
        finally:
            del pipeline
            gc.collect()
        
        # Create a new STAC item for the processed COPC file.
        stac_item = Item(
            id=item.id,
            geometry=item.geometry,
            bbox=item.bbox,
            datetime=item.datetime,
            properties=item.properties,
        )
        
        asset = Asset(
            href=output_filename,
            media_type="application/octet-stream",
            roles=["data"],
            title="Processed 3DEP Lidar COPC Point Cloud"
        )
        stac_item.add_asset("data", asset)
        
        if stac_catalog.get_item(stac_item.id) is None:
            stac_catalog.add_item(stac_item)
            print(f"STAC item {stac_item.id} added to catalog.")
        else:
            print(f"STAC item {stac_item.id} already exists in the catalog; skipping addition.")


def main():
    parser = argparse.ArgumentParser(
        description="Query 3DEP Lidar COPC point clouds for bounding boxes and a date range, "
                    "process each separately into individual COPC files, and build (or append to) a local STAC catalog."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("minx", "miny", "maxx", "maxy"),
        required=True,
        action="append",
        help="Bounding box coordinates (minx miny maxx maxy). Use multiple --bbox for multiple areas."
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
        default="./data/3dep_lidar_copc/",
        help="Directory to write the local STAC catalog and COPC files."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use in PDAL processing."
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default="",
        help="Optional target CRS for reprojection (e.g., 'EPSG:4326')."
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    date_range = f"{args.start}/{args.end}"
    
    # Open the Planetary Computer STAC API client.
    catalog_client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Load (or create) the local STAC catalog.
    stac_catalog = load_or_create_catalog(args.output)
    
    # Process each bounding box.
    for bbox in args.bbox:
        process_bbox(bbox, date_range, args.output, stac_catalog, catalog_client, args.threads, args.target_crs)
    
    # Save the updated catalog.
    stac_catalog.normalize_and_save(args.output, catalog_type=CatalogType.SELF_CONTAINED)
    print(f"\nLocal STAC catalog saved to {args.output}")


if __name__ == "__main__":
    main()
