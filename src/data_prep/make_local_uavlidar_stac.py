#!/usr/bin/env python
"""
Script Summary:
--------------
This script processes point cloud LAS/LAZ files by converting them to COPC files using PDAL.
For each input file in the specified input directory, a COPC file is created and saved in the output
directory. A corresponding STAC item is then generated using the spatial bounds extracted from the
COPC file and an acquisition date extracted from the filename. If the filename contains an 8-digit
date (in YYYYMMDD format), that date is used; otherwise, the current UTC time is used.
The script loads an existing STAC catalog from the output directory if one exists and appends new items,
skipping duplicates. The output directory will contain both the COPC files and the STAC catalog.

Command-line arguments:
  --input            Directory containing LAS/LAZ files.
  --output           Output directory to store both the COPC files and the STAC catalog.
  --collection-id    Collection ID for the STAC catalog.
  --collection-title Collection title for the STAC catalog.

Example usage:
  python make_local_uavlidar_stac.py \
    --input data/raw/study_las \
    --output data/stac/uavlidar \
    --collection-id uav_lidar \
    --collection-title "UAV LiDAR Point Clouds"
"""

import os
import re
import json
import argparse
import subprocess
import logging
from datetime import datetime

import pdal
from shapely.geometry import box, mapping
import pystac
from pystac import Asset, Catalog, CatalogType, Collection, Extent, Item, SpatialExtent, TemporalExtent

# Configure logging for helpful console output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_or_create_catalog(output_dir):
    """
    Load an existing STAC catalog from output_dir if it exists (catalog.json), or create a new one.
    """
    catalog_file = os.path.join(output_dir, "catalog.json")
    if os.path.exists(catalog_file):
        logging.info(f"Existing catalog found; loading catalog from {catalog_file}")
        return pystac.read_file(catalog_file)
    else:
        logging.info("No existing catalog found; creating a new catalog.")
        return Catalog(
            id="pointcloud-catalog",
            description="Local STAC catalog for COPC point cloud files",
            title="Point Cloud STAC Catalog"
        )


def extract_date_from_filename(filename):
    """
    Extracts an 8-digit date (YYYYMMDD) from the filename and returns a datetime object.
    If no date is found, returns the current UTC time.
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except Exception as e:
            logging.warning(f"Error parsing date from filename '{filename}': {e}")
    logging.warning(f"No date found in filename '{filename}'; using current UTC time.")
    return datetime.utcnow()


class PointCloudProcessor:
    """
    Processes a single LAS/LAZ file:
      - Converts the file to a COPC file using a PDAL pipeline.
      - Stores PDAL metadata (used for bounds extraction).
    """
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.metadata = None
        os.makedirs(output_dir, exist_ok=True)

    def create_copc(self, output_filename=None):
        """
        Create a COPC file from the input LAS/LAZ file using PDAL.
        The COPC filename defaults to the input file's basename with a .copc.laz extension.
        """
        logging.info(f"Reading input file: {self.input_file}")
        if output_filename is None:
            output_filename = f"{os.path.splitext(os.path.basename(self.input_file))[0]}.copc.laz"
        copc_path = os.path.join(self.output_dir, output_filename)

        # PDAL pipeline: read LAS, compute stats, and write COPC.
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": self.input_file,
                "threads": 24
            },
            {
                "type": "filters.stats",
                "dimensions": "X,Y,Z"
            },
            {
                "type": "writers.copc",
                "filename": copc_path,
                "forward": "all",
                "threads": 24
            }
        ]

        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        try:
            pipeline.execute()
        except RuntimeError as err:
            logging.error(f"Error executing PDAL pipeline for {self.input_file}: {err}")
            logging.error("PDAL log output:")
            logging.error(pipeline.log)
            return None

        logging.info("PDAL pipeline executed successfully.")
        if pipeline.log:
            logging.info("PDAL log output:")
            logging.info(pipeline.log)
        self.metadata = pipeline.metadata
        return copc_path

    def get_bounds(self):
        """
        Extracts spatial bounds from the PDAL metadata.
        Returns a dictionary with keys: minx, miny, maxx, maxy, minz, maxz.
        """
        if not self.metadata:
            raise ValueError("No metadata available. Run create_copc first.")
        stats = self.metadata['metadata']['filters.stats']
        x_stats = stats['statistic'][0]
        y_stats = stats['statistic'][1]
        z_stats = stats['statistic'][2]

        return {
            'minx': x_stats['minimum'],
            'miny': y_stats['minimum'],
            'maxx': x_stats['maximum'],
            'maxy': y_stats['maximum'],
            'minz': z_stats['minimum'],
            'maxz': z_stats['maximum']
        }


def create_stac_catalog(items, overall_bounds, output_dir, collection_id, collection_title):
    """
    Create (or update) a STAC catalog from a list of STAC items.
    overall_bounds: dictionary with keys: minx, miny, maxx, maxy, minz, maxz.
    The catalog is saved in the output_dir.
    """
    catalog = load_or_create_catalog(output_dir)

    # Build a spatial extent using the 2D (x,y) bounds.
    if overall_bounds is not None:
        spatial_extent = SpatialExtent([[
            overall_bounds['minx'],
            overall_bounds['miny'],
            overall_bounds['maxx'],
            overall_bounds['maxy']
        ]])
    else:
        spatial_extent = SpatialExtent([[0, 0, 0, 0]])

    # Create or update a collection.
    collection = Collection(
        id=collection_id,
        description="Collection for point cloud data",
        extent=Extent(
            spatial=spatial_extent,
            temporal=TemporalExtent([[datetime.utcnow(), None]])
        ),
        title=collection_title
    )

    # Add new items that are not already in the catalog.
    for item in items:
        if catalog.get_item(item.id) is None:
            collection.add_item(item)
            logging.info(f"Added item '{item.id}' to collection.")
        else:
            logging.info(f"Item '{item.id}' already exists in catalog; skipping.")

    catalog.add_child(collection)
    catalog.normalize_and_save(root_href=output_dir, catalog_type=CatalogType.SELF_CONTAINED)
    logging.info(f"STAC catalog saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAS/LAZ files to COPC and build (or update) a STAC catalog for point cloud data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing LAS/LAZ files."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to store the COPC files and the STAC catalog."
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        required=True,
        help="Collection ID for the STAC catalog."
    )
    parser.add_argument(
        "--collection-title",
        type=str,
        required=True,
        help="Collection title for the STAC catalog."
    )
    args = parser.parse_args()

    items = []
    overall_bounds = None

    # Load existing catalog to check for duplicates.
    catalog = load_or_create_catalog(args.output)

    # Process each LAS/LAZ file in the input directory.
    for filename in os.listdir(args.input):
        if filename.lower().endswith((".las", ".laz")):
            file_path = os.path.join(args.input, filename)
            # Use the base filename (without extension) as the STAC item ID.
            item_id = os.path.splitext(filename)[0]
            # Skip if the item already exists in the catalog.
            if catalog.get_item(item_id) is not None:
                logging.info(f"Item '{item_id}' already exists in the catalog; skipping {file_path}.")
                continue

            logging.info(f"Processing {file_path}")
            processor = PointCloudProcessor(file_path, args.output)
            copc_path = processor.create_copc()
            if copc_path is None:
                logging.error(f"Skipping {file_path} due to processing error.")
                continue

            bounds = processor.get_bounds()
            # Update overall bounds for collection extent.
            if overall_bounds is None:
                overall_bounds = bounds.copy()
            else:
                overall_bounds['minx'] = min(overall_bounds['minx'], bounds['minx'])
                overall_bounds['miny'] = min(overall_bounds['miny'], bounds['miny'])
                overall_bounds['minz'] = min(overall_bounds['minz'], bounds['minz'])
                overall_bounds['maxx'] = max(overall_bounds['maxx'], bounds['maxx'])
                overall_bounds['maxy'] = max(overall_bounds['maxy'], bounds['maxy'])
                overall_bounds['maxz'] = max(overall_bounds['maxz'], bounds['maxz'])

            # Extract the acquisition datetime from the filename.
            acq_datetime = extract_date_from_filename(filename)

            # Create spatial geometry using the 2D (x,y) bounds and a 3D bbox.
            geometry = mapping(box(bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']))
            bbox = [
                bounds['minx'], bounds['miny'], bounds['minz'],
                bounds['maxx'], bounds['maxy'], bounds['maxz']
            ]
            # Create the STAC item.
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=acq_datetime,
                properties={}
            )
            # Use a relative path for the COPC asset.
            rel_path = os.path.join(args.output, os.path.basename(copc_path))
            item.add_asset(
                "copc",
                Asset(
                    href=rel_path,
                    media_type="application/vnd.copc+laz",
                    roles=["data"]
                )
            )
            items.append(item)
            logging.info(f"Added item '{item_id}' from file {file_path}.")

    # Create (or update) the STAC catalog with the new items.
    create_stac_catalog(items, overall_bounds, args.output, args.collection_id, args.collection_title)


if __name__ == "__main__":
    main()
