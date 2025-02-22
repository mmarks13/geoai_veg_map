import pdal
import json
import os
import argparse
from datetime import datetime
from pystac import (
    Asset, Catalog, CatalogType, Collection, Extent, Item,
    SpatialExtent, TemporalExtent
)
from shapely.geometry import box, mapping


class PointCloudProcessor:
    def __init__(self, input_file, stac_dir, copc_dir):
        self.input_file = input_file
        self.stac_dir = stac_dir
        self.copc_dir = copc_dir
        self.metadata = None
        os.makedirs(stac_dir, exist_ok=True)
        os.makedirs(copc_dir, exist_ok=True)
        
    def create_copc(self, output_filename=None):
        print(f"Reading input file: {self.input_file}")
        if output_filename is None:
            # The COPC filename is the input basename with .copc.laz extension.
            output_filename = f"{os.path.splitext(os.path.basename(self.input_file))[0]}.copc.laz"
        copc_path = os.path.join(self.copc_dir, output_filename)
        
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
            print(f"Error executing PDAL pipeline: {err}")
            print("PDAL log output:")
            print(pipeline.log)
            return None
        
        print("PDAL pipeline executed successfully.")
        if pipeline.log:
            print("PDAL log output:")
            print(pipeline.log)
        
        self.metadata = pipeline.metadata
        return copc_path

    def get_bounds(self):
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


def get_bounds_from_copc(copc_path):
    """
    Reads a COPC file and returns its bounds.
    """
    pipeline_def = [
        {"type": "readers.copc", "filename": copc_path},
        {"type": "filters.stats", "dimensions": "X,Y,Z"}
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    try:
        pipeline.execute()
    except RuntimeError as err:
        print(f"Error processing COPC file {copc_path}: {err}")
        return None
    metadata = pipeline.metadata
    stats = metadata['metadata']['filters.stats']
    x_stats = stats['statistic'][0]
    y_stats = stats['statistic'][1]
    z_stats = stats['statistic'][2]
    bounds = {
        'minx': x_stats['minimum'],
        'miny': y_stats['minimum'],
        'maxx': x_stats['maximum'],
        'maxy': y_stats['maximum'],
        'minz': z_stats['minimum'],
        'maxz': z_stats['maximum']
    }
    return bounds


def create_stac_catalog(items, overall_bounds, stac_directory, collection_id, collection_title):
    catalog = Catalog(
        id="point-cloud-catalog",
        description="Cloud Optimized Point Cloud Dataset from multiple LAS/COPC files",
        title="Multi-Point Cloud STAC Catalog"
    )
    # If we have overall bounds, use them. Otherwise, set a dummy extent.
    if overall_bounds is not None:
        spatial_extent = SpatialExtent([[
            overall_bounds['minx'],
            overall_bounds['miny'],
            overall_bounds['maxx'],
            overall_bounds['maxy']
        ]])
    else:
        spatial_extent = SpatialExtent([[0, 0, 0, 0]])
    
    collection = Collection(
        id=collection_id,
        description="Collection for point cloud data",
        extent=Extent(
            spatial_extent,
            TemporalExtent([[datetime.utcnow(), None]])
        ),
        title=collection_title
    )
    
    for item in items:
        collection.add_item(item)
    
    catalog.add_child(collection)
    catalog.normalize_hrefs(stac_directory)
    catalog.save(catalog_type=CatalogType.SELF_CONTAINED)
    print(f"STAC catalog saved to {stac_directory}")


def process_mode_all(input_dir, stac_dir, copc_dir, collection_id, collection_title):
    """
    Mode "all": Process every LAS file found in the input directory.
    """
    items = []
    overall_bounds = None
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".las"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {file_path}")
            processor = PointCloudProcessor(file_path, stac_dir, copc_dir)
            copc_path = processor.create_copc()
            if copc_path is None:
                print(f"Skipping {file_path} due to processing error.")
                continue
            bounds = processor.get_bounds()
            # Update overall bounds
            if overall_bounds is None:
                overall_bounds = bounds.copy()
            else:
                overall_bounds['minx'] = min(overall_bounds['minx'], bounds['minx'])
                overall_bounds['miny'] = min(overall_bounds['miny'], bounds['miny'])
                overall_bounds['minz'] = min(overall_bounds['minz'], bounds['minz'])
                overall_bounds['maxx'] = max(overall_bounds['maxx'], bounds['maxx'])
                overall_bounds['maxy'] = max(overall_bounds['maxy'], bounds['maxy'])
                overall_bounds['maxz'] = max(overall_bounds['maxz'], bounds['maxz'])
            print(f"bounds: {bounds}")
            bbox = [
                bounds['minx'], bounds['miny'], bounds['minz'],
                bounds['maxx'], bounds['maxy'], bounds['maxz']
            ]
            geometry = mapping(box(bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']))
            item_id = os.path.splitext(filename)[0]
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=datetime.utcnow(),
                properties={}
            )
            # Use a relative path for the COPC asset.
            rel_path = os.path.join(copc_dir, os.path.basename(copc_path))
            item.add_asset(
                "copc",
                Asset(
                    href=rel_path,
                    media_type="application/vnd.copc+laz",
                    roles=["data"]
                )
            )
            items.append(item)
            print(f"Added item '{item_id}'.")
    print(f"Collection overall bounds now: '{overall_bounds}'")
    create_stac_catalog(items, overall_bounds, stac_dir, collection_id, collection_title)


def process_mode_new(input_dir, stac_dir, copc_dir, collection_id, collection_title):
    """
    Mode "new": Only process LAS files that do not already have a corresponding COPC file.
    Then, build the catalog using all COPC files in the copc directory.
    """
    # Process any new LAS files.
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".las"):
            base = os.path.splitext(filename)[0]
            expected_copc = f"{base}.copc.laz"
            copc_path = os.path.join(copc_dir, expected_copc)
            if not os.path.exists(copc_path):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing new file {file_path}")
                processor = PointCloudProcessor(file_path, stac_dir, copc_dir)
                result = processor.create_copc()
                if result is None:
                    print(f"Skipping {file_path} due to processing error.")
    
    # Now build the STAC catalog from all COPC files.
    items = []
    overall_bounds = None
    for filename in os.listdir(copc_dir):
        if filename.lower().endswith(".copc.laz"):
            copc_path = os.path.join(copc_dir, filename)
            bounds = get_bounds_from_copc(copc_path)
            if bounds is None:
                print(f"Skipping COPC file {copc_path} due to error retrieving bounds.")
                continue
            if overall_bounds is None:
                overall_bounds = bounds.copy()
            else:
                overall_bounds['minx'] = min(overall_bounds['minx'], bounds['minx'])
                overall_bounds['miny'] = min(overall_bounds['miny'], bounds['miny'])
                overall_bounds['minz'] = min(overall_bounds['minz'], bounds['minz'])
                overall_bounds['maxx'] = max(overall_bounds['maxx'], bounds['maxx'])
                overall_bounds['maxy'] = max(overall_bounds['maxy'], bounds['maxy'])
                overall_bounds['maxz'] = max(overall_bounds['maxz'], bounds['maxz'])
            print(f"bounds: {bounds}")
            bbox = [
                bounds['minx'], bounds['miny'], bounds['minz'],
                bounds['maxx'], bounds['maxy'], bounds['maxz']
            ]
            geometry = mapping(box(bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']))
            item_id = os.path.splitext(filename)[0]
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=datetime.utcnow(),
                properties={}
            )
            rel_path = os.path.join(copc_dir, filename)
            item.add_asset(
                "copc",
                Asset(
                    href=rel_path,
                    media_type="application/vnd.copc+laz",
                    roles=["data"]
                )
            )
            items.append(item)
            print(f"Added item '{item_id}' from COPC file.")
            print(f"Collection overall bounds now: '{overall_bounds}'")
    create_stac_catalog(items, overall_bounds, stac_dir, collection_id, collection_title)


def process_mode_catalog(stac_dir, copc_dir, collection_id, collection_title):
    """
    Mode "catalog": Do not process any LAS files; just create a STAC catalog from the
    existing COPC files.
    """
    items = []
    overall_bounds = None
    for filename in os.listdir(copc_dir):
        if filename.lower().endswith(".copc.laz"):
            copc_path = os.path.join(copc_dir, filename)
            bounds = get_bounds_from_copc(copc_path)
            if bounds is None:
                print(f"Skipping COPC file {copc_path} due to error retrieving bounds.")
                continue
            print(f"bounds: {bounds}")
            if overall_bounds is None:
                overall_bounds = bounds.copy()
            else:
                overall_bounds['minx'] = min(overall_bounds['minx'], bounds['minx'])
                overall_bounds['miny'] = min(overall_bounds['miny'], bounds['miny'])
                overall_bounds['minz'] = min(overall_bounds['minz'], bounds['minz'])
                overall_bounds['maxx'] = max(overall_bounds['maxx'], bounds['maxx'])
                overall_bounds['maxy'] = max(overall_bounds['maxy'], bounds['maxy'])
                overall_bounds['maxz'] = max(overall_bounds['maxz'], bounds['maxz'])
            
            bbox = [
                bounds['minx'], bounds['miny'], bounds['minz'],
                bounds['maxx'], bounds['maxy'], bounds['maxz']
            ]
            geometry = mapping(box(bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']))
            item_id = os.path.splitext(filename)[0]
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=datetime.utcnow(),
                properties={}
            )
            rel_path = os.path.join(copc_dir, filename)
            item.add_asset(
                "copc",
                Asset(
                    href=rel_path,
                    media_type="application/vnd.copc+laz",
                    roles=["data"]
                )
            )
            items.append(item)
            print(f"Added item '{item_id}' from COPC file.")
            print(f"Collection overall bounds now: '{overall_bounds}'")
    create_stac_catalog(items, overall_bounds, stac_dir, collection_id, collection_title)


def main():
    parser = argparse.ArgumentParser(
        description="Process point cloud LAS files to COPC and create a STAC catalog."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "new", "catalog"],
        default="all",
        help=(
            "Processing mode: 'all' reprocesses all LAS files, 'new' processes only new LAS files, "
            "and 'catalog' only creates the STAC catalog from COPC files."
        )
    )
    parser.add_argument(
        "--input-directory",
        type=str,
        default="uavlidar/original_las",
        help="Directory containing LAS files."
    )
    parser.add_argument(
        "--stac-directory",
        type=str,
        default="local_stac",
        help="Directory where the STAC catalog will be written."
    )
    parser.add_argument(
        "--copc-directory",
        type=str,
        default="uavlidar/processed_copc",
        help="Directory where COPC files are/will be written."
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default="pointcloud_collection",
        help="Collection ID for the STAC catalog."
    )
    parser.add_argument(
        "--collection-title",
        type=str,
        default="Point Cloud Collection",
        help="Collection title for the STAC catalog."
    )
    args = parser.parse_args()
    
    if args.mode == "all":
        process_mode_all(args.input_directory, args.stac_directory, args.copc_directory,
                         args.collection_id, args.collection_title)
    elif args.mode == "new":
        process_mode_new(args.input_directory, args.stac_directory, args.copc_directory,
                         args.collection_id, args.collection_title)
    elif args.mode == "catalog":
        process_mode_catalog(args.stac_directory, args.copc_directory,
                             args.collection_id, args.collection_title)


if __name__ == "__main__":
    main()
#python las_to_copc_stac.py --mode catalog  --collection-id volcan_mtn_uav_lidar --collection-title "Volcan Mountain UAV Point Clouds"