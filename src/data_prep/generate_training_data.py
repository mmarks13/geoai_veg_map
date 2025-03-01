#!/usr/bin/env python3
"""
Generate LiDAR Training Data from Tiles

This script reads tile geometries from a GeoJSON file (created by create_tiles.py),
retrieves LiDAR data for each tile from UAV and 3DEP sources, processes the data,
and saves the results as PyTorch tensors.

Usage:
    python generate_training_data.py --tiles_geojson data/processed/tiles.geojson --outdir output/test
    python generate_training_data.py --tiles_geojson data/processed/tiles.geojson --outdir output/test --sample 10
"""

import os
import gc
import json
import torch
import numpy as np
import geopandas as gpd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer
import argparse
import pdal
import pystac
from pystac_client import Client
import planetary_computer
from shapely.geometry import mapping, box


def calculate_geoid_undulation(x, y, input_crs="EPSG:32611"):
    """
    Calculate the adjustment value to convert from WGS84 Ellipsoidal Height 
    to NAVD88 orthometric height.
    
    This function returns a value that can be directly ADDED to ellipsoidal heights
    to convert them to orthometric heights in NAVD88 (Geoid12B).
    
    Parameters
    ----------
    x : float
        X coordinate (easting)
    y : float
        Y coordinate (northing)
    input_crs : str
        CRS of input coordinates
        
    Returns
    -------
    float
        Adjustment value in meters (to be added to ellipsoidal heights)
    """
    try:
        # Create transformer for horizontal coordinates
        transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        
        # Create a transformer from ellipsoidal to orthometric heights
        # WGS84 ellipsoidal heights to NAVD88 heights (Geoid12B)
        vert_transformer = Transformer.from_crs(
            "EPSG:4979",  # WGS84 3D geographic
            "EPSG:4326+5703",  # WGS84 2D geographic + NAVD88 height
            always_xy=True
        )
        
        # Use a sample ellipsoidal height of 0 to get the undulation
        _, _, orth_height = vert_transformer.transform(lon, lat, 0.0)
        
        # The geoid undulation = ellipsoidal height - orthometric height
        # Since we want a value to ADD to ellipsoidal heights, we return the orthometric height directly
        # When ellipsoidal_height = 0: adjustment_value = orthometric_height
        adjustment_value = orth_height
        
        # print(f"Calculated height adjustment: {adjustment_value:.4f}m at location ({lon:.6f}, {lat:.6f})")
        # print(f"(Add this value to ellipsoidal heights to get NAVD88 heights)")
        return adjustment_value
        
    except Exception as e:
        print(f"Error calculating height adjustment at location ({lon:.6f}, {lat:.6f})): {e}")
        return 


def bounding_box_to_geojson(bbox):
    """
    Converts a bounding box to a GeoJSON polygon.
    """
    return json.dumps(mapping(box(bbox[0], bbox[1], bbox[2], bbox[3])))


def create_pointcloud_stack(bbox, start_date, end_date, stac_source, threads=4,
                            bbox_crs="EPSG:4326", target_crs=None):
    """
    Create a stack of point clouds from STAC items filtered by date and bounding box.
    """
    # Read the STAC catalog
    catalog = pystac.read_file(stac_source)
    items = []

    def bboxes_intersect(bbox1, bbox2):
        # Return True if bbox1 intersects bbox2 (both in [xmin, ymin, xmax, ymax] format)
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
    
    # Filter items by date and spatial intersection
    for item in catalog.get_all_items():
        # Check the date
        item_date = item.datetime.date()
        if not (start_date <= str(item_date) <= end_date):
            continue

        # Check horizontal bounding box intersection
        if item.bbox:
            if len(item.bbox) == 6:
                # Assume item.bbox = [minx, miny, minz, maxx, maxy, maxz]
                item_xy_bbox = [item.bbox[0], item.bbox[1], item.bbox[3], item.bbox[4]]
            else:
                item_xy_bbox = item.bbox
            if bboxes_intersect(bbox, item_xy_bbox):
                items.append(item)

    if not items:
        raise ValueError("No items found for the specified parameters.")

    point_clouds = []

    # Read each point cloud via PDAL
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
                    "polygon": bounding_box_to_geojson(bbox),
                    "nosrs": True #not ideal but necessary to move forward with both Sedgwick and Volcan on 32611
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
            if arrays:
                point_clouds.append(arrays[0])
        except Exception as e:
            print(f"Error processing point cloud: {e}")
            continue
        finally:
            del pipeline
            gc.collect()
    
    try:
        # If more than one point cloud was returned, combine them
        if len(point_clouds) > 1:
            combined_pc = np.concatenate(point_clouds)
        else:
            combined_pc = point_clouds[0]

        # # Adjust Z values (e.g., convert from ellipsoidal to orthometric height)
        # combined_pc['Z'] = combined_pc['Z'] + 31.8684
            
        # Dynamically adjust Z values (convert from ellipsoidal to orthometric height)
        # Calculate conversion factor based on the x,y location of the first point
        pc_crs = target_crs if target_crs else bbox_crs
        if combined_pc.size > 0:
            first_point_x = combined_pc[0]['X']
            first_point_y = combined_pc[0]['Y']
            adjustment_value = calculate_geoid_undulation(first_point_x, first_point_y,pc_crs)
            # print(f"Using adjustment value of {adjustment_value:.4f}m for elevation conversion")
            combined_pc['Z'] = combined_pc['Z'] + adjustment_value
        
        # Return the combined point cloud
        point_clouds = [combined_pc]
                        
    except Exception as e:
        print(f"Error processing point cloud data: {e}")
    
    del items, catalog
    gc.collect()

    return point_clouds


def create_3dep_stack(bbox, start_date, end_date, threads=4, target_crs=None):
    """
    Create a stack of point clouds from 3DEP STAC items.
    
    Parameters
    ----------
    bbox : list
        Bounding box in the format [xmin, ymin, xmax, ymax]
    start_date : str
        Start date for filtering
    end_date : str
        End date for filtering
    threads : int
        Number of threads for processing
    target_crs : str
        Target coordinate reference system for reprojection
    """
    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
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
        
        polygon = bounding_box_to_geojson(bbox)
        point_clouds = []
        for tile in items:
            try:
                if "data" not in tile.assets:
                    continue
                    
                url = planetary_computer.sign(tile.assets["data"].href)
                
                pipeline_dict = {
                    "pipeline": [
                        {
                            "type": "readers.copc",
                            "filename": url,
                            "polygon": polygon,
                            "requests": 4,
                            "keep_alive": 30
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
        return filtered_point_clouds
    finally:
        del client
        if 'search' in locals():
            del search
        gc.collect()


def process_bbox(args):
    """
    Process a single bounding box to retrieve and format LiDAR data.
    """
    # The args tuple is: (i, bbox, start_date, end_date, stac_source, bbox_crs, max_retries, initial_delay)
    i, bbox, start_date, end_date, stac_source, bbox_crs = args[:6]
    # Extract optional retry parameters if provided
    max_retries = args[6] if len(args) > 6 else 5
    initial_delay = args[7] if len(args) > 7 else 2
    
    try:
        print(f"Processing tile {i}: {bbox}")
        
        # UAV LiDAR point clouds
        uav_pc = create_pointcloud_stack(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            stac_source=stac_source,
            bbox_crs=bbox_crs,
            threads=1
        )
        
        if not uav_pc or len(uav_pc) == 0:
            print(f"No UAV point cloud data found for tile {i}")
            return None

        # Extract x,y,z values from the UAV point cloud
        xyz_uav = np.array([(p['X'], p['Y'], p['Z']) for p in uav_pc[0]], dtype=np.float32)
        # Also extract additional fields from the UAV point cloud
        intensity_uav = np.array([p['Intensity'] for p in uav_pc[0]], dtype=np.int32)
        return_number_uav = np.array([p['ReturnNumber'] for p in uav_pc[0]], dtype=np.int32)
        num_returns_uav = np.array([p['NumberOfReturns'] for p in uav_pc[0]], dtype=np.int32)
        
        # 3DEP LiDAR point clouds - with retry logic
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        bbox_wgs84 = transformer.transform_bounds(*bbox)

        # Add retry logic for 3DEP stack creation
        import time
        from pystac_client.exceptions import APIError
        
        dep_pc = None
        attempt = 0
        last_exception = None
        
        while attempt < max_retries and dep_pc is None:
            try:
                if attempt > 0:
                    print(f"3DEP data retrieval retry attempt {attempt}/{max_retries} for tile {i}")
                
                dep_pc = create_3dep_stack(
                    bbox=bbox_wgs84,
                    start_date=start_date,
                    end_date=end_date,
                    threads=1
                )
                
                # If we got here, the call was successful
                if attempt > 0:
                    print(f"✅ 3DEP data retrieval succeeded after {attempt+1} attempts for tile {i}!")
                    
            except Exception as e:
                last_exception = e
                # print(f"Error retrieving 3DEP data for tile {i} (attempt {attempt+1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Calculate backoff time with exponential increase
                    sleep_time = initial_delay * (2 ** attempt)
                    # print(f"Retrying 3DEP data retrieval in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"❌ All {max_retries} 3DEP data retrieval attempts failed for tile {i}. Giving up.")
                
                attempt += 1
        
        # If all attempts failed, return None
        if dep_pc is None:
            print(f"Failed to retrieve 3DEP data for tile {i} after {max_retries} attempts")
            return None
        
        # For the 3DEP data, we concatenate the returned arrays
        xyz_dep = np.vstack([np.column_stack((p['X'], p['Y'], p['Z'])) for p in dep_pc]).astype(np.float32)
        intensity_dep = np.concatenate([p['Intensity'] for p in dep_pc]).astype(np.int32)
        return_number_dep = np.concatenate([p['ReturnNumber'] for p in dep_pc]).astype(np.int32)
        num_returns_dep = np.concatenate([p['NumberOfReturns'] for p in dep_pc]).astype(np.int32)
        
        del transformer
        del bbox_wgs84
        del uav_pc
        del dep_pc
        gc.collect()
        
        print(f"Successfully processed tile {i}: UAV points: {xyz_uav.shape[0]}, 3DEP points: {xyz_dep.shape[0]}")
        
        return {
            'dep_points': torch.tensor(xyz_dep, dtype=torch.float32),
            'uav_points': torch.tensor(xyz_uav, dtype=torch.float32),
            'uav_intensity': torch.tensor(intensity_uav, dtype=torch.int32),
            'uav_return_number': torch.tensor(return_number_uav, dtype=torch.int32),
            'uav_num_returns': torch.tensor(num_returns_uav, dtype=torch.int32),
            'dep_intensity': torch.tensor(intensity_dep, dtype=torch.int32),
            'dep_return_number': torch.tensor(return_number_dep, dtype=torch.int32),
            'dep_num_returns': torch.tensor(num_returns_dep, dtype=torch.int32),
            'bbox': bbox
        }
    except Exception as e:
        print(f"Error processing bounding box {i}: {e}")
        import traceback
        traceback.print_exc()
        return None


def chunk_data(data, chunk_size):
    """Split data into smaller chunks of size `chunk_size`."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def process_tiles_from_geojson(
    tiles_geojson,
    start_date, 
    end_date, 
    stac_source, 
    bbox_crs="EPSG:32611", 
    max_threads=2,
    chunk_size=20,
    output_dir="training_data_chunks",
    sample_size=None,
    max_api_retries=5,
    initial_retry_delay=2
):
    """
    Process LiDAR data using pre-defined tiles from a GeoJSON file.
    
    Parameters
    ----------
    tiles_geojson : str
        Path to GeoJSON file containing tile geometries.
    sample_size : int, optional
        Number of tiles to randomly sample from the input GeoJSON.
        If None, all tiles will be processed.
    """
    # Read the tiles from GeoJSON
    gdf = gpd.read_file(tiles_geojson)
    
    # Sample tiles if sample_size is specified
    if sample_size is not None and sample_size < len(gdf):
        print(f"Randomly sampling {sample_size} tiles from {len(gdf)} total tiles")
        gdf = gdf.sample(n=sample_size, random_state=42)  # Using fixed random_state for reproducibility
    
    # Extract bounding boxes from tile geometries
    tile_bounding_boxes = []
    tile_ids = []  # Track tile IDs if available, otherwise use index
    
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        bbox = [geom.bounds[0], geom.bounds[1], geom.bounds[2], geom.bounds[3]]
        tile_bounding_boxes.append(bbox)
        
        # Use ID field if available, otherwise use index
        tile_id = row.get('id', idx)
        tile_ids.append(tile_id)
    
    print(f"Processing {len(tile_bounding_boxes)} tiles from {tiles_geojson}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split tiles into chunks
    chunks = list(chunk_data(list(zip(tile_ids, tile_bounding_boxes)), chunk_size))
    print(f"Processing {len(chunks)} chunks of tiles, with up to {chunk_size} per chunk.")

    # Track processing results
    results_tracker = {
        'successful': [],
        'failed': []
    }

    for chunk_index, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {chunk_index}/{len(chunks)}...")
        
        # Prepare arguments for parallel processing
        args_list = [
            (i, bbox, start_date, end_date, stac_source, bbox_crs, max_api_retries, initial_retry_delay)
            for i, (tile_id, bbox) in enumerate(chunk, start=1)
        ]
        
        chunk_results = []
        chunk_failures = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_threads) as executor:
                future_to_tile = {executor.submit(process_bbox, args): (i, tile_id) for i, (args, (i, tile_id)) in 
                                 enumerate(zip(args_list, enumerate([tid for tid, _ in chunk], start=1)))}
                
                for future in future_to_tile:
                    i, tile_id = future_to_tile[future]
                    try:
                        result = future.result()
                        if result is not None:
                            chunk_results.append(result)
                            results_tracker['successful'].append((tile_id, "Successfully processed"))
                        else:
                            # Failed to process tile
                            chunk_failures.append(tile_id)
                            results_tracker['failed'].append((tile_id, "Failed to process - see logs for details"))
                    except Exception as e:
                        chunk_failures.append(tile_id)
                        results_tracker['failed'].append((tile_id, f"Error: {str(e)[:100]}..."))  # Truncate very long error messages
                
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

    # Print summary report
    print("\n" + "="*80)
    print(f"PROCESSING SUMMARY REPORT")
    print("="*80)
    
    total_tiles = len(tile_ids)
    successful_count = len(results_tracker['successful'])
    failed_count = len(results_tracker['failed'])
    
    print(f"Total tiles processed: {total_tiles}")
    print(f"Successfully processed: {successful_count} ({successful_count/total_tiles*100:.1f}%)")
    print(f"Failed to process: {failed_count} ({failed_count/total_tiles*100:.1f}%)")
    
    if failed_count > 0:
        print("\nFailed tiles:")
        for tile_id, reason in results_tracker['failed']:
            print(f"  - Tile {tile_id}: {reason}")
    
    print("="*80)
    print("Completed processing all chunks.")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process training data using tiles from a GeoJSON file."
    )
    parser.add_argument(
        "--tiles_geojson",
        type=str,
        required=True,
        help="Path to GeoJSON file containing tile geometries"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for training data chunks"
    )
    
    parser.add_argument(
        "--stac_source",
        type=str,
        default="local_stac/catalog.json",
        help="Path to STAC catalog"
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        default="2014-01-01",
        help="Start date for data filtering (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-02-27",
        help="End date for data filtering (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Maximum number of parallel threads"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="Number of tiles to process in each chunk"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of bounding boxes to randomly sample from the input GeoJSON (default: use all)"
    )
    
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for Planetary Computer API calls (default: 5)"
    )
    
    parser.add_argument(
        "--initial-retry-delay",
        type=float,
        default=2,
        help="Initial delay in seconds between API retry attempts - will increase exponentially (default: 2)"
    )

    args = parser.parse_args()
    
    # Call processing function with the given tiles and other parameters
    process_tiles_from_geojson(
        tiles_geojson=args.tiles_geojson, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        stac_source=args.stac_source, 
        bbox_crs="EPSG:32611", 
        max_threads=args.threads,
        chunk_size=args.chunk_size,
        output_dir=args.outdir,
        sample_size=args.sample,
        max_api_retries=args.max_api_retries,
        initial_retry_delay=args.initial_retry_delay
    )