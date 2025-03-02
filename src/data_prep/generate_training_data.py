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
import time
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
        return 0  # Return default value in case of error


def bounding_box_to_geojson(bbox):
    """
    Converts a bounding box to a GeoJSON polygon.
    """
    return json.dumps(mapping(box(bbox[0], bbox[1], bbox[2], bbox[3])))


def create_pointcloud_stack(bbox, start_date, end_date, stac_source, threads=4,
                            bbox_crs="EPSG:4326", target_crs=None):
    """
    Create a stack of point clouds from STAC items filtered by date and bounding box.
    
    Returns:
    --------
    tuple:
        - list of point clouds
        - metadata dictionary for the most recent point cloud
    """
    # Read the STAC catalog
    catalog = None
    items = []
    point_clouds = []
    datetimes = []
    item_metadata = []  # Add collection for item metadata

    try:
        catalog = pystac.read_file(stac_source)

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

            pipeline = None
            try:
                pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
                pipeline.execute()
                arrays = pipeline.arrays
                if arrays and len(arrays) > 0 and arrays[0].size > 0:
                    point_clouds.append(arrays[0].copy())  # Make a copy to ensure proper memory management
                    # Store the datetime from the STAC item
                    datetimes.append(item.datetime)
                    
                    # Create metadata for this item
                    meta = {
                        'id': item.id,
                        'href': item.assets['copc'].href,
                        'datetime': item.datetime.isoformat() if item.datetime else None,
                    }
                    
                    # Add projection info if available
                    if 'proj:epsg' in item.properties:
                        meta['proj:epsg'] = item.properties['proj:epsg']
                    
                    # Add additional metadata from properties if available
                    for key in ['platform', 'instrument', 'gsd', 'created', 'updated']:
                        if key in item.properties:
                            meta[key] = item.properties[key]
                            
                    # Add projection info from pipeline metadata if available
                    if pipeline.metadata and 'metadata' in pipeline.metadata:
                        meta_dict = pipeline.metadata['metadata']
                        if 'filters.reprojection' in meta_dict:
                            reprojection = meta_dict['filters.reprojection']
                            if 'srs' in reprojection and 'proj4' in reprojection['srs']:
                                meta['proj4'] = reprojection['srs']['proj4']
                    
                    item_metadata.append(meta)
            except Exception as e:
                print(f"Error processing point cloud: {e}")
                continue
            finally:
                # Explicitly clean up the pipeline
                if pipeline is not None:
                    del pipeline
                    pipeline = None
                    gc.collect()
        
        try:
        # If more than one point cloud was returned, combine them
            if len(point_clouds) > 1:
                combined_pc = np.concatenate(point_clouds)
                # Get the metadata from the latest item
                latest_metadata = {}
                if item_metadata:
                    # Sort by datetime if available
                    if all('datetime' in meta for meta in item_metadata):
                        item_metadata.sort(key=lambda x: x['datetime'], reverse=True)
                    latest_metadata = item_metadata[0]
                # Clean up individual point clouds after concatenation
                for pc in point_clouds:
                    del pc
                point_clouds = []  # Clear list to avoid double cleanup in finally block
            elif len(point_clouds) == 1:
                combined_pc = point_clouds[0]
                latest_metadata = item_metadata[0] if item_metadata else {}
                point_clouds = []  # Clear list to avoid double cleanup in finally block
            else:
                raise ValueError("No point cloud data was successfully retrieved")
                
            # Dynamically adjust Z values (convert from ellipsoidal to orthometric height)
            pc_crs = target_crs if target_crs else bbox_crs
            if combined_pc.size > 0:
                first_point_x = combined_pc[0]['X']
                first_point_y = combined_pc[0]['Y']
                adjustment_value = calculate_geoid_undulation(first_point_x, first_point_y, pc_crs)
                combined_pc['Z'] = combined_pc['Z'] + adjustment_value
            
            # Return the combined point cloud in a new list along with the metadata
            return [combined_pc], latest_metadata
                            
        except Exception as e:
            print(f"Error processing point cloud data: {e}")
            return [], {}
    
    except Exception as e:
        print(f"Error in create_pointcloud_stack: {e}")
        return [], {}
        
    finally:
        # Clean up resources
        if point_clouds:
            for pc in point_clouds:
                del pc
        
        if items:
            del items
            
        if catalog:
            del catalog
            
        gc.collect()


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
        
    Returns
    -------
    tuple:
        - list of filtered point clouds
        - metadata dictionary
    """
    client = None
    search = None
    items = []
    point_clouds = []
    item_metadata = []
    
    try:
        client = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )

        search = client.search(
            collections=["3dep-lidar-copc"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}"
        )
        items = list(search.items())
        if not items:
            raise ValueError("No items found for the specified parameters.")
        
        polygon = bounding_box_to_geojson(bbox)
        
        for tile in items:
            pipeline = None
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
                    point_clouds.append(arrays[0].copy())  # Make a copy to ensure proper memory management
                    
                    # Store metadata for this item
                    meta = {
                        'href': tile.assets["data"].href,
                    }
                    
                    # Add USGS ID if available
                    if 'start_datetime' in tile.properties:
                        meta['start_datetime'] = tile.properties['start_datetime']
                    if 'end_datetime' in tile.properties:
                        meta['end_datetime'] = tile.properties['end_datetime']
                    if '3dep:usgs_id' in tile.properties:
                        meta['3dep:usgs_id'] = tile.properties['3dep:usgs_id']
                        
                    # Add projection info from pipeline metadata if available
                    if pipeline.metadata and 'metadata' in pipeline.metadata:
                        meta_dict = pipeline.metadata['metadata']
                        if 'filters.reprojection' in meta_dict:
                            reprojection = meta_dict['filters.reprojection']
                            if 'srs' in reprojection and 'proj4' in reprojection['srs']:
                                meta['proj4'] = reprojection['srs']['proj4']
                    
                    item_metadata.append(meta)
            except Exception as e:
                print(f"Error processing tile {url}: {str(e)}")
                continue
            finally:
                # Explicitly clean up the pipeline
                if pipeline is not None:
                    del pipeline
                    pipeline = None
                gc.collect()
        
        if not point_clouds:
            raise ValueError("No point cloud data was successfully retrieved")
        
        # Find the common fields across all arrays
        common_fields = set(point_clouds[0].dtype.names)
        for pc in point_clouds[1:]:
            common_fields.intersection_update(pc.dtype.names)

        # Keep only the common fields and return as a list of structured arrays
        common_fields = list(common_fields)
        filtered_point_clouds = []
        
        for pc in point_clouds:
            # Extract only the common fields and make a copy
            filtered_pc = pc[common_fields].copy()
            filtered_point_clouds.append(filtered_pc)
            # Clean up the original point cloud
            del pc
        
        # Clear the original list to avoid double cleanup in finally block
        point_clouds = []
        
        # Select the metadata from the most recent item
        if item_metadata:
            # Sort by start_datetime if available
            if all('start_datetime' in meta for meta in item_metadata):
                item_metadata.sort(key=lambda x: x['start_datetime'], reverse=True)
            latest_metadata = item_metadata[0]
        else:
            latest_metadata = {}
        
        return filtered_point_clouds, latest_metadata
    
    except Exception as e:
        print(f"Error in create_3dep_stack: {e}")
        return [], {}
        
    finally:
        # Clean up resources
        if point_clouds:
            for pc in point_clouds:
                del pc
            
        if items:
            del items
            
        if client:
            del client
            
        if search:
            del search
            
        gc.collect()


def process_bbox(args):
    """
    Process a single bounding box to retrieve and format LiDAR data.
    """
    # The args tuple is: (i, tile_id, bbox, start_date, end_date, stac_source, bbox_crs, max_retries, initial_delay, current_tile_index, total_tiles, verbose)
    i, tile_id, bbox, start_date, end_date, stac_source, bbox_crs = args[:7]
    # Extract optional retry parameters if provided
    max_retries = args[7] if len(args) > 7 else 5
    initial_delay = args[8] if len(args) > 8 else 2
    # Extract progress tracking information
    current_tile_index = args[9] if len(args) > 9 else 0
    total_tiles = args[10] if len(args) > 10 else 0
    verbose = args[11] if len(args) > 11 else False
    
    # Calculate the overall tile number for progress reporting
    overall_tile_index = current_tile_index + i
    
    # Set up proper cleanup/resource tracking
    uav_pc = None
    dep_pc = None
    transformer = None
    xyz_uav = None
    xyz_dep = None
    uav_pnt_attr = None
    dep_pnt_attr = None
    bbox_wgs84 = None
    
    try:
        if verbose:
            print(f"Processing tile {tile_id} (index {i}): {bbox}")
        
        # UAV LiDAR point clouds with metadata
        uav_pc, uav_meta = create_pointcloud_stack(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            stac_source=stac_source,
            bbox_crs=bbox_crs,
            threads=1
        )
        
        if not uav_pc or len(uav_pc) == 0:
            print(f"No UAV point cloud data found for tile {tile_id}")
            return None

        # Extract x,y,z values from the UAV point cloud
        xyz_uav = np.array([(p['X'], p['Y'], p['Z']) for p in uav_pc[0]], dtype=np.float32)
        
        # Extract and combine point attributes (Intensity, ReturnNumber, NumberOfReturns)
        uav_pnt_attr = np.column_stack((
            np.array([p['Intensity'] for p in uav_pc[0]], dtype=np.float32),
            np.array([p['ReturnNumber'] for p in uav_pc[0]], dtype=np.float32),
            np.array([p['NumberOfReturns'] for p in uav_pc[0]], dtype=np.float32)
        ))
        
        # Clean up UAV point cloud data after extraction
        for pc in uav_pc:
            del pc
        uav_pc = None
        gc.collect()
        
        # 3DEP LiDAR point clouds - with retry logic
        transformer = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        bbox_wgs84 = transformer.transform_bounds(*bbox)

        # Add retry logic for 3DEP stack creation
        dep_pc = None
        dep_meta = None
        attempt = 0
        last_exception = None
        
        while attempt < max_retries and dep_pc is None:
            try:
                if attempt > 0 and verbose:
                    print(f"3DEP data retrieval retry attempt {attempt}/{max_retries} for tile {tile_id}")
                
                dep_pc, dep_meta = create_3dep_stack(
                    bbox=bbox_wgs84,
                    start_date=start_date,
                    end_date=end_date,
                    threads=1
                )
                
                # If we got here, the call was successful
                if attempt > 0 and verbose:
                    print(f"✅ 3DEP data retrieval succeeded after {attempt+1} attempts for tile {tile_id}!")
                    
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries - 1:
                    # Calculate backoff time with exponential increase
                    sleep_time = initial_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    print(f"❌ All {max_retries} 3DEP data retrieval attempts failed for tile {tile_id}. Giving up.")
                
                attempt += 1
        
        # If all attempts failed, return None
        if dep_pc is None:
            print(f"Failed to retrieve 3DEP data for tile {tile_id} after {max_retries} attempts")
            return None
        
        # For the 3DEP data, we concatenate the returned arrays
        xyz_dep = np.vstack([np.column_stack((p['X'], p['Y'], p['Z'])) for p in dep_pc]).astype(np.float32)
        
        # Combine point attributes for 3DEP data
        dep_pnt_attr = np.vstack([
            np.column_stack((
                p['Intensity'],
                p['ReturnNumber'],
                p['NumberOfReturns']
            )) for p in dep_pc
        ]).astype(np.float32)
        
        # Clean up 3DEP point cloud data after extraction
        for pc in dep_pc:
            del pc
        dep_pc = None
        gc.collect()
        
        # Create result dictionary with tensors
        result = {
            'dep_points': torch.tensor(xyz_dep, dtype=torch.float32),
            'dep_pnt_attr': torch.tensor(dep_pnt_attr, dtype=torch.float32),
            'dep_meta': dep_meta,
            'uav_points': torch.tensor(xyz_uav, dtype=torch.float32),
            'uav_pnt_attr': torch.tensor(uav_pnt_attr, dtype=torch.float32),
            'uav_meta': uav_meta,
            'bbox': bbox,
            'tile_id': tile_id
        }
        
        # Print progress with overall count if total_tiles is provided
        if total_tiles > 0:
            print(f"Completed tile {overall_tile_index}/{total_tiles} (ID: {tile_id}): UAV points: {xyz_uav.shape[0]:,}, 3DEP points: {xyz_dep.shape[0]:,}")
        else:
            print(f"Successfully processed tile {tile_id}: UAV points: {xyz_uav.shape[0]:,}, 3DEP points: {xyz_dep.shape[0]:,}")
        
        return result
        
    except Exception as e:
        print(f"Error processing bounding box {tile_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Always clean up resources, even if an exception occurred
        if transformer is not None:
            del transformer
            transformer = None
        
        if uav_pc is not None:
            for pc in uav_pc:
                del pc
            uav_pc = None
            
        if dep_pc is not None:
            for pc in dep_pc:
                del pc
            dep_pc = None
            
        # Clean up numpy arrays
        if xyz_uav is not None:
            del xyz_uav
        if xyz_dep is not None:
            del xyz_dep
        if uav_pnt_attr is not None:
            del uav_pnt_attr
        if dep_pnt_attr is not None:
            del dep_pnt_attr
        if bbox_wgs84 is not None:
            del bbox_wgs84
            
        # Force garbage collection
        gc.collect()


def chunk_data(data, chunk_size):
    """Split data into smaller chunks of size `chunk_size`."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def process_chunk(
    chunk, 
    chunk_index, 
    total_chunks,
    start_date, 
    end_date, 
    stac_source, 
    bbox_crs, 
    max_threads, 
    output_dir, 
    results_tracker,
    max_api_retries, 
    initial_retry_delay,
    current_tile_index=0,
    total_tiles=0,
    verbose=False
):
    """
    Process a single chunk of tiles.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Processing chunk {chunk_index}/{total_chunks}...")
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, tile_id, bbox, start_date, end_date, stac_source, bbox_crs, max_api_retries, initial_retry_delay,
         current_tile_index + i, total_tiles, verbose)
        for i, (tile_id, bbox) in enumerate(chunk, start=1)
    ]
    
    chunk_results = []
    future_to_tile = {}
    
    try:
        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            future_to_tile = {
                executor.submit(process_bbox, args): (i, tile_id) 
                for i, (args, (i, tile_id)) in enumerate(
                    zip(args_list, enumerate([tid for tid, _ in chunk], start=1))
                )
            }
            
            for future in future_to_tile:
                i, tile_id = future_to_tile[future]
                try:
                    result = future.result()
                    if result is not None:
                        chunk_results.append(result)
                        results_tracker['successful'].append((tile_id, "Successfully processed"))
                    else:
                        # Failed to process tile
                        results_tracker['failed'].append((tile_id, "Failed to process - see logs for details"))
                except Exception as e:
                    results_tracker['failed'].append((tile_id, f"Error: {str(e)[:100]}..."))  # Truncate very long error messages
            
    except Exception as e:
        print(f"Error during parallel processing of chunk {chunk_index}: {e}")
    
    # Write chunk results to disk
    if chunk_results:
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        chunk_file = os.path.join(output_dir, f"chunk_{timestamp}.pt")
        torch.save(chunk_results, chunk_file)
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{end_timestamp}] Saved chunk {chunk_index} with {len(chunk_results)} results to {chunk_file}")
    else:
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{end_timestamp}] No results for chunk {chunk_index}. Skipping saving.")

    # Clean up memory
    del args_list
    del chunk_results
    del future_to_tile
    gc.collect()


def print_summary_report(results_tracker, total_tiles):
    """
    Print a summary report of processing results.
    """
    successful_count = len(results_tracker['successful'])
    failed_count = len(results_tracker['failed'])
    
    print(f"Total tiles processed: {total_tiles}")
    print(f"Successfully processed: {successful_count} ({successful_count/total_tiles*100:.1f}%)")
    print(f"Failed to process: {failed_count} ({failed_count/total_tiles*100:.1f}%)")
    
    if failed_count > 0:
        print("\nFailed tiles:")
        for tile_id, reason in results_tracker['failed'][:20]:  # Show first 20 failures
            print(f"  - Tile {tile_id}: {reason}")
        
        if failed_count > 20:
            print(f"  ... and {failed_count - 20} more failures (not shown)")


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
    initial_retry_delay=2,
    retry_failed=True,  # New parameter to control retry behavior
    verbose=False  # Control level of logging detail
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
    retry_failed : bool, optional
        Whether to retry processing failed tiles after the initial processing is complete.
        Default is True.
    verbose : bool, optional
        Control the level of logging detail. If False, only show completion logs.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting processing of {tiles_geojson}")
    
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
    
    total_tiles = len(tile_bounding_boxes)
    print(f"Processing {total_tiles} tiles from {tiles_geojson}")
    
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

    # Process all chunks
    start_time = datetime.now()
    for chunk_index, chunk in enumerate(chunks, start=1):
        # Calculate the current position in the overall tile list
        current_tile_index = (chunk_index - 1) * chunk_size
        
        process_chunk(
            chunk, 
            chunk_index, 
            len(chunks),
            start_date, 
            end_date, 
            stac_source, 
            bbox_crs, 
            max_threads, 
            output_dir, 
            results_tracker,
            max_api_retries, 
            initial_retry_delay,
            current_tile_index,
            total_tiles,
            verbose
        )
        
        # Give the system a moment to recover between chunks
        time.sleep(2)
        
        # Force garbage collection between chunks
        gc.collect()

    # Calculate processing time
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    # Print summary report after initial processing
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] " + "="*80)
    print("INITIAL PROCESSING SUMMARY REPORT")
    print("="*80)
    print_summary_report(results_tracker, total_tiles)
    print(f"Processing time: {processing_time}")

    # Retry failed tiles if requested and if there are any failed tiles
    if retry_failed and results_tracker['failed']:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] " + "="*80)
        print("RETRYING FAILED TILES")
        print("="*80)
        
        # Extract the failed tile IDs and find their corresponding bounding boxes
        failed_tiles = []
        original_tile_id_to_bbox = dict(zip(tile_ids, tile_bounding_boxes))
        
        for tile_id, _ in results_tracker['failed']:
            # Get the bounding box for this tile ID
            if tile_id in original_tile_id_to_bbox:
                failed_tiles.append((tile_id, original_tile_id_to_bbox[tile_id]))
        
        print(f"Retrying {len(failed_tiles)} failed tiles...")
        
        # Create a new results tracker for the retry
        retry_results_tracker = {
            'successful': [],
            'failed': []
        }
        
        # Create a directory for retry results
        retry_output_dir = os.path.join(output_dir, "retries")
        os.makedirs(retry_output_dir, exist_ok=True)
        
        # Process the failed tiles in smaller chunks to reduce memory pressure
        retry_chunk_size = min(chunk_size // 2, 10)  # Use smaller chunks for retries
        retry_chunks = list(chunk_data(failed_tiles, retry_chunk_size))
        
        retry_start_time = datetime.now()
        for chunk_index, chunk in enumerate(retry_chunks, start=1):
            process_chunk(
                chunk, 
                chunk_index, 
                len(retry_chunks),
                start_date, 
                end_date, 
                stac_source, 
                bbox_crs, 
                max(max_threads // 2, 1),  # Use fewer threads for retries
                retry_output_dir, 
                retry_results_tracker,
                max_api_retries * 2,  # Double the retry attempts for failed tiles
                initial_retry_delay * 2,  # Use longer initial delay for retries
                0,  # Reset to 0 for retry chunks
                len(failed_tiles),
                verbose
            )
            
            # Give the system more time to recover between retry chunks
            time.sleep(5)
            
            # Force garbage collection between chunks
            gc.collect()
        
        # Calculate retry processing time
        retry_end_time = datetime.now()
        retry_processing_time = retry_end_time - retry_start_time
        
        # Print summary report after retries
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] " + "="*80)
        print("RETRY RESULTS SUMMARY")
        print("="*80)
        print_summary_report(retry_results_tracker, len(failed_tiles))
        print(f"Retry processing time: {retry_processing_time}")
        
        # Update the overall results tracker with retry results
        # Add the successful retries to the overall successful list
        results_tracker['successful'].extend(retry_results_tracker['successful'])
        
        # Update the failed list to only include tiles that still failed after retrying
        retry_failed_ids = [tile_id for tile_id, _ in retry_results_tracker['failed']]
        still_failed = []
        
        for tile_id, reason in results_tracker['failed']:
            if tile_id in retry_failed_ids:
                # Find the updated reason from the retry
                for retry_tile_id, retry_reason in retry_results_tracker['failed']:
                    if retry_tile_id == tile_id:
                        still_failed.append((tile_id, retry_reason))
                        break
            else:
                # This tile was successfully processed in the retry
                pass
        
        # Update the failed list
        results_tracker['failed'] = still_failed
        
        # Print final summary report
        total_processing_time = datetime.now() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] " + "="*80)
        print("FINAL PROCESSING SUMMARY (AFTER RETRIES)")
        print("="*80)
        print_summary_report(results_tracker, total_tiles)
        print(f"Total processing time (including retries): {total_processing_time}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] " + "="*80)
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
    
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="Disable retrying of failed tiles after initial processing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows more details during processing)"
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
        initial_retry_delay=args.initial_retry_delay,
        retry_failed=not args.no_retry_failed,  # By default, retry failed tiles unless --no-retry-failed is specified
        verbose=args.verbose  # By default, less verbose output unless --verbose is specified
    )