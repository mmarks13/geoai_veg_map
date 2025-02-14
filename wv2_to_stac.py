#!/usr/bin/env python3
"""
wv2_to_stac.py

This script processes WorldView-2 imagery by:
  - Extracting metadata (acquisition datetime and spatial footprint) from XML files.
  - Filtering images based on whether their footprints overlap any of the specified input bounding boxes.
  - Downloading a DEM for each image footprint.
  - Orthorectifying the imagery using RPC information and the downloaded DEM.
  - Converting the orthorectified image into single-band Cloud Optimized GeoTIFFs (COGs).
  - Creating a STAC (SpatioTemporal Asset Catalog) catalog with items for each processed image.

Usage from the terminal:
    python wv2_to_stac.py --input-dir path/to/ntf_xml_dir \
                          --output-dir path/to/cog_dir \
                          --stac-output path/to/stac_catalog_dir \
                          --input-bboxes "-120.0,35.0,-115.0,40.0" "-100.0,30.0,-95.0,35.0"

Usage from a Jupyter Notebook:
    >>> from wv2_to_stac import run_wv2_to_stac
    >>> catalog = run_wv2_to_stac(input_dir="path/to/ntf_xml_dir",
    ...                           output_dir="path/to/cog_dir",
    ...                           stac_output="path/to/stac_catalog_dir",
    ...                           input_bboxes=["-120.0,35.0,-115.0,40.0", "-100.0,30.0,-95.0,35.0"])
    >>> catalog  # inspect the catalog object
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import argparse

# Import PySTAC components
from pystac import Catalog, Collection, Item, Asset, MediaType, CatalogType, Extent, SpatialExtent, TemporalExtent
import elevation

# WorldView-2 Band Names in order.
WV2_BAND_NAMES = [
    "coastal",  # Band 1
    "blue",     # Band 2
    "green",    # Band 3
    "yellow",   # Band 4
    "red",      # Band 5
    "rededge",  # Band 6
    "nir1",     # Band 7
    "nir2"      # Band 8
]

def orthorectify_ntf_with_rpc(ntf_path, output_path, dem_path=None, target_srs="EPSG:32610", res=1.84):
    """
    Orthorectify an NTF image using RPCs with gdalwarp.

    Parameters:
      ntf_path (str): Path to the input NTF file.
      output_path (str): Path for the output orthorectified GeoTIFF.
      dem_path (str): Path to the DEM file (used for RPC correction).
      target_srs (str): Target spatial reference system (default is 'EPSG:32610').
      res (float): Resolution (pixel size) for the output image.
    """
    cmd = [
        "gdalwarp",
        "--config", "NITF_J2K_DRIVER", "JP2OPENJPEG",
        "-of", "GTiff",
        "-rpc",
        "-t_srs", target_srs,
        "-tr", str(res), str(res),
        "-overwrite"
    ]
    if dem_path:
        cmd += ["-to", f"RPC_DEM={dem_path}"]
    cmd += [ntf_path, output_path]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Orthorectification complete: {output_path}")

def extract_metadata_from_xml(xml_path):
    """
    Extract metadata from an XML file associated with a WorldView-2 image.

    This function extracts the acquisition datetime and spatial footprint coordinates.
    
    Parameters:
      xml_path (str): Path to the XML file.

    Returns:
      tuple: (acquisition_datetime, bbox, geometry) where:
          - acquisition_datetime (datetime): The image acquisition time.
          - bbox (list): Bounding box in the format [min_lon, min_lat, max_lon, max_lat].
          - geometry (dict): A GeoJSON-like polygon representing the image footprint.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract acquisition datetime from <FIRSTLINETIME>
    first_line_time = root.find(".//FIRSTLINETIME").text
    acquisition_datetime = datetime.fromisoformat(first_line_time.replace("Z", "+00:00"))

    # Extract coordinates from the BAND_C element
    band_c = root.find(".//BAND_C")
    ULLON = float(band_c.find("ULLON").text)
    ULLAT = float(band_c.find("ULLAT").text)
    URLON = float(band_c.find("URLON").text)
    URLAT = float(band_c.find("URLAT").text)
    LRLON = float(band_c.find("LRLON").text)
    LRLAT = float(band_c.find("LRLAT").text)
    LLLON = float(band_c.find("LLLON").text)
    LLLAT = float(band_c.find("LLLAT").text)

    # Calculate the bounding box of the image footprint
    min_lon = min(ULLON, URLON, LRLON, LLLON)
    max_lon = max(ULLON, URLON, LRLON, LLLON)
    min_lat = min(ULLAT, URLAT, LRLAT, LLLAT)
    max_lat = max(ULLAT, URLAT, LRLAT, LLLAT)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # Construct a polygon geometry (GeoJSON-like) from the coordinates.
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [ULLON, ULLAT],
            [URLON, URLAT],
            [LRLON, LRLAT],
            [LLLON, LLLAT],
            [ULLON, ULLAT]
        ]]
    }

    return acquisition_datetime, bbox, geometry

def download_dem_for_bbox(bbox, dem_path, buffer=0.05):
    """
    Download a DEM for the given bounding box using the elevation package.

    The bounding box is expanded slightly by the specified buffer.

    Parameters:
      bbox (list): Bounding box as [west, south, east, north].
      dem_path (str): Output path for the DEM file.
      buffer (float): Buffer (in degrees) to expand the bbox.
    """
    west, south, east, north = bbox

    # Expand the bbox slightly to ensure full coverage.
    west -= buffer
    south -= buffer
    east += buffer
    north += buffer

    print(f"Downloading DEM for bbox with buffer: [{west}, {south}, {east}, {north}]")
    elevation.clip(bounds=(west, south, east, north), output=dem_path, product="SRTM3")

def create_single_band_cogs(multiband_path, output_dir):
    """
    Create single-band Cloud Optimized GeoTIFFs (COGs) from a multiband orthorectified image.

    Parameters:
      multiband_path (str): Path to the input multiband GeoTIFF.
      output_dir (str): Directory where individual band COGs will be saved.

    Returns:
      list: A list of tuples (band_name, cog_path) for each created COG.
    """
    band_cogs = []
    for i, band_name in enumerate(WV2_BAND_NAMES, start=1):
        # Build output path for the single-band COG.
        cog_path = os.path.join(output_dir, f"{Path(multiband_path).stem}_{band_name}.tif")
        subprocess.check_call([
            "gdal_translate",
            "-b", str(i),
            "-of", "COG",
            "-co", "COMPRESS=LZW",
            multiband_path,
            cog_path
        ])
        band_cogs.append((band_name, cog_path))
    return band_cogs

def bboxes_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    
    Each bounding box is defined as [min_lon, min_lat, max_lon, max_lat].

    Returns:
      bool: True if the bounding boxes overlap; otherwise, False.
    """
    return not (
        bbox1[2] < bbox2[0] or  # bbox1 is completely to the left of bbox2
        bbox1[0] > bbox2[2] or  # bbox1 is completely to the right of bbox2
        bbox1[3] < bbox2[1] or  # bbox1 is completely below bbox2
        bbox1[1] > bbox2[3]     # bbox1 is completely above bbox2
    )

def overlaps_any(bbox, bbox_list):
    """
    Check if a given bounding box overlaps any bounding box in a list.

    Parameters:
      bbox (list): The bounding box to test.
      bbox_list (list): A list of bounding boxes to check against.

    Returns:
      bool: True if there is an overlap with any bbox in bbox_list; otherwise, False.
    """
    return any(bboxes_overlap(bbox, input_bbox) for input_bbox in bbox_list)

def process_image(ntf_file, input_dir, output_dir, input_bboxes, collection, items, all_bboxes, all_times):
    """
    Process a single image by extracting metadata, checking spatial overlap, downloading DEM,
    orthorectifying the image, creating COGs, and adding the result to the STAC catalog.

    Parameters:
      ntf_file (str): Filename of the NTF image.
      input_dir (str): Directory containing the image and its XML metadata.
      output_dir (str): Directory for saving output files.
      input_bboxes (list): List of input bounding boxes to filter images.
      collection (Collection): The PySTAC collection to add items to.
      items (list): List that will store the created STAC items.
      all_bboxes (list): List that collects spatial extents for all processed images.
      all_times (list): List that collects temporal extents for all processed images.
    """
    base_name = ntf_file.rsplit('.', 1)[0]
    xml_file = base_name + ".xml"
    ntf_path = os.path.join(input_dir, ntf_file)
    xml_path = os.path.join(input_dir, xml_file)

    # Skip if there is no corresponding XML metadata file.
    if not os.path.exists(xml_path):
        print(f"No corresponding XML for {ntf_file}, skipping.")
        return

    # Extract metadata from the XML file.
    acquisition_datetime, bbox, geometry = extract_metadata_from_xml(xml_path)
    print("Image bbox:", bbox)

    # Filter: process only if the image footprint overlaps any input bounding box.
    if not overlaps_any(bbox, input_bboxes):
        print(f"Image {ntf_file} does not overlap any input bounding box, skipping.")
        return

    # Download a DEM for the image footprint.
    dem_path = os.path.join(output_dir, f"{base_name}_dem.tif")
    download_dem_for_bbox(bbox, dem_path)

    # Orthorectify the image using RPCs.
    ortho_tif = os.path.join(output_dir, f"{base_name}_ortho.tif")
    orthorectify_ntf_with_rpc(ntf_path, ortho_tif, dem_path=dem_path)

    # Create single-band COGs for each band.
    band_cogs = create_single_band_cogs(ortho_tif, output_dir)

    # Collect spatial and temporal extents.
    all_bboxes.append(bbox)
    all_times.append(acquisition_datetime)

    # Create a unique STAC item for this image.
    item_id = f"{base_name}_{acquisition_datetime.strftime('%Y%m%dT%H%M%S')}"
    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=acquisition_datetime,
        properties={}
    )

    # Add each band as an asset to the item.
    for band_name, cog_path in band_cogs:
        item.add_asset(
            key=band_name,
            asset=Asset(
                href=os.path.relpath(cog_path),
                media_type=MediaType.COG,
                roles=["data"],
                title=f"{band_name.capitalize()} band"
            )
        )

    # Add the item to the collection and the global items list.
    collection.add_item(item)
    items.append(item)

def create_stac_catalog(catalog_id, collection_id, collection_description):
    """
    Create and initialize a STAC catalog and a collection.

    Parameters:
      catalog_id (str): Identifier for the catalog.
      collection_id (str): Identifier for the collection.
      collection_description (str): Description for the collection.

    Returns:
      tuple: (catalog, collection) objects.
    """
    catalog = Catalog(id=catalog_id, description="Local WV2 Imagery Catalog")
    collection = Collection(
        id=collection_id,
        description=collection_description,
        extent=None,
        license="proprietary"
    )
    catalog.add_child(collection)
    return catalog, collection

def update_collection_extent(collection, all_bboxes, all_times):
    """
    Update the spatial and temporal extents of the STAC collection based on the processed items.

    Parameters:
      collection (Collection): The PySTAC collection to update.
      all_bboxes (list): List of bounding boxes from processed images.
      all_times (list): List of acquisition datetimes from processed images.
    """
    if all_bboxes and all_times:
        minx = min(b[0] for b in all_bboxes)
        miny = min(b[1] for b in all_bboxes)
        maxx = max(b[2] for b in all_bboxes)
        maxy = max(b[3] for b in all_bboxes)
        temporal_start = min(all_times)
        temporal_end = max(all_times)
        collection.extent = Extent(
            SpatialExtent([[minx, miny, maxx, maxy]]),
            TemporalExtent([[temporal_start, temporal_end]])
        )
    else:
        now = datetime.utcnow()
        collection.extent = Extent(
            SpatialExtent([[0, 0, 0, 0]]),
            TemporalExtent([[now, now]])
        )

def parse_args():
    """
    Parse command-line arguments.

    Returns:
      Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process WorldView-2 imagery into a STAC catalog."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="imagery/worldview2/orig",
        help="Input directory containing NTF and XML files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="imagery/worldview2/cog",
        help="Output directory for COG files."
    )
    parser.add_argument(
        "--stac-output",
        type=str,
        default="wv2_stac",
        help="Directory for the STAC catalog."
    )
    parser.add_argument(
        "--input-bboxes",
        type=str,
        nargs="+",
        default=["-120.0,35.0,-115.0,40.0", "-100.0,30.0,-95.0,35.0"],
        help=(
            "Input bounding boxes in format min_lon,min_lat,max_lon,max_lat. "
            "Provide one or more separated by spaces."
        )
    )
    return parser.parse_args()

def parse_bbox(bbox_str):
    """
    Parse a bounding box string into a list of floats.

    Parameters:
      bbox_str (str): Bounding box as a comma-separated string (min_lon,min_lat,max_lon,max_lat).

    Returns:
      list: A list of four floats representing the bounding box.
    """
    parts = bbox_str.split(',')
    if len(parts) != 4:
        raise ValueError("Bounding box must have four comma-separated values: min_lon,min_lat,max_lon,max_lat")
    return [float(x) for x in parts]

def run_wv2_to_stac(input_dir, output_dir, stac_output, input_bboxes):
    """
    Run the complete processing workflow. This function is designed to be called
    directly from a Python script or a Jupyter Notebook.

    Parameters:
      input_dir (str): Directory containing the NTF and XML files.
      output_dir (str): Directory where COGs and intermediate files will be stored.
      stac_output (str): Directory where the STAC catalog will be saved.
      input_bboxes (list): A list of bounding boxes to filter images. Each element can be a string
                           ("min_lon,min_lat,max_lon,max_lat") or a list/tuple of four numbers.

    Returns:
      Catalog: The PySTAC catalog object.
    """
    # Resolve absolute paths and create output directories if needed.
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    stac_output = os.path.abspath(stac_output)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stac_output, exist_ok=True)

    # Parse input bounding boxes.
    parsed_input_bboxes = []
    for b in input_bboxes:
        if isinstance(b, str):
            parsed_input_bboxes.append(parse_bbox(b))
        elif isinstance(b, (list, tuple)):
            if len(b) != 4:
                raise ValueError("Bounding box must have 4 elements")
            parsed_input_bboxes.append([float(x) for x in b])
        else:
            raise ValueError("Invalid input bounding box type: must be a string or a list/tuple of 4 numbers")
    print("Processing with input bounding boxes:", parsed_input_bboxes)

    # Create the STAC catalog and collection.
    catalog, collection = create_stac_catalog(
        catalog_id="local-wv2-catalog",
        collection_id="worldview2-l1b",
        collection_description="WorldView-2 Orthorectified Imagery"
    )

    # List all NTF files in the input directory.
    ntf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".ntf")]
    items = []      # To store the created STAC items.
    all_bboxes = [] # To collect bounding boxes for extent calculation.
    all_times = []  # To collect acquisition times for extent calculation.

    # Process each image.
    for ntf_file in ntf_files:
        process_image(ntf_file, input_dir, output_dir, parsed_input_bboxes, collection, items, all_bboxes, all_times)

    # Update the collection's spatial and temporal extents.
    update_collection_extent(collection, all_bboxes, all_times)

    # Save the STAC catalog.
    catalog.normalize_hrefs(stac_output)
    catalog.save(catalog_type=CatalogType.SELF_CONTAINED)
    print(f"STAC Catalog saved to {stac_output}")

    # Return the catalog object for further inspection if desired.
    return catalog

def main():
    """
    Main function for terminal execution.
      - Parses command-line arguments.
      - Runs the processing workflow.
      - Saves the STAC catalog.
    """
    args = parse_args()

    # Run the processing workflow using command-line arguments.
    run_wv2_to_stac(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stac_output=args.stac_output,
        input_bboxes=args.input_bboxes
    )

if __name__ == '__main__':
    main()
