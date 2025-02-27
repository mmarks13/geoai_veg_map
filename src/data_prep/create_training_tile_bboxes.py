#!/usr/bin/env python3
"""
Training Data Tile Creation Script

This script extracts outlines from Digital Elevation Models (DEMs) and creates 
analysis tiles with the following steps:
1. Extracting outlines from specified DEMs
2. Creating tiles from these outlines
3. Saving the tiles as a GeoJSON file

The script processes data from Sedgwick Reserve and Volcan Mountain locations.

Usage:
    python create_tiles.py
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import json
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping, MultiPoint, Polygon, LineString, Point, box
from shapely import wkt
import numpy as np



# Define study DEMs for outline extraction
study_dems = [
    'data/processed/dems/sedgwick/T01-T09_LiDAR_20230928_Pre_LAS_classified_dsm_1.0m.tif',
    'data/processed/dems/sedgwick/T03-T13_LIDAR_20231025_Pre_LAS_classified_dsm_1.0m.tif',
    'data/processed/dems/sedgwick/T06-T14_LIDAR_20231025_Pre_LAS_classified_dsm_1.0m.tif',
    'data/processed/dems/sedgwick/TREX_LIDAR_20230630_Pre_LAS_classified_dsm_1.0m.tif',
    'data/processed/dems/volcan/VolcanMt_20231025_LAS_classified_dsm_1.0m.tif'
]


# Output file for tiles
tiles_output_file = 'data/processed/tiles.geojson'

# Tile creation parameters
TILE_WIDTH = 10
TILE_HEIGHT = 10
OVERLAP_RATIO = 0.15
REQUIRE_FULL_CONTAINMENT = True



def extract_geotiff_outline(geotiff_path, simplify_tolerance=None, out_CRS=None):
    """
    Extract the outline of valid data from a GeoTIFF file and return its WKT representation.
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the GeoTIFF file.
    simplify_tolerance : float, optional
        Tolerance for simplifying the polygon. Higher values result in more simplification.
        If None, no simplification is applied.
    out_CRS : str or pyproj.CRS, optional
        Target CRS for the output geometry. This can be an EPSG code ('EPSG:4326'),
        a WKT string, or any other format accepted by pyproj.CRS.
        If None, the original CRS of the GeoTIFF is used.
        
    Returns:
    --------
    str
        WKT (Well-Known Text) representation of the outline in the specified CRS.
    """
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Read the raster data
        image = src.read(1)
        
        # Create a mask of valid data (non-nodata values)
        # If the dataset has a defined nodata value, use it
        if src.nodata is not None:
            mask = image != src.nodata
        else:
            # Otherwise, assume 0 is nodata (this can be adjusted based on your data)
            mask = image != 0
        
        # Get the raster's transformation parameters
        transform = src.transform
        
        # Generate shapes from the mask
        shapes = features.shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform
        )
        
        # Create a polygon from the shapes
        all_polygons = []
        for geom, value in shapes:
            if value == 1:  # Valid data areas
                polygon = shape(geom)
                if not polygon.is_empty:
                    all_polygons.append(polygon)
        
        # Combine all polygons into a single multipolygon (or just use the first/largest polygon)
        if all_polygons:
            if len(all_polygons) == 1:
                outline = all_polygons[0]
            else:
                from shapely.ops import unary_union
                outline = unary_union(all_polygons)
            
            # Apply simplification if requested
            if simplify_tolerance is not None:
                outline = outline.simplify(simplify_tolerance, preserve_topology=True)
            
            # Transform to the target CRS if specified
            if out_CRS is not None:
                # Get the source CRS from the GeoTIFF
                src_crs = src.crs
                
                # Check if the source has a valid CRS
                if src_crs is None:
                    raise ValueError("Source GeoTIFF does not have a defined CRS, cannot transform coordinates.")
                
                # Import pyproj for CRS transformation
                import pyproj
                from shapely.ops import transform
                
                try:
                    # Convert both CRS to pyproj.CRS objects for comparison
                    src_pyproj_crs = pyproj.CRS.from_user_input(src_crs)
                    out_pyproj_crs = pyproj.CRS.from_user_input(out_CRS)
                    
                    # Only transform if the CRSs are different
                    if src_pyproj_crs != out_pyproj_crs:
                        project = pyproj.Transformer.from_crs(
                            src_pyproj_crs, out_pyproj_crs, always_xy=True).transform
                        outline = transform(project, outline)
                except Exception as e:
                    raise ValueError(f"Error in CRS transformation: {e}")
            
            # Return the WKT representation
            return outline.wkt
        else:
            return "POLYGON EMPTY"



def create_tiles_from_wkt(wkt_polygon, tile_width, tile_height, overlap_ratio, require_full_containment=True):
    """
    Creates a list of tile geometries within a given WKT polygon.

    Parameters:
        wkt_polygon (str): A WKT string representing a polygon (any shape).
        tile_width (float): The width of each tile.
        tile_height (float): The height of each tile.
        overlap_ratio (float): The fraction of overlap between adjacent tiles (0 to 1).
        require_full_containment (bool): If True, only include tiles fully inside the polygon.
                                         If False, include tiles that intersect the polygon.

    Returns:
        list: A list of shapely.geometry.box objects representing the tile bounding boxes.
    """
    # Parse the WKT polygon into a shapely geometry.
    polygon = wkt.loads(wkt_polygon)
    
    # Use the polygon's bounding box for tiling.
    minx, miny, maxx, maxy = polygon.bounds
    
    # Validate the overlap ratio.
    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be between 0 and 1 (non-inclusive of 1)")
    
    # Calculate step sizes (e.g., with an overlap_ratio of 0.5, step = half the tile size).
    step_x = tile_width * (1 - overlap_ratio)
    step_y = tile_height * (1 - overlap_ratio)
    
    tiles = []
    x = minx

    # Generate candidate tiles across the bounding box.
    while x + tile_width <= maxx:
        y = miny
        while y + tile_height <= maxy:
            candidate_tile = box(x, y, x + tile_width, y + tile_height)
            
            if require_full_containment:
                # Only add the tile if it is completely inside the polygon.
                if polygon.contains(candidate_tile):
                    tiles.append(candidate_tile)
            else:
                # Add the tile if it at least intersects the polygon.
                if polygon.intersects(candidate_tile):
                    tiles.append(candidate_tile)
            
            y += step_y
        x += step_x

    return tiles


def extract_substring_before_classified(filename):
    """
    Extract the original filename part before '_classified' suffix.
    
    Args:
        filename (str): Filename with potential '_classified' suffix
        
    Returns:
        str: Filename without the '_classified' suffix
    """
    return filename.split('_classified')[0]

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
import contextily as ctx
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def plot_wkt_tiling(wkt_list, 
                    crs=None,
                    figsize=(12, 10),
                    color='blue',
                    edge_color='black',
                    alpha=0.2,
                    linewidth=0.5,
                    title='Dataset Tiling Structure',
                    basemap=True,
                    output_path=None,
                    sample_size=None,
                    custom_colors=None,
                    custom_alphas=None,
                    custom_edge_colors=None,
                    custom_linewidths=None,
                    legend_labels=None):
    """
    Plot a large number of WKT strings efficiently, preserving all internal boundaries.
    Optimized for visualizing tiling structures where each WKT represents a tile.
    
    Parameters:
    -----------
    wkt_list : list
        List of WKT strings to plot (typically tile boundaries).
    crs : str or dict, optional
        Coordinate Reference System of the input WKT strings.
        Can be anything accepted by pyproj.CRS.from_user_input(), such as an 
        authority string (e.g. 'EPSG:4326') or a WKT string.
        If None, geometries are assumed to be in WGS 84 (EPSG:4326).
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (12, 10).
    color : str, optional
        Face color for the tiles. Default is 'blue'.
    edge_color : str, optional
        Edge color for the tiles. Default is 'black'.
    alpha : float, optional
        Transparency of the tiles. Default is 0.2.
    linewidth : float, optional
        Width of the tile boundaries. Default is 0.5.
    title : str, optional
        Title of the plot. Default is 'Dataset Tiling Structure'.
    basemap : bool, optional
        Whether to include a basemap. Default is True.
    output_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    sample_size : int, optional
        If provided, only a random sample of this size will be plotted.
        Useful for previewing very large datasets.
    custom_colors : list, optional
        List of colors for each geometry. If provided, overrides the color parameter.
    custom_alphas : list, optional
        List of alpha values for each geometry. If provided, overrides the alpha parameter.
    custom_edge_colors : list, optional
        List of edge colors for each geometry. If provided, overrides the edge_color parameter.
    custom_linewidths : list, optional
        List of linewidths for each geometry. If provided, overrides the linewidth parameter.
    legend_labels : list, optional
        List of labels for the legend. If provided, a legend will be added to the plot.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object.
    """
    print(f"Processing {len(wkt_list)} WKT strings...")
    
    # Sample data if needed
    if sample_size is not None and sample_size < len(wkt_list):
        print(f"Sampling {sample_size} tiles from the dataset...")
        indices = np.random.choice(len(wkt_list), sample_size, replace=False)
        wkt_list = [wkt_list[i] for i in indices]
    
    # Convert WKT strings to Shapely geometries
    try:
        geometries = [wkt.loads(wkt_str) for wkt_str in wkt_list]
        print("WKT strings loaded successfully.")
    except Exception as e:
        print(f"Error loading WKT strings: {e}")
        return None
    
    # Create a GeoDataFrame with all individual geometries
    print("Creating GeoDataFrame...")
    if crs is None:
        # Default to WGS 84 if no CRS is provided
        crs = "EPSG:4326"
        print("No CRS provided, defaulting to WGS 84 (EPSG:4326)")
    
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    
    # Calculate the combined boundary to determine the plot extent
    print("Calculating plot extent...")
    bounding_box = gdf.total_bounds
    
    # Create figure and axis
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if custom colors/alphas are provided
    if custom_colors and len(custom_colors) == len(gdf):
        # Plot each geometry with its own color/alpha/style
        for i, geom in enumerate(gdf.geometry):
            c = custom_colors[i] if custom_colors else color
            a = custom_alphas[i] if custom_alphas else alpha
            ec = custom_edge_colors[i] if custom_edge_colors else edge_color
            lw = custom_linewidths[i] if custom_linewidths else linewidth
            
            # Create a temporary GeoDataFrame with just this geometry
            temp_gdf = gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)
            
            # Plot with or without label for legend
            if legend_labels and i < len(legend_labels) and (i == 0 or legend_labels[i] != legend_labels[i-1]):
                temp_gdf.plot(ax=ax, color=c, alpha=a, edgecolor=ec, linewidth=lw, label=legend_labels[i])
            else:
                temp_gdf.plot(ax=ax, color=c, alpha=a, edgecolor=ec, linewidth=lw)
    else:
        # Plot all geometries with the same style
        gdf.plot(ax=ax, color=color, alpha=alpha, edgecolor=edge_color, linewidth=linewidth)
    
    # Add basemap if requested
    if basemap:
        try:
            print("Adding basemap...")
            # Check if the CRS is already set
            if gdf.crs is None:
                print("Warning: CRS was not set properly. Defaulting to WGS 84 (EPSG:4326)")
                gdf.set_crs(epsg=4326, inplace=True)
            
            # Reproject to Web Mercator for contextily
            print(f"Reprojecting from {gdf.crs} to Web Mercator (EPSG:3857)")
            gdf_3857 = gdf.to_crs(epsg=3857)
            bounds = gdf_3857.total_bounds
            
            # Calculate zoom level based on the extent
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            dimension = max(width, height)
            
            # Simple zoom level calculation
            if dimension > 5000000:
                zoom = 4
            elif dimension > 1000000:
                zoom = 6
            elif dimension > 100000:
                zoom = 9
            else:
                zoom = 12
                
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
            
            # Set extent with padding
            padding = 0.05  # 5% padding
            ax.set_xlim(bounds[0] - width * padding, bounds[2] + width * padding)
            ax.set_ylim(bounds[1] - height * padding, bounds[3] + height * padding)
        except Exception as e:
            print(f"Could not add basemap: {e}")
            # Fall back to using the GeoDataFrame's bounds
            ax.set_xlim(bounding_box[0], bounding_box[2])
            ax.set_ylim(bounding_box[1], bounding_box[3])
    else:
        # Without basemap, use the GeoDataFrame's bounds
        ax.set_xlim(bounding_box[0], bounding_box[2])
        ax.set_ylim(bounding_box[1], bounding_box[3])
    
    # Add title with tile count
    ax.set_title(f"{title} ({len(wkt_list)} tiles)", fontsize=14)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add a simple grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a note if sampling was used
    if sample_size is not None and sample_size < len(wkt_list):
        plt.figtext(0.5, 0.01, f"Note: Showing random sample of {sample_size} tiles from {len(wkt_list)} total",
                   ha='center', fontsize=9, style='italic')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        print(f"Saving figure to {output_path}...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("Plot created successfully.")
    return fig



def main():
    """Main execution function for creating tiles from DEMs."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(tiles_output_file), exist_ok=True)
    
    print("\n--- Extracting outlines from DEMs ---")
    outlines = []
    for dem in study_dems:
        print(f"Extracting outline from {os.path.basename(dem)}")
        outline = extract_geotiff_outline(dem, out_CRS='EPSG:32611')
        outlines.append(outline)
    
    # Extract original filenames from DEM paths
    filenames = [extract_substring_before_classified(os.path.basename(dem)) for dem in study_dems]
    
    print("\n--- Creating tiles from outlines ---")
    all_tiles = []
    for outline, filename in zip(outlines, filenames):
        print(f"Creating tiles for {filename}")
        tiles = create_tiles_from_wkt(
            outline, 
            tile_width=TILE_WIDTH, 
            tile_height=TILE_HEIGHT, 
            overlap_ratio=OVERLAP_RATIO, 
            require_full_containment=REQUIRE_FULL_CONTAINMENT
        )
        print(f"Created {len(tiles)} tiles for {filename}")
        
        for tile in tiles:
            all_tiles.append({
                "geometry": mapping(tile),
                "properties": {"filename": filename}
            })
    
    # Convert the tiles to GeoJSON format
    geojson_tiles = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": tile["geometry"],
                "properties": tile["properties"]
            } for tile in all_tiles
        ]
    }
    
    # Save the GeoJSON to a file
    print(f"\n--- Saving tiles to GeoJSON ---")
    with open(tiles_output_file, 'w') as f:
        json.dump(geojson_tiles, f)
    
    print(f"Tile creation completed. {len(all_tiles)} tiles saved to {tiles_output_file}")


# Run the script when executed directly
if __name__ == "__main__":
    main()