import geopandas as gpd
from shapely.wkt import loads
import os
import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import shape
import pyproj
from shapely.ops import transform, unary_union
from create_training_tile_bboxes import extract_geotiff_outline



def extract_dem_outlines(dem_paths, output_file, simplify_tolerance=None, out_CRS=None):
    """
    Extract outlines from multiple DEMs and combine them into a single GeoJSON file.
    
    Parameters:
    -----------
    dem_paths : list
        List of paths to the DEM files.
    output_file : str
        Path to the output GeoJSON file.
    simplify_tolerance : float, optional
        Tolerance for simplifying the polygons.
    out_CRS : str or pyproj.CRS, optional
        Target CRS for the output geometries. If None, the CRS of the first DEM is used.
        
    Returns:
    --------
    GeoDataFrame
        A GeoDataFrame containing the DEM outlines.
    """
    # Create an empty list to store each DEM's data
    dem_data = []
    default_crs = None
    
    # Process each DEM
    for dem_path in dem_paths:
        # Get the DEM name from the file path
        dem_name = os.path.basename(dem_path)
        
        # Read the CRS from the first DEM to use as default if not specified
        if default_crs is None and out_CRS is None:
            with rasterio.open(dem_path) as src:
                default_crs = src.crs
        
        # Extract the outline
        wkt_outline = extract_geotiff_outline(dem_path, simplify_tolerance, out_CRS)
        
        # Convert WKT to geometry
        geometry = loads(wkt_outline)
        
        # Calculate area of the geometry
        if geometry.is_empty:
            area = 0
        else:
            # If the target CRS is geographic (like EPSG:4326), calculate area using geodesic method
            if out_CRS is not None and pyproj.CRS.from_user_input(out_CRS).is_geographic:
                # Create a temporary GeoDataFrame with the geometry to use GeoPandas' area calculation
                temp_gdf = gpd.GeoDataFrame([{'geometry': geometry}], geometry='geometry', crs=out_CRS)
                # Calculate geodesic area in square meters
                area = temp_gdf.to_crs('+proj=cea').area.iloc[0]  # Equal area projection for accurate area
            else:
                # For projected CRS, direct calculation is fine
                area = geometry.area
            
        # Add to the list with area attribute
        dem_data.append({
            'name': dem_name,
            'path': dem_path,
            'area': area,
            'geometry': geometry
        })
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(dem_data, geometry='geometry')
    
    # Set CRS
    if out_CRS is not None:
        gdf.crs = out_CRS
    elif default_crs is not None:
        gdf.crs = default_crs
    
    # Save to GeoJSON
    gdf.to_file(output_file, driver='GeoJSON')
    
    return gdf

# Example usage:
if __name__ == "__main__":
    # Define study DEMs for outline extraction
    study_dems = [
        'data/processed/dems/sedgwick/T01-T09_LiDAR_20230928_Pre_LAS_classified_dsm_1.0m.tif',
        'data/processed/dems/sedgwick/T03-T13_LIDAR_20231025_Pre_LAS_classified_dsm_1.0m.tif',
        'data/processed/dems/sedgwick/T06-T14_LIDAR_20231025_Pre_LAS_classified_dsm_1.0m.tif',
        'data/processed/dems/sedgwick/TREX_LIDAR_20230630_Pre_LAS_classified_dsm_1.0m.tif',
        'data/processed/dems/volcan/VolcanMt_20231025_LAS_classified_dsm_1.0m.tif'
    ]

    # Define output file for DEM outlines
    dem_outlines_output_file = 'data/processed/dem_outlines.geojson'

    # Extract DEM outlines and save to GeoJSON
    dem_outlines = extract_dem_outlines(
        study_dems, 
        dem_outlines_output_file,
        simplify_tolerance=1.0,  # Adjust as needed based on your data resolution
        out_CRS='EPSG:4326'  # WGS84 - standard for GeoJSON and good for QGIS
    )

    print(f"DEM outlines saved to {dem_outlines_output_file}")