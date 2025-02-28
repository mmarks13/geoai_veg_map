"""
Point Cloud to Raster Aggregation Utilities

Functions for processing LiDAR point clouds and calculating vegetation structure metrics
as proposed by Moudry et al. (2023). Supports aggregating point cloud data into 
raster format with various statistical measures.

Moudry et al. (2023) proposed ten standardized aerial LiDAR-derived vegetation 
structure variables (e.g., maximum vegetation height, mean vegetation height, 
canopy cover percentages, and foliage height diversity) that should be made available 
in common raster formats to assist ecological research. This module calculates each of 
these standardized variables from point clouds at a chosen standard resolution.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import pdal
import laspy
import rasterio
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely import wkt
import matplotlib.colors as mcolors

import rasterio
from rasterio import features
from shapely.geometry import shape, mapping
import numpy as np




def get_pointcloud_footprint(las_file_path, simplify_tolerance=None, buffer_distance=None):
    """
    Determine the footprint of a point cloud and return it as a WKT string.
    
    This function extracts the X and Y coordinates from a LAS/LAZ file,
    computes the convex hull to get the footprint, and optionally simplifies
    and/or buffers the resulting polygon.
    
    Parameters:
        las_file_path (str): Path to the LAS/LAZ file
        simplify_tolerance (float, optional): Tolerance for simplifying the polygon.
                                             Larger values result in more simplification.
        buffer_distance (float, optional): Distance to buffer the polygon.
                                          Positive values expand, negative values shrink.
    
    Returns:
        str: WKT string representing the point cloud footprint
    """
    # Start timing
    start_time = time.time()
    
    # Load the LAS file
    print(f"Reading point cloud: {os.path.basename(las_file_path)}")
    las = laspy.read(las_file_path)
    
    # Extract X and Y coordinates - use a sample if the point cloud is very large
    num_points = len(las.x)
    if num_points > 200000:
        # Use a random sample of points if the point cloud is large
        sample_size = 200000
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        x_points = las.x[sample_indices]
        y_points = las.y[sample_indices]
        print(f"Using {sample_size} random points from {num_points} total")
    else:
        x_points = las.x
        y_points = las.y
    
    # Create MultiPoint and compute convex hull
    points = MultiPoint([(x, y) for x, y in zip(x_points, y_points)])
    hull = points.convex_hull
    
    # Apply simplification if requested
    if simplify_tolerance is not None:
        hull = hull.simplify(simplify_tolerance)
    
    # Apply buffer if requested
    if buffer_distance is not None:
        hull = hull.buffer(buffer_distance)
    
    # End timing
    end_time = time.time()
    execution_time = np.round(end_time - start_time, 1)
    
    # Check if the result is a valid polygon
    if isinstance(hull, Polygon):
        wkt = hull.wkt
        print(f"Footprint calculated in {execution_time} seconds. Number of vertices: {len(hull.exterior.coords)}")
        return wkt
    else:
        print(f"Warning: Generated footprint is not a polygon. Type: {type(hull)}")
        return hull.wkt


def create_dem(input_las, output_tif, dem_type='dtm', resolution=1.0, window_size=None, 
              create_xml=False):
    """
    Create a Digital Terrain Model (DTM) or Digital Surface Model (DSM) from a classified LAS file.
    
    Parameters:
        input_las (str): Path to the input classified LAS file
        output_tif (str): Path for the output GeoTIFF file
        dem_type (str): Type of Digital Elevation Model to create; either 'dtm' for terrain or 'dsm' for surface
        resolution (float): Output resolution in units of the LAS file (typically meters)
        window_size (int, optional): Window size for interpolation. If None, will be 3x resolution.
        create_xml (bool): If True, allows PDAL to create an XML metadata file alongside the GeoTIFF
                          Set to False to prevent creation of the additional XML file
    
    Returns:
        pdal.Pipeline: Configured PDAL pipeline object
    """
    # Set window size if not specified
    if window_size is None:
        window_size = int(3 * resolution)
    
    # Set up GDAL options
    gdalopts = "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256"
    
    # If XML creation should be suppressed
    if not create_xml:
        gdalopts += ",COPY_SRC_OVERVIEWS=YES,WRITE_METADATA=NO"
    
    # Define the pipeline based on DEM type
    if dem_type.lower() == 'dtm':
        # For DTM, use ground points (class 2) only
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Filter for ground points only
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "output_type": "idw",  # Inverse distance weighted interpolation
                "gdaldriver": "GTiff",
                "resolution": resolution,
                "window_size": window_size,
                "dimension": "Z",  # Use Z dimension for elevation
                "gdalopts": gdalopts
            }
        ]
    elif dem_type.lower() == 'dsm':
        # For DSM, use all points and take the maximum Z in each cell
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "filters.range",
                "limits": "ReturnNumber[1:1]"  # Use first returns for DSM
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "output_type": "max",  # Take highest point in each cell
                "gdaldriver": "GTiff",
                "resolution": resolution,
                "window_size": window_size,
                "dimension": "Z",  # Use Z dimension for elevation
                "gdalopts": gdalopts
            }
        ]
    else:
        raise ValueError("dem_type must be either 'dtm' or 'dsm'")
    
    # Create and return the pipeline
    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)

# PDAL Pipeline Functions

def process_and_classify_las(input_las, output_las, crop_polygon=None, min_hag=0, max_hag=25, filter_noise=False):
    """Process and classify LiDAR data with ground classification and height above ground calculation.
    
    This function creates a PDAL pipeline that:
    1. Reads the LAS file
    2. Optionally crops to a specific polygon
    3. Optionally filters out noise points
    4. Classifies ground points using Simple Morphological Filter (SMRF)
    5. Calculates Height Above Ground (HAG) for each point
    6. Filters points by min/max HAG
    7. Writes the processed data to a new LAS file
    
    Determining which LiDAR returns are from the ground surface is essential for vegetation structure
    analysis. This method follows similar ground classification approaches to those used in 
    the 3DEP program (Pingel et al. 2013).
    """
    pipeline_def = [
        {
            "type": "readers.las",
            "filename": input_las
        }
    ]

    if crop_polygon:
        pipeline_def.append({
            "type": "filters.crop",
            "polygon": crop_polygon
        })

    if filter_noise:
        pipeline_def.append({
            "type": "filters.range",
            "limits": "Classification![7:7], Classification![18:18]"
        })

    pipeline_def.extend([
        {
            "type": "filters.assign",
            "assignment": "Classification[:]=0"
        },
        {
            "type": "filters.smrf"
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "filters.range",
            "limits": f"HeightAboveGround[{min_hag}:{max_hag}]"
        },
        {
            "type": "writers.las",
            "filename": output_las,
            "compression": "laszip",
            "extra_dims":"all"            
        }
    ])

    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)


def process_and_classify_las_to_tif(input_las, output_tif, resolution=1, crop_polygon=None, min_hag=0, max_hag=25, filter_noise=False):
    """Create a PDAL pipeline for generating standard aggregate raster statistics from LiDAR data.
    
    This function creates a raster GeoTIFF using PDAL's writers.gdal with standard statistics:
    - Band 1: Minimum values
    - Band 2: Maximum values
    - Band 3: Mean values
    - Band 4: Inverse Distance Weighted (IDW) values
    - Band 5: Point count
    - Band 6: Standard deviation
    
    Note: This function is more limited than the custom aggregation approach but useful 
    for validation and basic statistics.
    """
    pipeline_def = [
        {
            "type": "readers.las",
            "filename": input_las
        }
    ]

    if crop_polygon:
        pipeline_def.append({
            "type": "filters.crop",
            "polygon": crop_polygon
        })

    if filter_noise:
        pipeline_def.append({
            "type": "filters.range",
            "limits": "Classification![7:7], Classification![18:18]"
        })

    pipeline_def.extend([
        {
            "type": "filters.assign",
            "assignment": "Classification[:]=0"
        },
        {
            "type": "filters.smrf"
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "filters.range",
            "limits": f"HeightAboveGround[{min_hag}:{max_hag}]"
        },
        {
        "type": "writers.gdal",
        "filename": output_tif,
        "gdaldriver": "GTiff",
        "dimension": "HeightAboveGround",
        "output_type": "all",
        "binmode": True,
        "resolution": resolution,
        "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
        }
    ])    

    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)


# Main Aggregation Function

def agg_las_to_array(las_file_path, resolution, dimension="HeightAboveGround", aggregate_func=np.mean, *args, **kwargs):
    """
    Convert a LAS file to a numpy array representing an aggregated raster.
    
    This is the core function that takes a point cloud and creates a raster grid by
    aggregating points that fall within each cell. It allows for custom aggregation
    functions that can return either a single value (creating a 2D raster) or multiple
    values (creating a 3D raster).
    
    The function handles both simple statistics (min, max, mean) and complex metrics
    that may return multiple values per cell (percentiles, density proportions).
    
    Parameters:
        las_file_path: Path to the LAS file
        resolution: Spatial resolution of each cell in X and Y direction
        dimension: LAS point dimension to aggregate (default: "HeightAboveGround")
        aggregate_func: Function to apply to the dimension values
        *args, **kwargs: Additional arguments for aggregate_func
    
    Returns:
        numpy.ndarray: Aggregated raster (2D for single values, 3D for multiple values)
    """
    start_time = time.time()

    las = laspy.read(las_file_path)
    x_points = las.x
    y_points = las.y
    dimension_values = getattr(las, dimension)

    min_x, max_x = np.min(x_points), np.max(x_points)
    min_y, max_y = np.min(y_points), np.max(y_points)

    width = int(np.round((max_x - min_x) / resolution))
    height = int(np.round((max_y - min_y) / resolution))

    x_indices = ((x_points - min_x) / resolution).astype(int)
    y_indices = (height - (y_points - min_y) / resolution).astype(int)

    df = pd.DataFrame({'x_indices': x_indices, 'y_indices': y_indices, 'dimension_values': dimension_values})

    try:
        # For aggregate functions that return a single value per cell (like mean, max, etc.)
        # Group by cell indices and apply the aggregation function to points in each cell
        grouped_df = df.groupby(['x_indices', 'y_indices'])['dimension_values'].agg(
            lambda x: aggregate_func(x.to_numpy(), *args, **kwargs)).reset_index()
        
        # Create empty 2D raster of appropriate size
        raster = np.zeros((height, width))
        
        # Filter out cells that would be outside the raster bounds
        valid_cells = (grouped_df['x_indices'].between(0, width-1)) & (grouped_df['y_indices'].between(0, height-1))
        grouped_df = grouped_df[valid_cells]
        
        # Efficiently assign values to the raster using numpy indexing
        raster[grouped_df['y_indices'].values, grouped_df['x_indices'].values] = grouped_df['dimension_values'].values

    except ValueError as e:
        # Special handling for aggregate functions that return multiple values per cell
        if str(e) == 'Must produce aggregated value':
            # For functions like percentile that return multiple values (e.g., 10th, 25th, 50th percentiles)
            grouped_df = df.groupby(['x_indices', 'y_indices'])['dimension_values'].agg(
                lambda x: aggregate_func(x.to_numpy(), *args, **kwargs).tolist()).reset_index()
            
            # Determine how many values the aggregation function returns per cell
            n_agg_values = len(grouped_df['dimension_values'][0])
            
            # Create empty 3D raster to hold multiple values per cell
            raster = np.zeros((height, width, n_agg_values))

            # Filter out cells outside raster bounds
            valid_cells = (grouped_df['x_indices'].between(0, width-1)) & (grouped_df['y_indices'].between(0, height-1))
            grouped_df = grouped_df[valid_cells]
            
            # Assign values cell by cell (can't use efficient numpy indexing for 3D case)
            for i, row in grouped_df.iterrows():
                if len(row['dimension_values']) > n_agg_values:
                    print(f"Warning: More than {n_agg_values} aggregated values returned. Using first {n_agg_values}.")
                raster[row['y_indices'], row['x_indices']] = row['dimension_values'][0:n_agg_values]
        else:
            # Re-raise any other errors
            raise
    
    end_time = time.time()
    execution_time = np.round(end_time - start_time, 1) 
    n_points = "{:,}".format(np.size(x_points))
    filename = os.path.basename(las_file_path)
    agg_func_name = aggregate_func.__name__
    shape_str = "x".join(map(str, raster.shape))
    print(f"{filename} ({n_points} pts) aggregated via {agg_func_name}() to {shape_str} array. [{execution_time} seconds]")            
    
    return raster

def plot_georeferenced_rasters_with_geometries(tiff_files, geometries=None, raster_alphas=None,
                                            raster_colormaps=None, raster_labels=None,
                                            geometry_colors=None, geometry_labels=None,
                                            figsize=(12, 12), alpha=0.5, title=None):
    """
    Load and plot multiple georeferenced GeoTIFF files on a single map with correct spatial positioning.
    
    Parameters:
        tiff_files (list): List of paths to GeoTIFF files
        geometries (str or list, optional): WKT string(s) for areas of interest
        raster_alphas (list, optional): Alpha values for each raster
        raster_colormaps (list, optional): Colormap names for each raster
        raster_labels (list, optional): Labels for each raster
        geometry_colors (str or list): Color(s) for geometry overlays
        geometry_labels (str or list): Label(s) for geometries
        figsize (tuple): Figure size in inches
        alpha (float): Transparency for geometry overlays
        title (str): Title for the plot
        
    Returns:
        tuple: (fig, ax) - The figure and axes objects
    """
    # Convert inputs to lists if they're not already
    if not isinstance(tiff_files, list):
        tiff_files = [tiff_files]
    
    if geometries is not None and not isinstance(geometries, list):
        geometries = [geometries]
    
    if raster_labels is not None and not isinstance(raster_labels, list):
        raster_labels = [raster_labels]
    
    if geometry_colors is not None and not isinstance(geometry_colors, list):
        geometry_colors = [geometry_colors]
    
    if geometry_labels is not None and not isinstance(geometry_labels, list):
        geometry_labels = [geometry_labels]
        
    if raster_colormaps is not None and not isinstance(raster_colormaps, list):
        raster_colormaps = [raster_colormaps]
        
    if raster_alphas is not None and not isinstance(raster_alphas, list):
        raster_alphas = [raster_alphas]
    
    # Set default alpha values
    if raster_alphas is None:
        raster_alphas = [1.0] + [max(0.2, 1.0 - (i * 0.15)) for i in range(1, len(tiff_files))]
        
    # Set default colormaps if not provided
    if raster_colormaps is None:
        default_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'terrain']
        raster_colormaps = [default_cmaps[i % len(default_cmaps)] for i in range(len(tiff_files))]
        
    # Set default colors for geometries if not provided
    if geometries is not None and geometry_colors is None:
        default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'teal']
        geometry_colors = [default_colors[i % len(default_colors)] for i in range(len(geometries))]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Store handles for the legend
    handles = []
    
    # Open and plot each raster
    for i, tiff_file in enumerate(tiff_files):
        with rasterio.open(tiff_file) as src:
            # Read the data and mask out no data values
            raster = src.read(1)
            
            # Determine the no data value
            nodata = src.nodata
            if nodata is not None:
                # Create a masked array to handle no data values
                raster = np.ma.masked_where(raster == nodata, raster)
            else:
                # If no explicit nodata value, mask very negative values
                raster = np.ma.masked_where(raster < -1000, raster)
            
            # Get the raster bounds
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            # Get the colormap
            cmap_name = raster_colormaps[i % len(raster_colormaps)]
            cmap = plt.cm.get_cmap(cmap_name)
            
            # Plot the raster
            im = ax.imshow(raster, 
                         extent=extent,  # This is key for georeferencing
                         cmap=cmap, 
                         alpha=raster_alphas[i],
                         origin='upper')  # Raster origin is usually upper left
            
            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            if raster_labels is not None and i < len(raster_labels):
                cbar.set_label(raster_labels[i])
            
            # Add to legend
            if raster_labels is not None and i < len(raster_labels):
                from matplotlib.patches import Patch
                patch = Patch(color=cmap(0.7), alpha=raster_alphas[i], label=raster_labels[i])
                handles.append(patch)
    
    # Plot geometries if provided
    if geometries is not None:
        for i, wkt_string in enumerate(geometries):
            # Parse the WKT string to a shapely geometry
            geom = wkt.loads(wkt_string)
            
            # Set color for this geometry
            color = geometry_colors[i % len(geometry_colors)]
            
            # Plot the geometry based on its type
            if isinstance(geom, Polygon):
                # Extract exterior coordinates
                x, y = geom.exterior.xy
                poly = MplPolygon(np.column_stack([x, y]), 
                                 facecolor=color, 
                                 alpha=alpha,
                                 edgecolor='black', 
                                 label=geometry_labels[i] if geometry_labels else None)
                ax.add_patch(poly)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(poly)
            
            elif isinstance(geom, LineString):
                x, y = geom.xy
                line, = ax.plot(x, y, color=color, linewidth=2, 
                              label=geometry_labels[i] if geometry_labels else None)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(line)
            
            elif isinstance(geom, Point):
                x, y = geom.x, geom.y
                point, = ax.plot(x, y, 'o', color=color, markersize=8,
                               label=geometry_labels[i] if geometry_labels else None)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(point)
    
    # Add legend if we have handles
    if handles:
        ax.legend(handles=handles, loc='upper right')
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add north arrow and scale bar (optional)
    # This is a simple north arrow - you might want to use a more sophisticated approach
    from matplotlib.patches import Arrow
    arrow_pos = (0.95, 0.05)
    arrow_length = 0.05
    ax.add_patch(Arrow(arrow_pos[0], arrow_pos[1], 0, arrow_length, 
                      width=0.03, transform=ax.transAxes, 
                      facecolor='black', edgecolor='black'))
    ax.text(arrow_pos[0], arrow_pos[1] + arrow_length + 0.01, 'N', 
           transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    return fig, ax







def visualize_pointclouds(point_clouds, elev=45, azim=45):
    """
    Visualize a list of point clouds in a 3D scatter plot with adjustable viewing angles.

    :param point_clouds: list of NumPy structured arrays, each containing point data (X, Y, Z, etc.).
    :param elev: elevation angle in degrees (default=30).
    :param azim: azimuth angle in degrees (default=45).
    """
    if not point_clouds:
        print("No point clouds to visualize.")
        return

    # Get the common fields across all point clouds
    common_fields = set(point_clouds[0].dtype.names)
    for pc in point_clouds:
        common_fields.intersection_update(pc.dtype.names)

    # Normalize all arrays to have only the common fields
    normalized_point_clouds = []
    for pc in point_clouds:
        # Create a new array with only the common fields
        normalized_pc = np.zeros(pc.shape, dtype=[(field, pc.dtype[field]) for field in common_fields])
        for field in common_fields:
            normalized_pc[field] = pc[field]
        normalized_point_clouds.append(normalized_pc)

    # Concatenate all normalized point clouds into one array
    all_points = np.concatenate(normalized_point_clouds)

    # Extract X, Y, Z
    x = all_points['X']
    y = all_points['Y']
    z = all_points['Z']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    sc = ax.scatter(x, y, z, c=z, s=0.1, cmap='viridis')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Optionally fix aspect ratio so the axes scales are consistent:
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    plt.title("Point Cloud Visualization")
    plt.show()



    

def stack_rasters(*rasters):
    """Stack multiple 2D rasters into a 3D array for plotting with earthpy."""
    stacked_image = np.stack(rasters, axis=2)
    return np.transpose(stacked_image, (2, 0, 1))


def plot_rasters(rasters, figsize=(12, 12), titles=None):
    """Plot multiple rasters side by side."""
    ncols = rasters.shape[0]
    ep.plot_bands(
        rasters,
        cols=ncols,
        figsize=figsize,
        cmap="viridis",
        title=titles
    )


def stack_and_plot_rasters(*rasters, titles=None, figsize=(12, 12)):
    """Stack and plot multiple 2D rasters side by side."""
    stacked_rasters = stack_rasters(*rasters)
    plot_rasters(stacked_rasters, titles=titles, figsize=figsize)


def compare_two_rasters(rstr1, rstr2, rstr1_name, rstr2_name, metric_str, 
                       remove_outliers=True, figsize=(12, 12)):
    """Compare two rasters by plotting them with their difference.
    
    Displays three plots side by side:
    1. First raster
    2. Second raster
    3. Difference between them
    
    When remove_outliers=True, extreme values in the difference plot are capped
    at the 95th percentile to make patterns more visible.
    """
    title1 = f"{rstr1_name} {metric_str}"
    title2 = f"{rstr2_name} {metric_str}"
    title3 = f"{rstr1_name}-{rstr2_name} {metric_str}"

    error_rstr = (rstr1 - rstr2)
    if remove_outliers:
        threshold = np.percentile(np.abs(error_rstr), 95)  
        error_rstr[(error_rstr < -threshold)] = -threshold
        error_rstr[(error_rstr > threshold)] = threshold

    stack_and_plot_rasters(rstr1, rstr2, error_rstr, titles=[title1, title2, title3], figsize=figsize)


def create_pctile_plot_titles(percentile):
    """Create titles for percentile plots."""
    return [f"{x:.0f}%" for x in percentile]


def create_density_plot_titles(density_lvl_minmax_hag, density_layers):
    """Create titles for density proportion plots."""
    plot_titles = []
    for i in range(density_layers):
        min_val = density_lvl_minmax_hag[0] + (i * (density_lvl_minmax_hag[1] - density_lvl_minmax_hag[0]) / density_layers)
        max_val = density_lvl_minmax_hag[0] + ((i+1) * (density_lvl_minmax_hag[1] - density_lvl_minmax_hag[0]) / density_layers)
        title = f" {min_val:.0f}-{max_val:.0f}m"
        plot_titles.append(title)
    return plot_titles


def compare_two_n_dim_rasters(rstr1, rstr2, rstr1_name, rstr2_name, plot_title_function, *args, **kwargs):
    """Compare two 3D rasters by plotting each dimension with differences."""
    t_rstr1 = np.transpose(rstr1, (2, 0, 1))
    t_rstr2 = np.transpose(rstr2, (2, 0, 1))
    error_rstr = (t_rstr1 - t_rstr2)

    threshold = np.percentile(np.abs(error_rstr), 95)  
    error_rstr[(error_rstr < -threshold)] = -threshold
    error_rstr[(error_rstr > threshold)] = threshold

    ncols = t_rstr1.shape[0]
    titles1 = [rstr1_name + title for title in plot_title_function(*args, **kwargs)]
    titles2 = [rstr2_name + title for title in plot_title_function(*args, **kwargs)]
    titles3 = [f"{rstr1_name}-{rstr2_name}" + title for title in plot_title_function(*args, **kwargs)]
    figsize = (20, 10)
    
    ep.plot_bands(t_rstr1, figsize=figsize, cols=ncols, cmap="viridis", title=titles1)
    ep.plot_bands(t_rstr2, figsize=figsize, cols=ncols, cmap="viridis", title=titles2)
    ep.plot_bands(error_rstr, figsize=figsize, cols=ncols, cmap="PiYG", title=titles3)


# Vegetation Structure Metric Functions
# The following functions implement the vegetation structure metrics proposed by Moudry et al. (2023)
# These metrics help characterize vegetation structure in terms of height, cover, density, 
# and vertical complexity

def point_count(point_array):
    """Count the number of points in an array.
    
    Used as an auxiliary function to count points in a cell.
    """
    return point_array.size


def canopy_density(hag_array, canopy_min_hag):
    """Calculate proportion of vegetation points in the canopy layer.
    
    Measures the amount of vegetation in the tree/canopy layer.
    A value of 0.65 means that 65% of all vegetation returns came from trees.
    
    Calculation: Number of returns at the top vegetation layer divided by 
    the total number of vegetation returns.
    """
    n_canopy_points = hag_array[(hag_array > canopy_min_hag)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_canopy_points / n_veg_points


def mid_story_density(hag_array, understory_max_hag, canopy_min_hag):
    """Calculate proportion of vegetation points in the mid-story layer.
    
    Measures the amount of vegetation in the shrub/mid-story layer.
    A value of 0.25 means that 25% of all vegetation returns came from shrub vegetation.
    
    Calculation: Number of returns at the middle vegetation layer divided by
    the total number of vegetation returns.
    """
    n_mid_story_points = hag_array[(hag_array >= understory_max_hag) & (hag_array < canopy_min_hag)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_mid_story_points / n_veg_points


def under_story_density(hag_array, understory_max_hag):
    """Calculate proportion of vegetation points in the understory layer.
    
    Measures the amount of vegetation in the herbaceous/understory layer.
    A value of 0.10 means that 10% of all vegetation returns came from herbaceous vegetation.
    
    Calculation: Number of returns at the lowest vegetation layer divided by
    the total number of vegetation returns.
    """
    n_under_story_points = hag_array[(hag_array <= understory_max_hag) & (hag_array > 0.1)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_under_story_points / n_veg_points


def canopy_cover(hag_array, canopy_min_hag):
    """Calculate proportion of all points in the canopy layer.
    
    Measures the extent/percentage of the ground covered by vegetation.
    A value of 0.85 means that 85% of returns were reflected above the height threshold.
    The higher the value, the denser the canopy (closed stands).
    Low values reflect open or scattered stands.
    
    Calculation: Number of returns above a given height cutoff divided by
    the total number of returns.
    """
    n_canopy_points = hag_array[(hag_array > canopy_min_hag)].size
    n_points = hag_array.size
    return 0 if n_points == 0 else n_canopy_points / n_points


def density_proportions(hag_array, hag_rng, num_layers):
    """Calculate proportion of points in each vertical layer.
    
    Measures the vertical distribution of points (vegetation architecture).
    Creates fixed height bins between the minimum and maximum height and
    calculates the proportion of returns in each bin.
    
    Calculation: For each height bin, divide the number of returns in that bin
    by the total number of returns.
    
    Returns:
        np.array: Array of length num_layers containing proportions for each layer
    """
    bins = np.linspace(hag_rng[0], hag_rng[1], num_layers+1)
    layer_indices = np.digitize(hag_array, bins) - 1
    counts = np.bincount(layer_indices, minlength=num_layers)[0:num_layers]
    total_counts = np.sum(counts)
    return np.zeros(num_layers) if total_counts == 0 else counts / total_counts


def foliage_height_diversity(hag_array, hag_rng, num_layers):
    """Calculate Foliage Height Diversity using Shannon-Wiener index.
    
    A measure of canopy layering complexity (MacArthur & MacArthur, 1961).
    The maximum possible value increases with the number of layers.
    The maximum value occurs when all layers have the same number of returns
    (i.e., the Shannon-Wiener index increases with a more even distribution 
    of points over the layers).
    
    Calculation: FHD = -∑(p_i * ln(p_i))
    where p_i is the proportion of returns in each vertical layer i,
    and n is the total number of layers.
    """
    proportions = density_proportions(hag_array, hag_rng, num_layers)
    mask = proportions > 0
    if np.sum(mask) == 0:
        return 0
    return -np.sum(proportions[mask] * np.log(proportions[mask]))


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: Process and classify a LAS file
    
    input_las = 'raw_point_cloud.las'
    classified_las = 'classified_with_hag.las'
    
    # Define a crop polygon (in the coordinate system of the LAS file)
    crop_polygon = "POLYGON ((769896 3842418, 769896 3842929, 770189 3842929, 770189 3842418, 769896 3842418))"
    
    # Create and execute the PDAL pipeline
    las_pipeline = process_and_classify_las(
        input_las=input_las,
        output_las=classified_las,
        crop_polygon=crop_polygon,
        min_hag=0,
        max_hag=25,
        filter_noise=True
    )
    las_pipeline.execute()
    print(f"Processed and classified {input_las} to {classified_las}")
    
    # Example 2: Basic raster statistics
    
    # Set the resolution for all rasters
    resolution_m = 1  
    
    # Calculate various basic statistical measures
    max_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.max)
    mean_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.mean)
    min_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.min)
    std_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.std)
    point_density = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=point_count)
    
    # Example 3: Height percentile metrics
    
    # Calculate height percentiles (10%, 25%, 50%, 75%, 90%)
    percentile_values = [10, 25, 50, 75, 90]
    height_percentiles = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=np.percentile, 
        q=percentile_values
    )
    
    # Example 4: Vegetation structure metrics
    
    # Parameters for vegetation structure calculation
    canopy_min_hag = 3      # Minimum height for canopy points (meters)
    understory_max_hag = 1  # Maximum height for understory points (meters)
    
    # Calculate density by vegetation layer
    canopy_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=canopy_density, 
        canopy_min_hag=canopy_min_hag
    )
    
    midstory_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=mid_story_density, 
        understory_max_hag=understory_max_hag, 
        canopy_min_hag=canopy_min_hag
    )
    
    understory_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=under_story_density, 
        understory_max_hag=understory_max_hag
    )
    
    # Calculate canopy cover
    canopy_cov = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=canopy_cover, 
        canopy_min_hag=2  # Different threshold than canopy density
    )
    
    # Example 5: More complex metrics
    
    # Parameters for density proportions and FHD
    density_layers = 10
    height_range = [0, 18]  # Min and max height for vertical layers (meters)
    
    # Calculate density proportions (vegetation architecture)
    density_props = agg_las_to_array(
        classified_las, 
        resolution=resolution_m,
        aggregate_func=density_proportions, 
        hag_rng=height_range, 
        num_layers=density_layers
    )
    
    # Calculate Foliage Height Diversity (FHD)
    fhd = agg_las_to_array(
        classified_las, 
        resolution=resolution_m,
        aggregate_func=foliage_height_diversity, 
        hag_rng=height_range, 
        num_layers=density_layers
    )
    
    # Example 6: Visualization
    
    # Plot single rasters
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(max_height, cmap='viridis')
    plt.colorbar(label='Height (m)')
    plt.title('Maximum Vegetation Height')
    
    # Compare two rasters
    compare_two_rasters(
        canopy_dens, 
        midstory_dens, 
        'Canopy', 
        'Midstory', 
        'Density',
        figsize=(15, 5)
    )
    
    # Example 7: Get point cloud footprint
    
    # Get the footprint of a point cloud as WKT
    footprint_wkt = get_pointcloud_footprint(
        classified_las,
        simplify_tolerance=1.0,  # Simplify the polygon (units in CRS units, e.g., meters)
        buffer_distance=5.0      # Add 5 meter buffer around the point cloud
    )
    
    # Print the WKT string
    print(f"Point cloud footprint WKT: {footprint_wkt[:100]}...")
    
    # Example 8: Plot rasters with geometry overlays
    
    # Plot a single raster with a geometry overlay
    max_height_copy = max_height.copy()  # Create a copy to avoid modifying original
    
    # Get the footprint of the point cloud
    footprint = get_pointcloud_footprint(
        classified_las,
        simplify_tolerance=2.0  # More aggressive simplification for cleaner visualization
    )
    
    # Create a smaller region inside the footprint for demonstration
    interior_geom = wkt.loads(footprint)
    interior_footprint = interior_geom.buffer(-10).wkt  # 10m inside the original footprint
    
    # Plot max height raster with footprints overlaid
    plot_rasters_with_geometries(
        rasters=max_height_copy,
        geometries=[footprint, interior_footprint],
        raster_titles="Maximum Vegetation Height",
        geometry_colors=['red', 'blue'],
        geometry_labels=['Original Footprint', 'Interior Region'],
        figsize=(10, 8),
        alpha=0.3
    )
    
    # Plot multiple rasters with the same geometry overlay
    plot_rasters_with_geometries(
        rasters=[max_height, mean_height, std_height],
        geometries=footprint,
        raster_titles=["Max Height", "Mean Height", "Std Dev"],
        geometry_colors='green',
        geometry_labels='Point Cloud Extent',
        figsize=(15, 5),
        alpha=0.2
    )