import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

# Try importing PyKrige for kriging; if not installed, set to None.
try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None
    print("PyKrige not installed. dtm_kriging() will not work.")

# =============================================================================
# 1. Raster-Based Minimum Z (Simple Gridding)
# =============================================================================
def dtm_raster_min_z(points, grid_res, bounding_box):
    """
    Create a DTM by gridding the point cloud and taking the minimum z value in each cell.
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Grid resolution.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the grid extent.
    
    Returns:
        grid (ndarray): 2D array of minimum z values per grid cell.
        x_min (float): Minimum x coordinate (from bounding_box).
        y_min (float): Minimum y coordinate (from bounding_box).
        grid_res (float): Resolution used.
    """
    x_min, y_min, x_max, y_max = bounding_box
    nx = int(np.ceil((x_max - x_min) / grid_res))
    ny = int(np.ceil((y_max - y_min) / grid_res))
    
    # Initialize grid with NaNs.
    grid = np.full((ny, nx), np.nan)
    
    # Compute cell indices for each point.
    # Only consider points within the bounding box.
    for pt in points:
        x, y, z = pt
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            continue
        i = int((x - x_min) / grid_res)
        j = int((y - y_min) / grid_res)
        # If the cell is empty, assign the z value; otherwise, keep the minimum.
        if np.isnan(grid[j, i]):
            grid[j, i] = z
        else:
            grid[j, i] = min(grid[j, i], z)
    
    return grid, x_min, y_min, grid_res

# =============================================================================
# 2. Progressive Morphological Filter (PMF)
# =============================================================================
def dtm_progressive_morphological(points, grid_res, bounding_box, init_window=3, max_window=21, slope=0.15):
    """
    Create a DTM using a simplified Progressive Morphological Filter.
    
    The function first creates a raster using minimum z per cell within the 
    provided bounding box and then iteratively “opens” the surface using grey-erosion.
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Grid resolution.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the grid extent.
        init_window (int): Initial window size (must be odd).
        max_window (int): Maximum window size.
        slope (float): Slope factor (determines threshold).
    
    Returns:
        filtered (ndarray): 2D array of filtered ground elevations.
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        grid_res (float): Resolution used.
    """
    # Start with a raster of minimum z values.
    dtm, x_min, y_min, res = dtm_raster_min_z(points, grid_res, bounding_box)
    filtered = dtm.copy()
    
    # Progressive filtering: increase the window size gradually.
    window = init_window
    while window <= max_window:
        # Use grey-erosion (minimum filter) to simulate morphological opening.
        min_filter = ndimage.grey_erosion(filtered, size=(window, window))
        # Calculate threshold (adjustable formulation)
        threshold = slope * window * res
        diff = filtered - min_filter
        mask = diff > threshold
        filtered[mask] = min_filter[mask] + threshold
        window += 2  # Increase window size in odd steps.
    
    return filtered, x_min, y_min, res

# =============================================================================
# 3. Triangulated Irregular Network (TIN) + Ground Classification
# =============================================================================
def dtm_tin(points, grid_res, bounding_box):
    """
    Create a DTM by first extracting ground points via a raster min Z and then
    interpolating with a TIN (using linear interpolation over Delaunay triangles).
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Desired grid resolution for the output DTM.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the grid extent.
    
    Returns:
        dtm (ndarray): 2D array of interpolated ground elevations.
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        res (float): Resolution used.
    """
    grid, x_min, y_min, res = dtm_raster_min_z(points, grid_res, bounding_box)
    ny, nx = grid.shape
    # Extract “ground” points from valid grid cells.
    ground_points = []
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(grid[j, i]):
                # Use the center of the grid cell.
                x_coord = x_min + i * res + res / 2.
                y_coord = y_min + j * res + res / 2.
                ground_points.append([x_coord, y_coord, grid[j, i]])
    ground_points = np.array(ground_points)
    
    # Create a regular grid for interpolation.
    x_vals = np.linspace(x_min, x_min + nx * res, nx)
    y_vals = np.linspace(y_min, y_min + ny * res, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    # Linear interpolation (uses underlying Delaunay triangulation).
    dtm = griddata(ground_points[:, :2], ground_points[:, 2], (xx, yy), method='linear')
    
    return dtm, x_min, y_min, res

# =============================================================================
# 4. Inverse Distance Weighting (IDW)
# =============================================================================
def dtm_idw(points, grid_res, bounding_box, power=2, search_radius=None):
    """
    Create a DTM using Inverse Distance Weighting (IDW) interpolation.
    
    This function first extracts ground points from a raster based on minimum
    z values within the given bounding box, then interpolates the surface at 
    each grid node using weighted averages.
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Desired grid resolution for the output DTM.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the grid extent.
        power (float): Power parameter for IDW.
        search_radius (float or None): If given, only points within this distance
                                       are used for the interpolation.
    
    Returns:
        dtm (ndarray): 2D array of interpolated ground elevations.
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        res (float): Resolution used.
    """
    grid, x_min, y_min, res = dtm_raster_min_z(points, grid_res, bounding_box)
    ny, nx = grid.shape
    ground_points = []
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(grid[j, i]):
                x_coord = x_min + i * res + res / 2.
                y_coord = y_min + j * res + res / 2.
                ground_points.append([x_coord, y_coord, grid[j, i]])
    ground_points = np.array(ground_points)
    
    # Prepare the output grid.
    x_vals = np.linspace(x_min, x_min + nx * res, nx)
    y_vals = np.linspace(y_min, y_min + ny * res, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    dtm = np.full(xx.shape, np.nan)
    
    # For each grid cell, perform the IDW interpolation.
    for j in range(xx.shape[0]):
        for i in range(xx.shape[1]):
            query = np.array([xx[j, i], yy[j, i]])
            dists = np.sqrt((ground_points[:, 0] - query[0])**2 +
                            (ground_points[:, 1] - query[1])**2)
            if search_radius is not None:
                mask = dists < search_radius
                if np.sum(mask) == 0:
                    continue
                dists = dists[mask]
                values = ground_points[mask, 2]
            else:
                values = ground_points[:, 2]
            if np.any(dists == 0):
                dtm[j, i] = values[dists == 0][0]
            else:
                weights = 1.0 / (dists ** power)
                dtm[j, i] = np.sum(weights * values) / np.sum(weights)
    
    return dtm, x_min, y_min, res

# =============================================================================
# 5. Kriging (Using PyKrige)
# =============================================================================
def dtm_kriging(points, grid_res, bounding_box, variogram_model='linear'):
    """
    Create a DTM using Ordinary Kriging interpolation.
    
    This function extracts ground points from a raster based on minimum z values 
    within the provided bounding box and uses PyKrige for geostatistical interpolation.
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Desired grid resolution for the output DTM.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the grid extent.
        variogram_model (str): Variogram model to use in kriging.
    
    Returns:
        dtm (ndarray): 2D array of interpolated ground elevations.
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        res (float): Resolution used.
    """
    if OrdinaryKriging is None:
        raise ImportError("PyKrige is required for dtm_kriging. Please install it.")
    
    grid, x_min, y_min, res = dtm_raster_min_z(points, grid_res, bounding_box)
    ny, nx = grid.shape
    ground_points = []
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(grid[j, i]):
                x_coord = x_min + i * res + res / 2.
                y_coord = y_min + j * res + res / 2.
                ground_points.append([x_coord, y_coord, grid[j, i]])
    ground_points = np.array(ground_points)
    
    # Create grid coordinates for kriging.
    x_vals = np.linspace(x_min, x_min + nx * res, nx)
    y_vals = np.linspace(y_min, y_min + ny * res, ny)
    
    OK = OrdinaryKriging(
        ground_points[:, 0],
        ground_points[:, 1],
        ground_points[:, 2],
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )
    z, ss = OK.execute('grid', x_vals, y_vals)
    
    return z, x_min, y_min, res

# =============================================================================
# 6. Cloth Simulation Filtering (CSF) + Surface Interpolation
# =============================================================================
def dtm_csf(points, grid_res, bounding_box, cloth_resolution=1.0, iterations=200, class_threshold=0.5):
    """
    Create a DTM using a simplified Cloth Simulation Filtering (CSF) approach.
    
    The function simulates a cloth (a grid of nodes) that is iteratively lowered 
    onto the point cloud (only considering points within the bounding box). Points 
    that lie close to the final cloth surface are classified as ground. The cloth 
    is then interpolated to form the DTM.
    
    Parameters:
        points (ndarray): (N,3) array of [x, y, z] coordinates.
        grid_res (float): Desired output grid resolution.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) defining the simulation extent.
        cloth_resolution (float): Spacing of the cloth simulation grid.
        iterations (int): Number of simulation iterations.
        class_threshold (float): Maximum vertical distance from the cloth to classify a point as ground.
    
    Returns:
        dtm (ndarray): 2D array of interpolated ground elevations.
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        grid_res (float): Output grid resolution.
        ground_points (ndarray): (M,3) array of points classified as ground.
    """
    x_min, y_min, x_max, y_max = bounding_box
    nx = int(np.ceil((x_max - x_min) / cloth_resolution))
    ny = int(np.ceil((y_max - y_min) / cloth_resolution))
    
    # Initialize the cloth at a height slightly above the maximum z in the area.
    # Only consider points within the bounding box.
    in_bbox = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
              (points[:, 1] >= y_min) & (points[:, 1] < y_max)
    pts_in_bbox = points[in_bbox]
    initial_height = np.max(pts_in_bbox[:, 2]) + 5
    cloth = np.full((ny, nx), initial_height)
    
    # Define cloth node coordinates.
    xs = x_min + (np.arange(nx) + 0.5) * cloth_resolution
    ys = y_min + (np.arange(ny) + 0.5) * cloth_resolution
    xx, yy = np.meshgrid(xs, ys)
    
    # Simulate the cloth descending onto the point cloud.
    for _ in range(iterations):
        new_cloth = cloth.copy()
        for j in range(ny):
            for i in range(nx):
                x_center, y_center = xx[j, i], yy[j, i]
                # Consider nearby points (within 1.5 * cloth_resolution).
                dists = np.sqrt((pts_in_bbox[:, 0] - x_center)**2 +
                                (pts_in_bbox[:, 1] - y_center)**2)
                mask = dists < (cloth_resolution * 1.5)
                if np.any(mask):
                    new_cloth[j, i] = min(cloth[j, i], np.min(pts_in_bbox[mask, 2]) + 0.1)
        # Simple smoothing.
        cloth = ndimage.uniform_filter(new_cloth, size=3)
    
    # Classify ground points as those within a threshold of the cloth surface.
    ground_mask = np.zeros(points.shape[0], dtype=bool)
    for idx, pt in enumerate(points):
        x, y, z = pt
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            continue
        i = int((x - x_min) / cloth_resolution)
        j = int((y - y_min) / cloth_resolution)
        if (z - cloth[j, i]) < class_threshold:
            ground_mask[idx] = True
    ground_points = points[ground_mask]
    
    # Interpolate the cloth to the desired output grid resolution.
    x_vals = np.linspace(x_min, x_max, int(np.ceil((x_max - x_min) / grid_res)))
    y_vals = np.linspace(y_min, y_max, int(np.ceil((y_max - y_min) / grid_res)))
    xx_fine, yy_fine = np.meshgrid(x_vals, y_vals)
    dtm = griddata((xx.ravel(), yy.ravel()), cloth.ravel(), (xx_fine, yy_fine), method='linear')
    
    return dtm, x_min, y_min, grid_res#, ground_points

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # Create synthetic point cloud data for demonstration.
    np.random.seed(0)
    n_points = 10000
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    z = 0.05 * x + 0.03 * y + np.random.normal(0, 0.5, n_points)
    points = np.vstack((x, y, z)).T

    # Define a bounding box and grid resolution.
    bounding_box = (0, 0, 100, 100)  # (x_min, y_min, x_max, y_max)
    grid_resolution = 1.0

    # Method 1: Raster Min Z
    grid_minz, x0, y0, res = dtm_raster_min_z(points, grid_resolution, bounding_box)
    print("Raster Min Z complete.")

    # Method 2: Progressive Morphological Filter
    dtm_pmf, x0, y0, res = dtm_progressive_morphological(points, grid_resolution, bounding_box)
    print("Progressive Morphological Filter complete.")

    # Method 3: TIN interpolation
    dtm_tin_out, x0, y0, res = dtm_tin(points, grid_resolution, bounding_box)
    print("TIN interpolation complete.")

    # Method 4: IDW interpolation
    dtm_idw_out, x0, y0, res = dtm_idw(points, grid_resolution, bounding_box)
    print("IDW interpolation complete.")

    # Method 5: Kriging interpolation (if PyKrige is available)
    if OrdinaryKriging is not None:
        dtm_krig, x0, y0, res = dtm_kriging(points, grid_resolution, bounding_box)
        print("Kriging interpolation complete.")
    else:
        print("Skipping kriging due to missing PyKrige.")

    # Method 6: Cloth Simulation Filtering (CSF)
    dtm_csf_out, x0, y0, res, ground_pts = dtm_csf(points, grid_resolution, bounding_box)
    print("Cloth Simulation Filtering complete.")
