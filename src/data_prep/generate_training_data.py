from shapely.geometry import box
from shapely import wkt

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

# Example usage:
if __name__ == '__main__':
    # Define an arbitrary polygon as a WKT string (a simple concave polygon example)
    wkt_poly = "POLYGON ((0 0, 10 0, 10 10, 6 5, 0 10, 0 0))"
    tiles = create_tiles_from_wkt(wkt_poly, tile_width=3, tile_height=3, overlap_ratio=0.2)
    for t in tiles:
        print(t.bounds)



def create_tiles(bbox, tile_width, tile_height, overlap_ratio):
    """
    Creates a list of tile bounding boxes within a given bounding box.

    Parameters:
        bbox (tuple): A tuple (minx, miny, maxx, maxy) defining the full bounding box.
        tile_width (float): The width of each tile.
        tile_height (float): The height of each tile.
        overlap_ratio (float): The fraction of overlap between adjacent tiles (0 to 1).

    Returns:
        list: A list of tuples, each representing a tile's bounding box as 
              (tile_minx, tile_miny, tile_maxx, tile_maxy).
    """
    # Unpack the full bounding box
    minx, miny, maxx, maxy = bbox

    # Validate the overlap ratio
    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be between 0 and 1 (non-inclusive of 1)")
    
    # Calculate step sizes. With an overlap_ratio of 0.5, the step is half the tile size.
    step_x = tile_width * (1 - overlap_ratio)
    step_y = tile_height * (1 - overlap_ratio)

    tiles = []
    x = minx

    # Slide the tile window along the x-axis.
    while x + tile_width <= maxx:
        y = miny
        # Slide along the y-axis.
        while y + tile_height <= maxy:
            tile = (x, y, x + tile_width, y + tile_height)
            tiles.append(tile)
            y += step_y
        x += step_x

    return tiles
