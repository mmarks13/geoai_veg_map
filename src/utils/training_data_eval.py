import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from rasterio.transform import from_origin
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from matplotlib.ticker import FuncFormatter
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import torch
from matplotlib.patches import Rectangle
from rasterio.transform import from_origin
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from matplotlib.ticker import FuncFormatter

# --- New H5 Loading Functions ---
def load_tile_from_h5(h5_file_path, tile_id=None, tile_index=None):
    """
    Load a single tile from an HDF5 file, with improved handling of bounding boxes and metadata.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    tile_id : str or int, optional
        Specific tile ID to load
    tile_index : int, optional
        Index of the tile to load (0-based)
        
    Returns:
    --------
    dict
        Dictionary containing all the data for the selected tile
    """
    with h5py.File(h5_file_path, 'r') as f:
        # Find available tiles
        tiles = [k for k in f.keys() if k.startswith('tile_')]
        if not tiles:
            raise ValueError(f"No tiles found in {h5_file_path}")
        
        # Select the appropriate tile
        if tile_index is not None and isinstance(tile_index, int):
            if 0 <= tile_index < len(tiles):
                tile_key = tiles[tile_index]
            else:
                raise ValueError(f"Tile index {tile_index} out of range. File has {len(tiles)} tiles.")
        elif tile_id is not None:
            # Handle both numeric and string IDs
            tile_key = f"tile_{tile_id}" if not str(tile_id).startswith('tile_') else str(tile_id)
            if tile_key not in tiles:
                try:
                    idx = int(tile_id)
                    if 0 <= idx < len(tiles):
                        tile_key = tiles[idx]
                    else:
                        raise ValueError(f"Tile {tile_id} not found. Available tiles: {[t.split('_')[1] for t in tiles]}")
                except ValueError:
                    raise ValueError(f"Tile {tile_id} not found. Available tiles: {[t.split('_')[1] for t in tiles]}")
        else:
            # Default to first tile
            tile_key = tiles[0]
        
        print(f"Loading tile: {tile_key}")
        tile_group = f[tile_key]
        
        # Initialize result dictionary with default values
        result = {
            'has_imagery': False,
            'has_pointcloud': True,
            'has_uavsar': False,
            'has_naip': False
        }
        
        # Load basic attributes
        for attr in tile_group.attrs:
            result[attr] = tile_group.attrs[attr]
        
        # Get the original and expanded bounding boxes
        orig_bbox = result.get('bbox', None)
        img_bbox = tile_group.attrs.get('img_bbox', None)
        
        # Calculate expanded bbox if not stored (fallback)
        if img_bbox is None and orig_bbox is not None:
            centroid_x = (orig_bbox[0] + orig_bbox[2]) / 2
            centroid_y = (orig_bbox[1] + orig_bbox[3]) / 2
            width = orig_bbox[2] - orig_bbox[0]
            height = orig_bbox[3] - orig_bbox[1]
            new_width = width * 2
            new_height = height * 2
            img_bbox = [
                centroid_x - new_width / 2,
                centroid_y - new_height / 2,
                centroid_x + new_width / 2,
                centroid_y + new_height / 2
            ]
            result['img_bbox'] = img_bbox
        
        # Load point cloud data
        if 'point_clouds' in tile_group:
            pc_group = tile_group['point_clouds']
            result['uav_points'] = pc_group['uav_points'][...]
            result['uav_pnt_attr'] = pc_group['uav_pnt_attr'][...]
            result['dep_points'] = pc_group['dep_points'][...]
            result['dep_pnt_attr'] = pc_group['dep_pnt_attr'][...]
        
        # Load downsample masks
        if 'downsample_masks' in tile_group:
            masks_group = tile_group['downsample_masks']
            result['uav_downsample_masks'] = [masks_group[m][...] for m in sorted(masks_group.keys())]
            
            # Set first mask as default for visualization
            if result['uav_downsample_masks']:
                result['uav_downsample_mask'] = result['uav_downsample_masks'][0]
        
        # Load datasets if present at top level
        if 'max_points_list' in tile_group and isinstance(tile_group['max_points_list'], h5py.Dataset):
            result['max_points_list'] = tile_group['max_points_list'][...].tolist()
        
        if 'voxel_sizes_cm' in tile_group and isinstance(tile_group['voxel_sizes_cm'], h5py.Dataset):
            result['voxel_sizes_cm'] = tile_group['voxel_sizes_cm'][...].tolist()
        
        # Load metadata
        if 'metadata' in tile_group:
            meta_group = tile_group['metadata']
            
            # Recursive function to load nested metadata
            def load_nested_metadata(group):
                result = {}
                # Load attributes
                for attr in group.attrs:
                    result[attr] = group.attrs[attr]
                
                # Load nested groups
                for key in group:
                    if isinstance(group[key], h5py.Group):
                        result[key] = load_nested_metadata(group[key])
                    elif isinstance(group[key], h5py.Dataset):
                        result[key] = group[key][...]
                
                return result
            
            # Load UAV metadata
            if 'uav_meta' in meta_group:
                result['uav_meta'] = load_nested_metadata(meta_group['uav_meta'])
                
            # Load 3DEP metadata
            if 'dep_meta' in meta_group:
                result['dep_meta'] = load_nested_metadata(meta_group['dep_meta'])
        
        # Load UAVSAR imagery
        if 'uavsar' in tile_group:
            uavsar_group = tile_group['uavsar']
            result['has_uavsar'] = True
            result['has_imagery'] = True
            
            # Get resolution from the UAVSAR group
            result['uavsar_resolution'] = uavsar_group.attrs.get('resolution', 5.0)
            
            # Get UAVSAR specific bbox if available, otherwise use the general expanded bbox
            uavsar_bbox = uavsar_group.attrs.get('img_bbox', img_bbox)
            result['uavsar_img_bbox'] = uavsar_bbox
            
            # Load images
            if 'images' in uavsar_group:
                imgs_group = uavsar_group['images']
                result['uavsar_imgs'] = [imgs_group[img][...] for img in sorted(imgs_group.keys())]
                
                # Load metadata
                if 'metadata' in uavsar_group:
                    meta_group = uavsar_group['metadata']
                    result['uavsar_imgs_meta'] = []
                    
                    for i, img_key in enumerate(sorted(meta_group.keys())):
                        img_meta_group = meta_group[img_key]
                        img_meta = {}
                        
                        # Load attributes (including explicitly stored bbox and resolution)
                        for attr in img_meta_group.attrs:
                            img_meta[attr] = img_meta_group.attrs[attr]
                        
                        # Ensure bbox is available - check multiple sources
                        if 'bbox' not in img_meta:
                            if 'bbox' in img_meta_group.attrs:
                                img_meta['bbox'] = img_meta_group.attrs['bbox']
                            elif 'img_bbox' in img_meta_group.attrs:
                                img_meta['bbox'] = img_meta_group.attrs['img_bbox']
                            else:
                                img_meta['bbox'] = uavsar_bbox
                        
                        # Ensure resolution is available
                        if 'resolution' not in img_meta:
                            img_meta['resolution'] = result['uavsar_resolution']
                        
                        # Load datasets in the metadata group
                        for key in img_meta_group:
                            if isinstance(img_meta_group[key], h5py.Dataset):
                                img_meta[key] = img_meta_group[key][...]
                        
                        result['uavsar_imgs_meta'].append(img_meta)
                else:
                    # Create minimal metadata if none available
                    result['uavsar_imgs_meta'] = []
                    for i in range(len(result['uavsar_imgs'])):
                        result['uavsar_imgs_meta'].append({
                            'bbox': uavsar_bbox,
                            'resolution': result['uavsar_resolution'],
                            'index': i,
                            'date': '2023-01-01',  # Default date if not available
                            'datetime': '2023-01-01T00:00:00Z'  # Default datetime
                        })
            
            # Load stacked array if available
            if 'stacked_imgs' in uavsar_group:
                result['uavsar_imgs_array'] = uavsar_group['stacked_imgs'][...]
        
        # Load NAIP imagery with similar approach
        if 'naip' in tile_group:
            naip_group = tile_group['naip']
            result['has_naip'] = True
            result['has_imagery'] = True
            
            # Get resolution from the NAIP group
            result['naip_resolution'] = naip_group.attrs.get('resolution', 0.5)
            
            # Get NAIP specific bbox if available, otherwise use the general expanded bbox
            naip_bbox = naip_group.attrs.get('img_bbox', img_bbox)
            result['naip_img_bbox'] = naip_bbox
            
            # Load images
            if 'images' in naip_group:
                imgs_group = naip_group['images']
                result['naip_imgs'] = [imgs_group[img][...] for img in sorted(imgs_group.keys())]
                
                # Load metadata
                if 'metadata' in naip_group:
                    meta_group = naip_group['metadata']
                    result['naip_imgs_meta'] = []
                    
                    for i, img_key in enumerate(sorted(meta_group.keys())):
                        img_meta_group = meta_group[img_key]
                        img_meta = {}
                        
                        # Load attributes (including explicitly stored bbox and resolution)
                        for attr in img_meta_group.attrs:
                            img_meta[attr] = img_meta_group.attrs[attr]
                        
                        # Ensure bbox is available - check multiple sources
                        if 'bbox' not in img_meta:
                            if 'bbox' in img_meta_group.attrs:
                                img_meta['bbox'] = img_meta_group.attrs['bbox']
                            elif 'img_bbox' in img_meta_group.attrs:
                                img_meta['bbox'] = img_meta_group.attrs['img_bbox']
                            else:
                                img_meta['bbox'] = naip_bbox
                        
                        # Ensure resolution is available
                        if 'resolution' not in img_meta:
                            img_meta['resolution'] = result['naip_resolution']
                        
                        # Load datasets in the metadata group
                        for key in img_meta_group:
                            if isinstance(img_meta_group[key], h5py.Dataset):
                                img_meta[key] = img_meta_group[key][...]
                        
                        result['naip_imgs_meta'].append(img_meta)
                else:
                    # Create minimal metadata if none available
                    result['naip_imgs_meta'] = []
                    for i in range(len(result['naip_imgs'])):
                        result['naip_imgs_meta'].append({
                            'bbox': naip_bbox,
                            'resolution': result['naip_resolution'],
                            'index': i,
                            'date': '2023-01-01',  # Default date if not available
                            'datetime': '2023-01-01T00:00:00Z'  # Default datetime
                        })
            
            # Load stacked array if available
            if 'stacked_imgs' in naip_group:
                result['naip_imgs_array'] = naip_group['stacked_imgs'][...]
        
        return result




def list_tiles_in_h5(h5_file_path):
    """List all tiles in an HDF5 file."""
    with h5py.File(h5_file_path, 'r') as f:
        tiles = [k for k in f.keys() if k.startswith('tile_')]
        return [t.split('_')[1] if '_' in t else t for t in tiles]



# --- Helper Functions ---
# --- Dictionary Structure Inspection Functions ---

def print_dict_structure(d, prefix='', max_depth=5, current_depth=0, max_array_items=3, max_str_len=50):
    """
    Print the structure of a nested dictionary with hierarchical indentation.
    
    Parameters:
    -----------
    d : dict or object
        Dictionary or object to inspect
    prefix : str
        Prefix for indentation
    max_depth : int
        Maximum recursion depth
    current_depth : int
        Current recursion depth (for internal use)
    max_array_items : int
        Maximum number of array items to show
    max_str_len : int
        Maximum string length to show before truncating
    """
    if current_depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
        
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: <dict> ({len(value)} keys)")
                print_dict_structure(value, prefix + '  ', max_depth, current_depth + 1, max_array_items, max_str_len)
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}{key}: <{type(value).__name__}> (length: {len(value)})")
                if len(value) > 0 and current_depth < max_depth:
                    # Print first few elements
                    for i, item in enumerate(value[:max_array_items]):
                        if isinstance(item, dict):
                            print(f"{prefix}  [{i}]: <dict> ({len(item)} keys)")
                            print_dict_structure(item, prefix + '    ', max_depth, current_depth + 2, max_array_items, max_str_len)
                        else:
                            item_str = str(item)
                            if len(item_str) > max_str_len:
                                item_str = item_str[:max_str_len] + "..."
                            print(f"{prefix}  [{i}]: {type(item).__name__} - {item_str}")
                    if len(value) > max_array_items:
                        print(f"{prefix}  ... ({len(value) - max_array_items} more items)")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: <ndarray> shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, str):
                val_str = value if len(value) <= max_str_len else value[:max_str_len] + "..."
                print(f"{prefix}{key}: <str> '{val_str}'")
            else:
                print(f"{prefix}{key}: <{type(value).__name__}> {value}")
    else:
        if isinstance(d, np.ndarray):
            print(f"{prefix}<ndarray> shape={d.shape}, dtype={d.dtype}")
        elif isinstance(d, (list, tuple)):
            print(f"{prefix}<{type(d).__name__}> (length: {len(d)})")
            # Print first few elements 
            for i, item in enumerate(d[:min(len(d), max_array_items)]):
                item_str = str(item)
                if len(item_str) > max_str_len:
                    item_str = item_str[:max_str_len] + "..."
                print(f"{prefix}  [{i}]: {type(item).__name__} - {item_str}")
            if len(d) > max_array_items:
                print(f"{prefix}  ... ({len(d) - max_array_items} more items)")
        else:
            val_str = str(d)
            if len(val_str) > max_str_len:
                val_str = val_str[:max_str_len] + "..."
            print(f"{prefix}<{type(d).__name__}> {val_str}")



def inspect_h5_structure(h5_file_path, max_depth=5):
    """
    Inspect the structure of an HDF5 file directly.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    max_depth : int
        Maximum recursion depth
    """

    def _print_h5_group(name, obj, prefix='', current_depth=0):
        if current_depth > max_depth:
            print(f"{prefix}... (max depth reached)")
            return
            
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{name}: <Group> ({len(obj.keys())} items)")
            if current_depth < max_depth:
                for key in obj.keys():
                    _print_h5_group(key, obj[key], prefix + '  ', current_depth + 1)
                    
                # Print attributes if any
                if obj.attrs:
                    print(f"{prefix}  - Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"{prefix}    {key}: {value}")
                        
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}{name}: <Dataset> shape={obj.shape}, dtype={obj.dtype}")
            
            # Print attributes if any
            if obj.attrs:
                print(f"{prefix}  - Attributes:")
                for key, value in obj.attrs.items():
                    print(f"{prefix}    {key}: {value}")
                    
            # Print a sample of data for small datasets
            if len(obj.shape) == 1 and obj.shape[0] < 10:
                print(f"{prefix}  - Values: {obj[...]}")
    
    with h5py.File(h5_file_path, 'r') as f:
        print(f"Structure of {h5_file_path}:")
        for name, obj in f.items():
            _print_h5_group(name, obj)


# --- HDF5 Tile Metadata Inspection Functions ---

def get_all_tiles_metadata(h5_file_path):
    """
    Get a summary of metadata for all tiles in an HDF5 file.
    
    Returns:
    --------
    dict
        Dictionary with tile IDs as keys and metadata summaries as values
    """
    metadata = {}
    
    with h5py.File(h5_file_path, 'r') as f:
        tiles = [k for k in f.keys() if k.startswith('tile_')]
        
        for tile_key in tiles:
            tile_group = f[tile_key]
            tile_id = tile_key
            
            # Extract basic metadata
            meta = {
                'bbox': tile_group.attrs.get('bbox', None),
                'has_imagery': tile_group.attrs.get('has_imagery', False),
                'has_pointcloud': tile_group.attrs.get('has_pointcloud', True)
            }
            # Count number of points
            if 'point_clouds' in tile_group:
                pc_group = tile_group['point_clouds']
                meta['uav_points_count'] = pc_group['uav_points'].shape[0] if 'uav_points' in pc_group else 0
                meta['dep_points_count'] = pc_group['dep_points'].shape[0] if 'dep_points' in pc_group else 0
            
            # Check for UAVSAR imagery
            if 'uavsar' in tile_group and 'images' in tile_group['uavsar']:
                meta['uavsar_imgs_count'] = len(tile_group['uavsar']['images'])
                meta['uavsar_resolution'] = tile_group['uavsar'].attrs.get('resolution', None)
            
            # Check for NAIP imagery
            if 'naip' in tile_group and 'images' in tile_group['naip']:
                meta['naip_imgs_count'] = len(tile_group['naip']['images'])
                meta['naip_resolution'] = tile_group['naip'].attrs.get('resolution', None)
            
            metadata[tile_id] = meta
    
    return metadata

def print_tile_metadata_summary(h5_file_path):
    """Print a summary of metadata for all tiles in an HDF5 file."""
    metadata = get_all_tiles_metadata(h5_file_path)
    
    print(f"Found {len(metadata)} tiles in {h5_file_path}:")
    print("-" * 80)
    
    for tile_id, meta in metadata.items():
        print(f"Tile: {tile_id}")
        bbox_str = f"[{meta['bbox'][0]:.1f}, {meta['bbox'][1]:.1f}, {meta['bbox'][2]:.1f}, {meta['bbox'][3]:.1f}]" if meta.get('bbox') is not None else "N/A"
        print(f"  Bbox: {bbox_str}")
        print(f"  UAV Points: {meta.get('uav_points_count', 0):,}")
        print(f"  3DEP Points: {meta.get('dep_points_count', 0):,}")
        
        if meta.get('uavsar_imgs_count', 0) > 0:
            print(f"  UAVSAR: {meta.get('uavsar_imgs_count', 0)} images, resolution: {meta.get('uavsar_resolution', 'N/A')}")
        
        if meta.get('naip_imgs_count', 0) > 0:
            print(f"  NAIP: {meta.get('naip_imgs_count', 0)} images, resolution: {meta.get('naip_resolution', 'N/A')}")
        
        print("-" * 80)

def examine_tile_metadata(data_dict):
    """
    Examine and print the detailed structure of a tile's metadata.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing tile data and metadata
    """
    print("=" * 80)
    print("TILE METADATA STRUCTURE")
    print("=" * 80)
    
    # Extract all metadata sections
    metadata_sections = {
        'Basic Attributes': {k: v for k, v in data_dict.items() 
                            if not isinstance(v, (list, np.ndarray)) 
                            and k not in ['uav_meta', 'dep_meta', 'uavsar_imgs_meta', 'naip_imgs_meta']},
        'UAV Metadata': data_dict.get('uav_meta', {}),
        '3DEP Metadata': data_dict.get('dep_meta', {}),
    }
    
    # Print UAVSAR image metadata if available
    if 'uavsar_imgs_meta' in data_dict and data_dict['uavsar_imgs_meta']:
        for i, meta in enumerate(data_dict['uavsar_imgs_meta']):
            metadata_sections[f'UAVSAR Image {i} Metadata'] = meta
    
    # Print NAIP image metadata if available
    if 'naip_imgs_meta' in data_dict and data_dict['naip_imgs_meta']:
        for i, meta in enumerate(data_dict['naip_imgs_meta']):
            metadata_sections[f'NAIP Image {i} Metadata'] = meta
    
    # Print each section
    for section, meta in metadata_sections.items():
        print(f"\n{section}:")
        print("-" * 80)
        print_dict_structure(meta)
    
    print("\nData Structure Overview:")
    print("-" * 80)
    
    # List the keys and shapes of arrays
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: <ndarray> shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            print(f"{key}: <list of ndarrays> length={len(value)}")
            for i, arr in enumerate(value[:3]):  # Show first 3
                print(f"  [{i}]: shape={arr.shape}, dtype={arr.dtype}")
            if len(value) > 3:
                print(f"  ... ({len(value) - 3} more arrays)")
    
    print("=" * 80)
# --- Updated Helper Functions ---

def array_to_structured_array(arr):
    """
    Convert a PyTorch tensor or NumPy array to a structured NumPy array.
    Works with both data types.
    """
    # Convert tensor to numpy if needed
    if hasattr(arr, 'is_cuda'):
        data = arr.cpu().numpy() if arr.is_cuda else arr.numpy()
    else:
        data = arr
        
    # Create structured array
    structured = np.zeros(data.shape[0], dtype=[('X', data.dtype), ('Y', data.dtype), ('Z', data.dtype)])
    structured['X'] = data[:, 0]
    structured['Y'] = data[:, 1]
    structured['Z'] = data[:, 2]
    return structured

def print_tile_debug_info(data_dict, tile_index=0):
    """Print debug information about a tile."""
    # Handle both tensor and numpy formats
    bbox = data_dict.get('bbox')
    bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if bbox is not None else "N/A"
    
    # Get tile ID if available
    tile_id = data_dict.get('tile_id', str(tile_index))
    print(f"Tile ID: {tile_id} | Bounding box: {bbox_str}")
    
    if 'uav_points' in data_dict:
        uav_np = data_dict['uav_points'].cpu().numpy() if hasattr(data_dict['uav_points'], 'is_cuda') else data_dict['uav_points']
        num_uav = uav_np.shape[0]
        uav_min = np.min(uav_np[:, :3], axis=0)
        uav_max = np.max(uav_np[:, :3], axis=0)
        print(f"UAV pts: {num_uav:,} | Min: {uav_min.tolist()} | Max: {uav_max.tolist()}")
    
    if 'dep_points' in data_dict:
        dep_np = data_dict['dep_points'].cpu().numpy() if hasattr(data_dict['dep_points'], 'is_cuda') else data_dict['dep_points']
        num_dep = dep_np.shape[0]
        dep_min = np.min(dep_np[:, :3], axis=0)
        dep_max = np.max(dep_np[:, :3], axis=0)
        diff_min = (uav_min - dep_min).tolist() if 'uav_points' in data_dict else "N/A"
        diff_max = (uav_max - dep_max).tolist() if 'uav_points' in data_dict else "N/A"
        print(f"3DEP pts: {num_dep:,} | Min: {dep_min.tolist()} | Max: {dep_max.tolist()} | Diff Min: {diff_min} | Diff Max: {diff_max}")
    
    if 'uavsar_imgs' in data_dict and data_dict['uavsar_imgs']:
        meta = max(enumerate(data_dict.get('uavsar_imgs_meta', [])), key=lambda x: pd.Timestamp(x[1]['datetime']))[1]
        uavsar_img = data_dict['uavsar_imgs'][0]
        shape = uavsar_img.shape
        bands = meta.get('bands', list(range(1, shape[0]+1))) if shape[0] > 1 else "N/A"
        date = meta.get('date', meta.get('datetime', 'N/A'))
        print(f"UAVSAR: {shape} | Bands: {bands} | Date: {date}")
    
    if 'naip_imgs' in data_dict and data_dict['naip_imgs']:
        meta = max(enumerate(data_dict.get('naip_imgs_meta', [])), key=lambda x: pd.Timestamp(x[1]['datetime']))[1]
        naip_img = data_dict['naip_imgs'][0]
        shape = naip_img.shape
        bands = meta.get('bands', list(range(1, shape[0]+1))) if shape[0] > 1 else "N/A"
        date = meta.get('date', meta.get('datetime', 'N/A'))
        res = data_dict.get('naip_resolution', "N/A")
        num_naip = len(data_dict['naip_imgs'])
        print(f"NAIP: Res: {res} m, {num_naip} imgs | Img1: {shape} | Bands: {bands} | Date: {date}")
    
    print("-" * 80)


def reconstruct_raster_from_array(img_data, meta):
    # print(meta)
    """Reconstruct a raster from array data. Works with both tensor and numpy."""
    # More defensive metadata extraction
    bbox = meta.get('bbox', None)
    resolution = meta.get('resolution', None)
    
    # If bbox not in meta, try to get it from a higher level
    if bbox is None and 'bbox' in globals():
        bbox = globals()['bbox']
    
    if bbox is None or resolution is None:
        raise ValueError(f"Metadata must contain 'bbox' and 'resolution'. Got metadata: {meta}")
    
    # Convert tensor to numpy if needed
    img_np = img_data.cpu().numpy() if hasattr(img_data, 'is_cuda') else img_data
    
    
    # Handle various array shapes
    if img_np.ndim == 4:
        img_np = img_np[0]  # Take first batch item
    
    if img_np.ndim == 3:
        if img_np.shape[0] >= 3:  # Multi-band (RGB)
            rgb = img_np[:3, :, :]
            rgb = np.transpose(rgb, (1, 2, 0))
        else:  # Single band
            rgb = img_np[0, :, :]
    else:
        rgb = img_np  # Assume already in right shape
    
    # Create rasterio transform and extent
    xmin, ymin, xmax, ymax = bbox
    transform = from_origin(xmin, ymax, resolution, resolution)
    extent = (xmin, xmax, ymin, ymax)
    
    return rgb, extent, transform

def compute_dsm(points, bbox, resolution=0.5):
    """Compute a digital surface model from points."""
    # Original function unchanged
    xmin, ymin, xmax, ymax = bbox
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    dsm = np.full((height, width), np.nan, dtype=np.float32)
    cols = np.floor((points[:, 0] - xmin) / resolution).astype(int)
    rows = np.floor((points[:, 1] - ymin) / resolution).astype(int)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols = cols[valid]
    rows = rows[valid]
    zvals = points[valid, 2]
    for r, c, z in zip(rows, cols, zvals):
        if np.isnan(dsm[r, c]) or z > dsm[r, c]:
            dsm[r, c] = z
    return dsm

def plot_pointcloud_on_axis(ax, structured_pc, xlim, ylim, zlim, title, standard_ticks=None):
    x = structured_pc['X']
    y = structured_pc['Y']
    z = structured_pc['Z']
    ax.scatter(x, y, z, s=0.1, c=z, cmap='viridis')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)
    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

def plot_dsm(ax, dsm, extent, title, standard_ticks=None):
    im = ax.imshow(dsm, cmap='terrain', extent=extent, origin='lower')
    ax.set_title(title, fontsize=10)

    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_uavsar_grid(ax, img_tensor, meta, standard_ticks=None, band_names=None):
    try:
        raster, extent, _ = reconstruct_raster_from_array(img_tensor, meta)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        return

    # Default to displaying first band in grayscale if bands not specified
    if band_names is None or len(band_names) < 3:
        hh_band = raster[:, :, 0] if raster.ndim == 3 else raster
        vmin, vmax = np.percentile(hh_band, 2), np.percentile(hh_band, 98)
        ax.imshow(hh_band, cmap='gray', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
    else:
        # Get band indices from metadata
        bands_map = {}
        if 'bands' in meta:
            for i, band_name in enumerate(meta['bands']):
                bands_map[band_name] = i
        else:
            # Default mapping if not specified in metadata
            possible_bands = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]
            for i, band_name in enumerate(possible_bands):
                if i < (raster.shape[2] if raster.ndim == 3 else 1):
                    bands_map[band_name] = i
        
        # Create RGB composite
        rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.float32)
        
        # Extract bands for R, G, B channels
        for i, band_name in enumerate(band_names[:3]):
            if band_name in bands_map:
                band_idx = bands_map[band_name]
                if raster.ndim == 3 and band_idx < raster.shape[2]:
                    band_data = raster[:, :, band_idx]
                elif band_idx == 0:
                    band_data = raster
                else:
                    # Skip if band doesn't exist
                    continue
                
                # Normalize to 0-1 range for this channel
                vmin, vmax = np.percentile(band_data, 2), np.percentile(band_data, 98)
                normalized = np.clip((band_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
                rgb[:, :, i] = normalized
        
        ax.imshow(rgb, extent=extent, origin='upper')
    
    date_str = meta.get('date', meta.get('datetime', ''))
    title = f"{date_str}"
    if band_names and len(band_names) >= 3:
        title += f"\n{band_names[0]}-{band_names[1]}-{band_names[2]}"
    ax.set_title(title, fontsize=10)
    
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))

def plot_naip_grid(ax, img_tensor, meta, bbox_overlay=None, standard_ticks=None):
    try:
        rgb, extent, _ = reconstruct_raster_from_array(img_tensor, meta)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        return

    ax.imshow(rgb.astype(np.uint8), extent=extent, origin='upper')
    
    # Draw the original bounding box on top (if provided)
    if bbox_overlay is not None:
        rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                         bbox_overlay[2]-bbox_overlay[0],
                         bbox_overlay[3]-bbox_overlay[1],
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    date_str = meta.get('date', meta.get('datetime', ''))
    ax.set_title(f"{date_str}", fontsize=10)
    # ax.set_xlabel("X (CRS)")
    # ax.set_ylabel("Y (CRS)")
    
    # Apply standard ticks if provided
    if standard_ticks:
        x_ticks, y_ticks = standard_ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(np.linspace(extent[0], extent[1], num=5))
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5))



# --- Main Visualization Function ---

def visualize_tile_data(data_dict, tile_index=0, elev=45, azim=45, modalities=None, uavsar_bands=None):
    """
    Visualize selected modalities from a tile data dictionary.
    Works with both PyTorch tensors and NumPy arrays from HDF5.
    """
    if modalities is None:
        modalities = ["PointClouds", "DSMs", "NAIP"]
    
    print_tile_debug_info(data_dict, tile_index)
    
    figures = {}
    
    # Get the bounding box for standardized ticks
    bbox = data_dict.get('bbox')
    if bbox is None:
        print("Warning: No bounding box found in the data dictionary. Using point cloud extents instead.")
        uav_arr = array_to_structured_array(data_dict['uav_points'])
        dep_arr = array_to_structured_array(data_dict['dep_points'])
        all_x = np.concatenate([uav_arr['X'], dep_arr['X']])
        all_y = np.concatenate([uav_arr['Y'], dep_arr['Y']])
        bbox = (np.min(all_x), np.min(all_y), np.max(all_x), np.max(all_y))
    
    # Generate standardized ticks based on the bounding box
    x_ticks = np.linspace(bbox[0], bbox[2], num=3)
    y_ticks = np.linspace(bbox[1], bbox[3], num=3)

    def apply_standard_ticks(ax, is_3d=False):
        # Custom formatter that rounds to one decimal place and shows only last 3 digits
        def format_tick(x, pos):
            # Round to one decimal place
            rounded = round(x, 1)
            # Get the last 3 digits plus the decimal place
            last_digits = rounded % 1000
            # Format with one decimal place
            return f"{last_digits:.1f}"
        
        formatter = FuncFormatter(format_tick)
        
        if is_3d:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            # Apply the custom formatter
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            # Force the offset text to be hidden
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        else:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            # Apply the custom formatter
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            # Force the offset text to be hidden
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.tick_params(axis='both', which='both', labelsize=6)
        
    # Modified helper functions that apply standard ticks internally
    
    def plot_pointcloud_with_ticks(ax, structured_pc, xlim, ylim, zlim, title):
        x = structured_pc['X']
        y = structured_pc['Y']
        z = structured_pc['Z']
        ax.scatter(x, y, z, s=0.1, c=z, cmap='viridis')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=True)
    
    def plot_dsm_with_ticks(ax, dsm, extent, title):
        im = ax.imshow(dsm, cmap='terrain', extent=extent, origin='lower')
        ax.set_title(title, fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_uavsar_with_ticks(ax, img_tensor, meta, band_names=None, bbox_overlay=None):
        try:
            raster, extent, _ = reconstruct_raster_from_array(img_tensor, meta)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            return

        # Default to displaying first band in grayscale if bands not specified
        if band_names is None or len(band_names) < 3:
            hh_band = raster[:, :, 0] if raster.ndim == 3 else raster
            vmin, vmax = np.percentile(hh_band, 2), np.percentile(hh_band, 98)
            ax.imshow(hh_band, cmap='gray', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
        else:
            # Get band indices from metadata
            bands_map = {}
            if 'bands' in meta:
                for i, band_name in enumerate(meta['bands']):
                    bands_map[band_name] = i
            else:
                # Default mapping if not specified in metadata
                possible_bands = ["HHHH", "HHHV", "VVVV", "HVVV", "HVHV", "HHVV"]
                for i, band_name in enumerate(possible_bands):
                    if i < (raster.shape[2] if raster.ndim == 3 else 1):
                        bands_map[band_name] = i
            
            # Create RGB composite
            rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.float32)
            
            # Extract bands for R, G, B channels
            for i, band_name in enumerate(band_names[:3]):
                if band_name in bands_map:
                    band_idx = bands_map[band_name]
                    if raster.ndim == 3 and band_idx < raster.shape[2]:
                        band_data = raster[:, :, band_idx]
                    elif band_idx == 0:
                        band_data = raster
                    else:
                        # Skip if band doesn't exist
                        continue
                    
                    # Normalize to 0-1 range for this channel
                    vmin, vmax = np.percentile(band_data, 2), np.percentile(band_data, 98)
                    normalized = np.clip((band_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    rgb[:, :, i] = normalized
            
            ax.imshow(rgb, extent=extent, origin='upper')
        
        # Draw the original bounding box on top (if provided)
        if bbox_overlay is not None:
            rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                             bbox_overlay[2]-bbox_overlay[0],
                             bbox_overlay[3]-bbox_overlay[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        date_str = meta.get('date', meta.get('datetime', ''))
        title = f"{date_str}"
        if band_names and len(band_names) >= 3:
            title += f"\n{band_names[0]}-{band_names[1]}-{band_names[2]}"
        ax.set_title(title, fontsize=10)
        
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
    
    def plot_naip_with_ticks(ax, img_tensor, meta, bbox_overlay=None):
        try:
            rgb, extent, _ = reconstruct_raster_from_array(img_tensor, meta)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            return

        ax.imshow(rgb.astype(np.uint8), extent=extent, origin='upper')
        
        # Draw the original bounding box on top (if provided)
        if bbox_overlay is not None:
            rect = Rectangle((bbox_overlay[0], bbox_overlay[1]),
                             bbox_overlay[2]-bbox_overlay[0],
                             bbox_overlay[3]-bbox_overlay[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        date_str = meta.get('date', meta.get('datetime', ''))
        ax.set_title(f"{date_str}", fontsize=10)
        # Apply standard ticks directly
        apply_standard_ticks(ax, is_3d=False)
        
    # Compute common variables for point clouds and DSMs
    uav_arr = array_to_structured_array(data_dict['uav_points'])
    dep_arr = array_to_structured_array(data_dict['dep_points'])
    
    all_x = np.concatenate([uav_arr['X'], dep_arr['X']])
    all_y = np.concatenate([uav_arr['Y'], dep_arr['Y']])
    all_z = np.concatenate([uav_arr['Z'], dep_arr['Z']])
    xlim = (np.min(all_x), np.max(all_x))
    ylim = (np.min(all_y), np.max(all_y))
    zlim = (np.min(all_z), np.max(all_z))
    
    # Downsampled UAV points
    num_uav = data_dict['uav_points'].shape[0]
    mask = data_dict.get('uav_downsample_mask', np.ones(num_uav, dtype=bool))
    down_uav = data_dict['uav_points'][mask]
    num_down = down_uav.shape[0]
    num_dep = data_dict['dep_points'].shape[0]
    
    # Figure for Point Clouds
    if "PointClouds" in modalities:
        # Calculate natural limits for point clouds
        pc_xlim = (np.min(all_x), np.max(all_x))
        pc_ylim = (np.min(all_y), np.max(all_y))
        pc_zlim = (np.min(all_z), np.max(all_z))
        
        fig_pc = plt.figure(figsize=(18, 4))
        ax_uav = fig_pc.add_subplot(131, projection='3d')
        ax_uav_ds = fig_pc.add_subplot(132, projection='3d')
        ax_dep = fig_pc.add_subplot(133, projection='3d')
        
        # Use our new helper functions with built-in tick formatting
        plot_pointcloud_with_ticks(ax_uav, uav_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"UAV LiDAR ({num_uav:,} pts)")
        
        down_arr = array_to_structured_array(down_uav)
        plot_pointcloud_with_ticks(ax_uav_ds, down_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"Downsampled UAV ({num_down:,} pts)")
        
        plot_pointcloud_with_ticks(ax_dep, dep_arr, pc_xlim, pc_ylim, pc_zlim, 
                              title=f"3DEP LiDAR ({num_dep:,} pts)")
        
        figures["PointClouds"] = fig_pc

    # Figure for DSMs
    if "DSMs" in modalities:
        dsm_extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        uav_np = data_dict['uav_points'].cpu().numpy() if hasattr(data_dict['uav_points'], 'is_cuda') else data_dict['uav_points']
        dep_np = data_dict['dep_points'].cpu().numpy() if hasattr(data_dict['dep_points'], 'is_cuda') else data_dict['dep_points']
        dsm_uav = compute_dsm(uav_np, bbox, resolution=0.3333333)
        dsm_dep = compute_dsm(dep_np, bbox, resolution=0.3333333)
        
        fig_dsm, (ax_dsm_uav, ax_dsm_dep) = plt.subplots(1, 2, figsize=(9, 2))
        plot_dsm_with_ticks(ax_dsm_uav, dsm_uav, dsm_extent, title="UAV DSM (max Z)")
        plot_dsm_with_ticks(ax_dsm_dep, dsm_dep, dsm_extent, title="3DEP DSM (max Z)")
        
        figures["DSMs"] = fig_dsm

    # Figure for UAVSAR images with 3 plots per row and band composites
    if "UAVSAR" in modalities and data_dict.get('uavsar_imgs', []):
        uavsar_imgs = data_dict['uavsar_imgs']
        n_uavsar = len(uavsar_imgs)
        
        # Change from 2 to 3 plots per row
        n_cols_uavsar = 3
        n_rows_uavsar = math.ceil(n_uavsar / n_cols_uavsar)
        
        fig_uavsar, axes_uavsar = plt.subplots(n_rows_uavsar, n_cols_uavsar, figsize=(10, 3 * n_rows_uavsar))
        
        # Handle single row case
        if n_rows_uavsar == 1:
            axes_uavsar = np.array([axes_uavsar])
        
        for i in range(n_uavsar):
            row = i // n_cols_uavsar
            col = i % n_cols_uavsar
            plot_uavsar_with_ticks(axes_uavsar[row, col], uavsar_imgs[i], 
                                 data_dict['uavsar_imgs_meta'][i],
                                 band_names=uavsar_bands, 
                                 bbox_overlay=bbox)
        
        # Hide unused axes
        for i in range(n_uavsar, n_rows_uavsar * n_cols_uavsar):
            row = i // n_cols_uavsar
            col = i % n_cols_uavsar
            axes_uavsar[row, col].axis("off")
        
        figures["UAVSAR"] = fig_uavsar

    # Figure for NAIP images with bbox overlay
    if "NAIP" in modalities and data_dict.get('naip_imgs', []):
        naip_imgs = data_dict['naip_imgs']
        n_naip = len(naip_imgs)
        
        # Change from 2 to 3 plots per row
        n_cols_naip = 3
        n_rows_naip = math.ceil(n_naip / n_cols_naip)
        
        # Adjust figure size for 3 columns
        fig_naip, axes_naip = plt.subplots(n_rows_naip, n_cols_naip, figsize=(10, 3 * n_rows_naip))
        
        # Ensure axes_naip is always a 2D array even with just one row
        if n_rows_naip == 1:
            axes_naip = np.array([axes_naip])
        
        for i in range(n_naip):
            # Calculate row and column indices for 3 columns
            row = i // n_cols_naip
            col = i % n_cols_naip
            plot_naip_with_ticks(axes_naip[row, col], naip_imgs[i], 
                               data_dict['naip_imgs_meta'][i], 
                               bbox_overlay=bbox)
        
        # Hide unused axes
        for i in range(n_naip, n_rows_naip * n_cols_naip):
            row = i // n_cols_naip
            col = i % n_cols_naip
            axes_naip[row, col].axis("off")
        
        figures["NAIP"] = fig_naip
    
    return figures

def load_and_visualize_from_file(file_path, tile_index=0, **kwargs):
    """Load data from a PyTorch file and visualize it."""
    try:
        data = torch.load(file_path)
        if isinstance(data, list) and len(data) > 0:
            if tile_index < len(data):
                return visualize_tile_data(data[tile_index], tile_index=tile_index, **kwargs)
            else:
                print(f"Tile index {tile_index} is out of range; file contains {len(data)} tiles.")
                return None
        else:
            print("File does not contain a list of tile data or is empty.")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_and_visualize_from_h5(file_path, tile_id=None, tile_index=None, examine_metadata=False, **kwargs):
    """
    Load data from an HDF5 file and visualize it.
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    tile_id : str, optional
        Specific tile ID to load
    tile_index : int, optional
        Index of the tile to load (0-based)
    examine_metadata : bool, optional
        If True, print detailed metadata structure
    **kwargs : 
        Additional arguments for visualization
    """
    try:
        # Optionally print a summary of all tiles first
        if examine_metadata and tile_id is None and tile_index is None:
            print_tile_metadata_summary(file_path)
        
        # Load the tile
        data = load_tile_from_h5(file_path, tile_id=tile_id, tile_index=tile_index)
        
        # Optionally examine this tile's metadata
        if examine_metadata:
            examine_tile_metadata(data)
        
        # Visualize
        return visualize_tile_data(data, **kwargs)
    except Exception as e:
        print(f"Error loading and visualizing H5 file: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage:
# figs = load_and_visualize_from_file("path/to/your/file.pt", tile_index=0, 
#                                     modalities=["PointClouds", "DSMs", "UAVSAR", "NAIP"],
#                                     uavsar_bands=["HHHH", "HVHV", "VVVV"])
# if figs is not None:
#     figs["PointClouds"].show()
#     figs["DSMs"].show()
#     figs["UAVSAR"].show()
#     figs["NAIP"].show()