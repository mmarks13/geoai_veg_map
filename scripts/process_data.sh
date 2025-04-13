




# python src/data_prep/process_uav_lidar.py
# python src/data_prep/create_training_tile_bboxes.py 

# # based the tile bounding boxes, generate training data from the local stac catalogs of UAVSAR, UAV LiDAR, and NAIP, and the planetary computer 3DEP stac
# python src/data_prep/generate_training_data.py\
#  --tiles_geojson data/processed/tiles.geojson \
#  --lidar_stac_source data/stac/uavlidar/catalog.json \
#  --outdir data/processed/training_data_chunks \
#  --chunk_size 200 \
#  --max-api-retries 20\
#  --uavsar_stac_source data/stac/uavsar/catalog.json \
#  --naip_stac_source data/stac/naip/catalog.json\
#  --threads 12 \
#  --initial-voxel-size-cm 4 \
#  --max-points 20000

# #this script combines the training data chunks into a single pytorch file
# python src/data_prep/h5_chunk_loader.py \
#  --input_dir data/processed/training_data_chunks \
#  --output_path data/processed/model_data/combined_training_data_v2.pt \
#  --verbose


# ## generate a geojson file with the footprints of the point clouds
# # python src/data_prep/pointcloud_footprints_to_geojson.py
# #
# ## the resuling geojson file is used to create the training and test regions
# ## this was done manually in qgis

# #split the training data into training, validation, and test sets and filter out tiles with too few points
# python src/data_prep/split_train_test_val_tiles.py \
#     --pt-file data/processed/model_data/combined_training_data_v2.pt \
#     --geojson-file /home/jovyan/geoai_veg_map/data/processed/test_val_polygons.geojson \
#     --output-dir data/processed/model_data \
#     --min-uav-points 18000 \
#     --min-dep-points 200 \
#     --min-uav-to-dep-ratio 1.1 \
#     --min-coverage 95 \
#     --test-val-ratio 0.6 \
#     --min-points-per-cell 6 \
#     --random-seed 123 

#precompute data for pytorch training
python src/data_prep/precompute_data.py

# # augment the original training data
# python src/data_prep/data_augmentation.py