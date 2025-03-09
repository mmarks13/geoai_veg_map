#!/bin/bash
cd /home/jovyan/geoai_veg_map/

#set environment variables for getting uavsar data
export EARTHDATA_USERNAME=mmarks13
export EARTHDATA_PASSWORD=vuj@zmp2CQX5bkp2kbd

# #UAVSAR
# python src/data_prep/make_local_uavsar_stac.py\
#   --bbox -120.127900 34.649349 -119.938771 34.775782 \
#   --bbox -116.674113 33.096041 -116.567707 33.170647 \
#   --start 2014-01-01 \
#   --end 2024-12-31 \
#   --output /home/jovyan/geoai_veg_map/data/stac/uavsar/ \
#   --temp data/raw/uavsar

# #UAV LiDAR Point Clouds
# python src/data_prep/make_local_uavlidar_stac.py \
#   --input data/raw/uavlidar/study_las \
#   --output data/stac/uavlidar \
#   --collection-id uav_lidar \
#   --collection-title "UAV LiDAR Point Clouds"\
#   --target-crs "EPSG:32611"  #\
#   #--filename-regex "210014|TREX"

# # get naip imagery from planetary computer and save it locally. 
# python src/data_prep/make_local_naip_stac.py\
#   --bbox -116.674113 33.096041 -116.567707 33.170647 \
#   --bbox -120.127900 34.649349 -119.938771 34.775782 \
#   --start 2014-01-01 \
#   --end 2025-12-31 \
#   --output /home/jovyan/geoai_veg_map/data/stac/naip/


# # get 3dep point clouds from planetary computer and save it locally. 
# python src/data_prep/make_local_3dep_stac.py\
#   --bbox -116.674113 33.096041 -116.567707 33.170647 \
#   --bbox -120.127900 34.649349 -119.938771 34.775782 \
#   --start 2014-01-01 \
#   --end 2025-12-31 \
#   --output /home/jovyan/geoai_veg_map/data/stac/3dep/



# python src/data_prep/process_uav_lidar.py
# python src/data_prep/create_training_tile_bboxes.py 

python src/data_prep/generate_training_data.py\
 --tiles_geojson data/processed/tiles.geojson \
 --lidar_stac_source data/stac/uavlidar/catalog.json \
 --outdir data/processed/training_data_chunks/ \
 --chunk_size 200 \
 --max-api-retries 20\
 --uavsar_stac_source data/stac/uavsar/catalog.json \
 --naip_stac_source data/stac/naip/catalog.json\
 --threads 12 \
 --initial-voxel-size-cm 4 \
 --max-points 20000