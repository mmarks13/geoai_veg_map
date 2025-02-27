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
#   --collection-title "UAV LiDAR Point Clouds"


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
python src/data_prep/create_training_tile_bboxes.py