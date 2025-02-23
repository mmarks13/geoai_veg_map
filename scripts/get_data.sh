#!/bin/bash
cd /home/jovyan/geoai_veg_map/

# get naip imagery from planetary computer and save it locally. 
python src/data_prep/make_local_naip_stac.py\
  --bbox -116.674113 33.096041 -116.567707 33.170647 \
  --bbox -120.127900 34.649349 -119.938771 34.775782 \
  --start 2014-01-01 \
  --end 2025-12-31 \
  --output /home/jovyan/geoai_veg_map/data/stac/naip/


# get 3dep point clouds from planetary computer and save it locally. 
python src/data_prep/make_local_3dep_stac.py\
  --bbox -116.674113 33.096041 -116.567707 33.170647 \
  --bbox -120.127900 34.649349 -119.938771 34.775782 \
  --start 2014-01-01 \
  --end 2025-12-31 \
  --output /home/jovyan/geoai_veg_map/data/stac/3dep/