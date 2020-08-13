#!/usr/bin/bash

for N in {1..6}; do 
    cdo import_fv3grid /mnt/lustre02/work/ka1081/DYAMOND/FV3-3.25km/2016081100/grid_spec.tile$N.nc /mnt/lustre02/work/mh1126/m300773/DYAMOND/FV3/FV3-3.25km_gridspec.tile$N.nc
done
cdo collgrid /mnt/lustre02/work/mh1126/m300773/DYAMOND/FV3/FV3-3.25km_gridspec.tile?.nc /mnt/lustre02/work/mh1126/m300773/DYAMOND/FV3/FV3-3.25km_gridspec.nc