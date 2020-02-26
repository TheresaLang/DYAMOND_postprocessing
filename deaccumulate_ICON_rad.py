#!/usr/bin/env python
#SBATCH --job-name=cdo_command
#SBATCH --output=logs/cdo_command-%j.out
##SBATCH --error=cdo_command-%j.err

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

import numpy as np
from netCDF4 import Dataset
import processing_tools as ptools

model = 'ICON'
run = '5.0km_1'
variable = 'STOA'
variable_name = 'STOA'
path2file = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{model}-{run}_{variable}_hinterp_merged_0801-0910_acc.nc'

print('read data')
with Dataset(path2file) as ds:
     #olr = ds.variables['ATHB_T'][:].filled(np.nan)
    time = ds.variables['time'][:].filled(np.nan)
    lat = ds.variables['lat'][:].filled(np.nan)
    lon = ds.variables['lon'][:].filled(np.nan)
num_timesteps = time.shape[0]
num_lat = lat.shape[0]
num_lon = lon.shape[0]

print('calc')
olr_new = np.zeros((num_timesteps, num_lat, num_lon))
for i in range(1, num_timesteps):
    print(i)
    with Dataset(path2file) as ds:
        olr = ds.variables[variable_name][i-1:i+1].filled(np.nan)
        olr_new[i] = i * olr[1] - (i-1) * olr[0]

olr_new_3h = olr_new[::12]
    
time_new = np.arange(olr_new_3h.shape[0])
outname = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{variable}_test.py'
#{model}-{run}_{variable}_hinterp_merged_0801-0908.nc'
ptools.latlonfield_to_netCDF(lat, lon, olr_new_3h, variable_name, 'W m^-2', outname, time_dim=True, time_var=time_new, overwrite=True)