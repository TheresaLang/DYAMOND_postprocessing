#!/usr/bin/env python
#SBATCH --job-name=temporal_variability
#SBATCH --output=logs/temporal_variability-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from moisture_space import plots
import matplotlib.gridspec as gridspec

model = 'MPAS'
run = '3.75km'
num_profiles = 1000000
variables_3D = ['RH', 'QI']
variables_2D = ['OLR']
time_bounds = ['2016-08-10', '2016-09-08']
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
time_periods_array = [2, 3, 6, 10, 15]
time_periods = time_periods_array[ID]

time = pd.date_range(time_bounds[0], time_bounds[1], freq='1D')
days_total = len(time)
days_per_period = days_total // time_periods
perc_values = np.arange(1, 101, 1)
filebase = '{}-{}_{}_sample_{}_{}-{}.nc'
datapath = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/random_samples'
time_start = time[0].strftime('%m%d')
time_end = time[-1].strftime('%m%d')
outname = f'temporal_variablilty_perc_means_{time_periods}_periods_{time_start}-{time_end}.pkl'
print(os.path.join(datapath, outname))

def average_profiles(profiles_sorted_IWV, profiles_sorted, percs, var, var_3D=True):
    percentiles = []
    percentiles_ind = []
    for p in percs:
        perc = np.percentile(profiles_sorted_IWV, p)
        percentiles.append(perc)
        percentiles_ind.append(np.argmin(np.abs(profiles_sorted_IWV - perc)))

    profiles_perc_mean = {}
    profiles_perc_sdt = {}

    if var_3D:
        num_levels = profiles_sorted.shape[0]
        profiles_perc_mean = np.ones((len(percentiles_ind), num_levels)) * np.nan 
        print(profiles_perc_mean.shape)
        profiles_perc_std = np.ones((len(percentiles_ind), num_levels)) * np.nan

        for j in range(len(percentiles_ind)-1):
            start_ind = percentiles_ind[j]
            end_ind = percentiles_ind[j+1]
            profiles_perc_mean[j] = np.nanmean(profiles_sorted[:, start_ind:end_ind], axis=1)
            profiles_perc_std[j] = np.nanstd(profiles_sorted[:, start_ind:end_ind], axis=1)
    else:
        profiles_perc_mean = np.ones((len(percentiles_ind))) * np.nan 
        profiles_perc_std = np.ones((len(percentiles_ind))) * np.nan 

        for j in range(len(percentiles_ind)-1):
            start_ind = percentiles_ind[j]
            end_ind = percentiles_ind[j+1]
            profiles_perc_mean[j] = np.nanmean(profiles_sorted[start_ind:end_ind], axis=0)
            profiles_perc_std[j] = np.nanstd(profiles_sorted[start_ind:end_ind], axis=0)
   
    return profiles_perc_mean, profiles_perc_std

filename = filebase.format(model, run, variables_3D[0], num_profiles, time[0].strftime('%m%d'), time[0].strftime('%m%d'))
filename = os.path.join(datapath, filename)
with Dataset(filename) as ds:
    height = ds.variables['height'][:].filled(np.nan)
num_levs = len(height)

profiles = {}
for var in ['IWV']+variables_2D+variables_3D:
    print(var)
    if var in variables_3D:
        profiles[var] = np.ones((days_total, num_levs, num_profiles)) * np.nan
    else:
        profiles[var] = np.ones((days_total, num_profiles)) * np.nan
    for i, d in enumerate(time):
        timestr = d.strftime('%m%d')    
        filename = filebase.format(model, run, var, num_profiles, timestr, timestr)
        filename = os.path.join(datapath, filename)
        with Dataset(filename) as ds:
            profiles[var][i] = ds.variables[var][:].filled(np.nan)  
            
profiles_selected = {}
num_profiles_reshape = np.arange(0, num_profiles, days_per_period).shape[0] * days_per_period
profiles_selected['IWV'] = np.ones((time_periods, num_profiles)) * np.nan
sort_idx = np.ones((time_periods, num_profiles)) * np.nan
for p in range(time_periods):
    start = p * days_per_period
    print(start)
    end = start + days_per_period
    print(end)
    #print(profiles['IWV'][start:end, ::days_per_period].shape)
    profiles_selected['IWV'][p, :] = profiles['IWV'][start:end, ::days_per_period].reshape((num_profiles_reshape))[:num_profiles] 
    sort_idx[p] = np.argsort(profiles_selected['IWV'][p]).astype(int)
    profiles_selected['IWV'][p] = profiles_selected['IWV'][p][sort_idx[p].astype(int)]

for var in variables_2D:
    profiles_selected[var] = np.ones((time_periods, num_profiles)) * np.nan
    for p in range(time_periods):
        start = p * days_per_period
        end = start + days_per_period
        sortidx = sort_idx[p].astype(int)
        profiles_selected[var][p, :] = profiles[var][start:end, ::days_per_period].reshape((num_profiles_reshape))[:num_profiles][sortidx]  
        
for var in variables_3D:
    profiles_selected[var] = np.ones((time_periods, num_levs, num_profiles)) * np.nan
    for p in range(time_periods):
        start = p * days_per_period
        end = start + days_per_period
        #profiles_period = profiles[var][start:end, :, ::days_per_period]
        sortidx = sort_idx[p].astype(int)
        profiles_selected[var][p, :, :] = profiles[var][start:end, :, ::days_per_period].transpose(1, 0, 2).reshape((num_levs, -1))[:, sortidx]
        
profiles_perc_mean = {}
profiles_perc_std = {}
for var in variables_3D+variables_2D+['IWV']:
    print(var)
    var_3D = var in variables_3D
    if var_3D:
        profiles_perc_mean[var] = np.ones((time_periods, len(perc_values), num_levs)) * np.nan
        profiles_perc_std[var] = np.ones((time_periods, len(perc_values), num_levs)) * np.nan
    else:
        profiles_perc_mean[var] = np.ones((time_periods, len(perc_values))) * np.nan
        profiles_perc_std[var] = np.ones((time_periods, len(perc_values))) * np.nan
    for p in range(time_periods):
        print(p)
        profiles_perc_mean[var][p], profiles_perc_std[var][p] = average_profiles(profiles_selected['IWV'][p], profiles_selected[var][p], perc_values, var, var_3D=var_3D)
        
perc = {}
perc['mean'] = profiles_perc_mean
perc['std'] = profiles_perc_std
with open(os.path.join(datapath, outname), "wb" ) as outfile:
    pickle.dump(perc, outfile) 