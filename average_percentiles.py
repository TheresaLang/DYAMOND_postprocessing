#!/usr/bin/env python
#SBATCH --job-name=average_percentiles
#SBATCH --output=logs/average_percentiles-%j.out
##SBATCH --error=average_percentiles-%j.err

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

from importlib import reload
import numpy as np
import os
import random
import pickle
import multiprocessing
import functools
from os.path import join
from netCDF4 import Dataset
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import processing_tools as ptools
import analysis_tools as atools

model = 'NICAM'
run = '3.5km'
time_period = ['0830', '0908']
variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'RH']
variables_2D = ['OLR', 'IWV']
datapath = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/random_samples/'
filenames = '{}-{}_{}_sample_{}_{}-{}_{}.nc'
num_profiles = int(1 * 1e6)
experiments = np.arange(10)
perc_values = np.arange(0, 100.5, 1.0)

#TODO: Jedes experiment einzeln und einzeln abspeichern
filename = filenames.format(model, run, variables_3D[0], num_profiles, time_period[0], time_period[1], 0)
filename = join(datapath, filename)
with(Dataset(filename)) as ds:
    height = ds.variables['height'][:].filled(np.nan)
num_levels = len(height)
profiles_sorted = {}

for exp in experiments:
    print(exp)
    for var in variables_3D+variables_2D:
        print(var)
        filename = filenames.format(model, run, var, num_profiles, time_period[0], time_period[1], exp)
        filename = join(datapath, filename)
        with(Dataset(filename)) as ds:
            profiles_sorted[var] = ds.variables[var][:].filled(np.nan)

    percentiles = []
    percentiles_ind = []
    profiles_perc_mean = {}
    profiles_perc_std = {}
    profiles_perc_median = {}
    profiles_perc_quart25 = {}
    profiles_perc_quart75 = {}
    profiles_perc_max = {}
    profiles_perc_min = {}
    
    for p in perc_values:
        perc = np.percentile(profiles_sorted['IWV'], p)
        percentiles.append(perc)
        percentiles_ind.append(np.argmin(np.abs(profiles_sorted['IWV'] - perc)))
    for var in variables_2D:
        print(var)
        profiles_perc_mean[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_std[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_median[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_quart25[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_quart75[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_min[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        profiles_perc_max[var] = np.ones((len(percentiles_ind)-1)) * np.nan 
        
        for j in range(len(percentiles_ind)-1):
            start_ind = percentiles_ind[j]
            end_ind = percentiles_ind[j+1]
            profiles_perc_mean[var][j] = np.nanmean(profiles_sorted[var][start_ind:end_ind], axis=0)
            profiles_perc_std[var][j] = np.nanstd(profiles_sorted[var][start_ind:end_ind], axis=0)
            profiles_perc_median[var][j] = np.median(profiles_sorted[var][start_ind:end_ind], axis=0)
            profiles_perc_quart25[var][j] = np.percentile(profiles_sorted[var][start_ind:end_ind], 25, axis=0)
            profiles_perc_quart75[var][j] = np.percentile(profiles_sorted[var][start_ind:end_ind], 75, axis=0)
            profiles_perc_min[var][j] = np.min(profiles_sorted[var][start_ind:end_ind], axis=0)
            profiles_perc_max[var][j] = np.max(profiles_sorted[var][start_ind:end_ind], axis=0)
            
    for var in variables_3D:
        print(var)
        profiles_perc_mean[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan    
        profiles_perc_std[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_median[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_quart25[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_quart75[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_min[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_max[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        
        for j in range(len(percentiles_ind)-1):
            start_ind = percentiles_ind[j]
            end_ind = percentiles_ind[j+1]
            profiles_perc_mean[var][j] = np.nanmean(profiles_sorted[var][:, start_ind:end_ind], axis=1)
            profiles_perc_std[var][j] = np.nanstd(profiles_sorted[var][:, start_ind:end_ind], axis=1)
            profiles_perc_median[var][j] = np.median(profiles_sorted[var][:, start_ind:end_ind], axis=1)
            profiles_perc_quart25[var][j] = np.percentile(profiles_sorted[var][:, start_ind:end_ind], 25, axis=1)
            profiles_perc_quart75[var][j] = np.percentile(profiles_sorted[var][:, start_ind:end_ind], 75, axis=1)
            profiles_perc_min[var][j] = np.min(profiles_sorted[var][:, start_ind:end_ind], axis=1)
            profiles_perc_max[var][j] = np.max(profiles_sorted[var][:, start_ind:end_ind], axis=1)

    perc = {}
    perc['mean'] = profiles_perc_mean
    perc['std'] = profiles_perc_std
    perc['median'] = profiles_perc_median
    perc['quart25'] = profiles_perc_quart25
    perc['quart75'] = profiles_perc_quart75
    perc['min'] = profiles_perc_min
    perc['max'] = profiles_perc_max
    perc['percentiles'] = percentiles
    with open(join(datapath, f"{model}-{run}_{time_period[0]}-{time_period[1]}_perc_means_{num_profiles}_{exp}.pkl"), "wb" ) as outfile:
        pickle.dump(perc, outfile)
