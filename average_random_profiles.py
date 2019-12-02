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

model = 'ICON'
run = '2.5km'
time_period = ['0830', '0908']
variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'RH', 'WHL']
variables_2D = ['OLR', 'IWV']
datapath = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/random_samples/'
filenames = '{}-{}_{}_sample_{}_{}-{}_{}.nc'
num_profiles = int(1 * 1e6)
experiments = np.arange(10)
perc_values = np.arange(0, 100.5, 1.0)
iwv_bin_bnds = np.arange(0, 101, 1)
icic_thres = 0.

#TODO: Jedes experiment einzeln und einzeln abspeichern
filename = filenames.format(model, run, variables_3D[0], num_profiles, time_period[0], time_period[1], 0)
filename = join(datapath, filename)
with(Dataset(filename)) as ds:
    height = ds.variables['height'][:].filled(np.nan)
num_levels = len(height)
profiles_sorted = {}
bin_count = np.zeros(len(iwv_bin_bnds) - 1)
bins = range(len(iwv_bin_bnds) - 1) 

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
    
    profiles_bin_mean = {}
    profiles_bin_std = {}
    profiles_bin_median = {}
    profiles_bin_quart25 = {}
    profiles_bin_quart75 = {}
    profiles_bin_max = {}
    profiles_bin_min = {}
    
    # Binning to IWV bins
    bin_idx = np.digitize(profiles_sorted['IWV'], iwv_bin_bnds)
    bin_count = bin_count + np.asarray([len(np.where(bin_idx == i)[0]) for i in bins])
    print(bin_count)

    for var in variables_2D:
        profiles_bin_mean[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_std[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_median[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_quart25[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_quart75[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_max[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        profiles_bin_min[var] = np.ones(len(iwv_bin_bnds) - 1) * np.nan
        
        for b in bins:
            bin_profiles = profiles_sorted[var][bin_idx == b]
            profiles_bin_mean[var][b] = np.mean(bin_profiles) 
            profiles_bin_std[var][b] = np.std(bin_profiles)
            profiles_bin_median[var][b] = np.median(bin_profiles)
            profiles_bin_quart25[var][b] = np.percentile(bin_profiles, 25)
            profiles_bin_quart75[var][b] = np.percentile(bin_profiles, 75)
            profiles_bin_max[var][b] = np.max(bin_profiles, 25)
            profiles_bin_min[var][b] = np.min(bin_profiles, 25)
            
    for var in variables_3D:
        profiles_bin_mean[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_std[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_median[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_quart25[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_quart75[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_max[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_min[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        
        for b in bins:
            bin_profiles = profiles_sorted[var][:, bin_idx == b]
            profiles_bin_mean[var][:, b] = np.mean(bin_profiles, axis=1)
            profiles_bin_std[var][:, b] = np.std(bin_profiles, axis=1) 
            profiles_bin_median[var][:, b] = np.median(bin_profiles, axis=1)
            profiles_bin_quart25[var][:, b] = np.percentile(bin_profiles, 25, axis=1)
            profiles_bin_quart75[var][:, b] = np.percentile(bin_profiles, 75, axis=1) 
            profiles_bin_max[var][:, b] = np.max(bin_profiles, axis=1)
            profiles_bin_min[var][:, b] = np.min(bin_profiles, axis=1)
    
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
    
    # calculate in-cloud ice content
    # add integrated cloud ice?
    for var in ['ICQI', 'CF']:
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
        qi_profiles_perc = profiles_sorted['QI'][:, start_ind:end_ind]
        qi_profiles_perc[qi_profiles_perc <= icic_thres] = np.nan
        profiles_perc_mean['ICQI'][j] = np.nanmean(qi_profiles_perc, axis=1)
        profiles_perc_mean['CF'][j] = np.sum(~np.isnan(qi_profiles_perc), axis=1) / qi_profiles_perc.shape[1]
        profiles_perc_std['ICQI'][j] = np.nanstd(qi_profiles_perc, axis=1)
        profiles_perc_median['ICQI'][j] = np.median(qi_profiles_perc, axis=1)
        profiles_perc_quart25['ICQI'][j] = np.percentile(qi_profiles_perc, 25, axis=1)
        profiles_perc_quart75['ICQI'][j] = np.percentile(qi_profiles_perc, 75, axis=1)
        profiles_perc_min['ICQI'][j] = np.min(qi_profiles_perc, axis=1)
        profiles_perc_max['ICQI'][j] = np.max(qi_profiles_perc, axis=1)
            
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
        
    bins = {}
    bins['mean'] = profiles_bin_mean
    bins['std'] = profiles_bin_std
    bins['median'] = profiles_bin_median
    bins['quart25'] = profiles_bin_quart25
    bins['quart75'] = profiles_bin_quart75
    bins['min'] = profiles_bin_min
    bins['max'] = profiles_bin_max
    with open(join(datapath, f"{model}-{run}_{time_period[0]}-{time_period[1]}_bin_means_{num_profiles}_{exp}.pkl"), "wb" ) as outfile:
        pickle.dump(bins, outfile) 