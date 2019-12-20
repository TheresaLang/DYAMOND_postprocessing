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
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
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
from moisture_space import utils

model = 'ICON'
run = '2.5km'
time_period = ['0830', '0908']
variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'RH', 'W']
variables_2D = ['OLR', 'IWV']
datapath = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/random_samples/'
filenames = '{}-{}_{}_sample_{}_{}-{}{}.nc'
num_profiles = int(1 * 1e6)
experiments = [0]#np.arange(10)
perc_values = np.arange(0, 100.5, 1.0)
iwv_bin_bnds = np.arange(0, 101, 1)
ic_thres = {
    'QI': 0.001 * 1e-6,
    'QC': 0.001 * 1e-6
}
    
#TODO: Jedes experiment einzeln und einzeln abspeichern
if len(experiments) == 1:
    filename = filenames.format(model, run, variables_3D[0], num_profiles, time_period[0], time_period[1], '')
else:
    filename = filenames.format(model, run, variables_3D[0], num_profiles, time_period[0], time_period[1], '_0')
    
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
        if model == 'ICON' and var == 'W':
            var_read = 'WHL'
        else:
            var_read = var
        if len(experiments) == 1:
            filename = filenames.format(model, run, var_read, num_profiles, time_period[0], time_period[1], '')
            print(filename)
            filename = join(datapath, filename)
        else:
            filename = filenames.format(model, run, var_read, num_profiles, time_period[0], time_period[1], '_'+str(exp))
            filename = join(datapath, filename)
        with(Dataset(filename)) as ds:
            profiles_sorted[var] = ds.variables[var_read][:].filled(np.nan)

    #Calculate UTH and IWP
    profiles_sorted['UTH'] = np.ones(len(profiles_sorted['RH'])) * np.nan
    profiles_sorted['IWP'] = np.ones(len(profiles_sorted['RH'])) * np.nan
    
    profiles_sorted['UTH'] = utils.calc_UTH(
        profiles_sorted['RH'], 
        profiles_sorted['QV'], 
        profiles_sorted['TEMP'], 
        profiles_sorted['PRES'], 
        height
    )
    profiles_sorted['IWP'] = utils.calc_IWP(
        profiles_sorted['QI'], 
        profiles_sorted['TEMP'], 
        profiles_sorted['PRES'], 
        profiles_sorted['QV'], 
        height
    )
    
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
    bin_count = bin_count + np.asarray([len(np.where(bin_idx == b)[0]) for b in bins])
    print(bin_count)

    for var in variables_2D+['UTH', 'IWP']:
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

            try:
                profiles_bin_quart25[var][b] = np.percentile(bin_profiles, 25)
                profiles_bin_quart75[var][b] = np.percentile(bin_profiles, 75)
                profiles_bin_max[var][b] = np.max(bin_profiles)
                profiles_bin_min[var][b] = np.min(bin_profiles)
            except:
                profiles_bin_quart25[var][b] = np.nan
                profiles_bin_quart75[var][b] = np.nan
                profiles_bin_max[var][b] = np.nan
                profiles_bin_min[var][b] = np.nan
                
            
    for var in variables_3D+['ICQI', 'ICQC', 'CFI', 'CFL']:
        profiles_bin_mean[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_std[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_median[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_quart25[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_quart75[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_max[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
        profiles_bin_min[var] = np.ones((num_levels, len(iwv_bin_bnds) - 1)) * np.nan
   
    for var in variables_3D: 
        for b in bins:
            bin_profiles = profiles_sorted[var][:, bin_idx == b]
            profiles_bin_mean[var][:, b] = np.mean(bin_profiles, axis=1)
            profiles_bin_std[var][:, b] = np.std(bin_profiles, axis=1) 
            profiles_bin_median[var][:, b] = np.median(bin_profiles, axis=1) 

            try:
                profiles_bin_quart25[var][:, b] = np.percentile(bin_profiles, 25, axis=1)
                profiles_bin_quart75[var][:, b] = np.percentile(bin_profiles, 75, axis=1)
                profiles_bin_max[var][:, b] = np.max(bin_profiles, axis=1)
                profiles_bin_min[var][:, b] = np.min(bin_profiles, axis=1)
            except:
                profiles_bin_quart25[var][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_quart75[var][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_max[var][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_min[var][:, b] = np.ones(num_levels) * np.nan
                
    
    for p in perc_values:
        perc = np.percentile(profiles_sorted['IWV'], p)
        percentiles.append(perc)
        percentiles_ind.append(np.argmin(np.abs(profiles_sorted['IWV'] - perc)))
        
    for var in variables_2D+['UTH', 'IWP']:
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
            
    for var in variables_3D+['ICQI', 'ICQC', 'CFI', 'CFL']:
        profiles_perc_mean[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan    
        profiles_perc_std[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_median[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_quart25[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_quart75[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_min[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
        profiles_perc_max[var] = np.ones((len(percentiles_ind)-1, num_levels)) * np.nan
    
    for var in variables_3D:
        print(var)
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
        
    for j in range(len(percentiles_ind)-1):
        start_ind = percentiles_ind[j]
        end_ind = percentiles_ind[j+1]
        
        for var, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
            q_profiles_perc = profiles_sorted[var][:, start_ind:end_ind]
            q_profiles_perc[q_profiles_perc <= ic_thres[var]] = np.nan
            profiles_perc_mean[content][j] = np.nanmean(q_profiles_perc, axis=1)
            profiles_perc_mean[fraction][j] = np.sum(~np.isnan(q_profiles_perc), axis=1) / q_profiles_perc.shape[1]
            profiles_perc_std[content][j] = np.nanstd(q_profiles_perc, axis=1)
            profiles_perc_median[content][j] = np.nanmedian(q_profiles_perc, axis=1)
            profiles_perc_quart25[content][j] = np.nanpercentile(q_profiles_perc, 25, axis=1)
            profiles_perc_quart75[content][j] = np.nanpercentile(q_profiles_perc, 75, axis=1)
            profiles_perc_min[content][j] = np.nanmin(q_profiles_perc, axis=1)
            profiles_perc_max[content][j] = np.nanmax(q_profiles_perc, axis=1)
    
    for b in bins:
        for var, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
            q_profiles_bin = profiles_sorted[var][:, bin_idx == b]
            q_profiles_bin[q_profiles_bin <= ic_thres[var]] = np.nan
            profiles_bin_mean[content][:, b] = np.nanmean(q_profiles_bin, axis=1)
            profiles_bin_mean[fraction][:, b] = np.sum(~np.isnan(q_profiles_bin), axis=1) / q_profiles_bin.shape[1]
            profiles_bin_std[content][:, b] = np.nanstd(q_profiles_bin, axis=1) 
            profiles_bin_median[content][:, b] = np.nanmedian(q_profiles_bin, axis=1) 

            try:
                profiles_bin_quart25[content][:, b] = np.nanpercentile(q_profiles_bin, 25, axis=1)
                profiles_bin_quart75[content][:, b] = np.nanpercentile(q_profiles_bin, 75, axis=1)
                profiles_bin_max[content][:, b] = np.nanmax(q_profiles_bin, axis=1)
                profiles_bin_min[content][:, b] = np.nanmin(q_profiles_bin, axis=1)
            except:
                profiles_bin_quart25[content][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_quart75[content][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_max[content][:, b] = np.ones(num_levels) * np.nan
                profiles_bin_min[content][:, b] = np.ones(num_levels) * np.nan
    
    # write output:
    #output files
    if len(experiments) > 1:
        outname_perc = f"{model}-{run}_{time_period[0]}-{time_period[1]}_perc_means_{num_profiles}_{exp}.pkl"
        outname_bins = f"{model}-{run}_{time_period[0]}-{time_period[1]}_bin_means_{num_profiles}_{exp}.pkl"
    else:
        outname_perc = f"{model}-{run}_{time_period[0]}-{time_period[1]}_perc_means_{num_profiles}.pkl"
        outname_bins = f"{model}-{run}_{time_period[0]}-{time_period[1]}_bin_means_{num_profiles}.pkl"
    
    perc = {}
    perc['mean'] = profiles_perc_mean
    perc['std'] = profiles_perc_std
    perc['median'] = profiles_perc_median
    perc['quart25'] = profiles_perc_quart25
    perc['quart75'] = profiles_perc_quart75
    perc['min'] = profiles_perc_min
    perc['max'] = profiles_perc_max
    perc['percentiles'] = percentiles
    with open(join(datapath, outname_perc), "wb" ) as outfile:
        pickle.dump(perc, outfile) 
        
    bins = {}
    bins['mean'] = profiles_bin_mean
    bins['std'] = profiles_bin_std
    bins['median'] = profiles_bin_median
    bins['quart25'] = profiles_bin_quart25
    bins['quart75'] = profiles_bin_quart75
    bins['min'] = profiles_bin_min
    bins['max'] = profiles_bin_max
    bins['count'] = bin_count
    with open(join(datapath, outname_bins), "wb" ) as outfile:
        pickle.dump(bins, outfile) 