import numpy as np
import pandas as pd
import os
import logging
import json
import typhon
import random
import pickle
import filelists
import netCDF_tools as nctools
import xarray as xr
import filenames
from netCDF4 import Dataset
from moisture_space import utils
from importlib import reload

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def config():
    """ Reads specifications for processing from confing.json and returns a dictionary
    with specifications.
    """
    with open('/mnt/lustre02/work/um0878/users/tlang/work/dyamond/processing/postprocessing/postprocessing_config.json') as handle:
        config = json.loads(handle.read())
    
    return config

def read_var(infile, model, varname):
    """ Reads in one variable from a given netCDF file.
    
    Parameters:
        infile (str): Path to input file
        model (str): Name of model that the data belongs to
        varname (str): Name of netCDF variable to read in
    
    Returns:
        float or ndarray: Data array with variable
    """
    with Dataset(infile) as ds:
        if model == 'SAM' and varname == 'PRES':
            pres_mean = ds.variables['p'][:].filled(np.nan)
            pres_pert = ds.variables['PP'][:].filled(np.nan)
            var = np.zeros((pres_pert.shape))
            for k in range(pres_pert.shape[0]):
                for i in range(pres_pert.shape[2]):
                    for j in range(pres_pert.shape[3]):
                        var[k, :, i, j] = pres_mean * 1e2 + pres_pert[k, :, i, j]
        else:
            var = ds.variables[varname][:].filled(np.nan)
            
        if model == 'MPAS' and len(var.shape) == 4 and varname != 'RH': 
            var = var.transpose((0, 3, 1, 2))
            
    return var

def read_var_timestep(infile, model, varname, timestep):
    """ Reads in one variable from a given netCDF file.
    
    Parameters:
        infile (str): Path to input file
        model (str): Name of model that the data belongs to
        varname (str): Name of netCDF variable to read in
        timestep (int): Timestep of field to read in
        
    Returns:
        float or ndarray: Data array with variable
    """
    with Dataset(infile) as ds:
        if model == 'SAM' and varname == 'PRES':
            pres_mean = ds.variables['p'][:].filled(np.nan)
            pres_pert = ds.variables['PP'][timestep].filled(np.nan)
            var = np.zeros((pres_pert.shape[0], pres_pert.shape[1], pres_pert.shape[2]))
            for i in range(pres_pert.shape[1]):
                for j in range(pres_pert.shape[2]):
                    var[:, i, j] = pres_mean * 1e2 + pres_pert[:, i, j]
        else:
            var = ds.variables[varname][timestep].filled(np.nan)
            
        if model == 'MPAS' and len(var.shape) == 4 and varname != 'RH': 
            var = var.transpose((0, 3, 1, 2))
            
    return var

def read_var_timestep_latlon(infile, model, variable, timestep, lat_inds, lon_inds):
    """
    """
    variables_3D = varlist_3D()
    if variable in variables_3D:
        is_3D = True
    else:
        is_3D = False
        
    if variable == 'SST':
        timestep = timestep // 2
        
    if is_3D:
        with Dataset(infile) as ds:
            if model == 'MPAS' and variable in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
                profiles = ds.variables[variable][timestep][lat_inds, lon_inds, :].filled(np.nan).transpose([1, 0])
            else:
                if model == 'SAM' and variable == 'PRES':
                    pres_mean = np.expand_dims(ds.variables['p'][:], 1) * 1e2
                    pres_pert = ds.variables['PP'][timestep][:, lat_inds, lon_inds].filled(np.nan)
                    profiles = pres_mean + pres_pert
                else:
                    profiles = ds.variables[variable][timestep][:, lat_inds, lon_inds].filled(np.nan)
                        
        if model == 'SAM' and var in ['QV', 'QI', 'QC']:
            profiles *= 1e-3
        profiles = profiles.T
    
    else:
        with Dataset(infile) as ds:
            if model == 'NICAM' and var != 'SST':
                profiles = ds.variables[variable][timestep][0, lat_inds, lon_inds].filled(np.nan)
            elif model == 'IFS' and var == 'SURF_PRES':
                profiles = np.exp(ds.variables[variable][timestep][0, lat_inds, lon_inds].filled(np.nan))
            else:
                profiles = ds.variables[variable][timestep][lat_inds, lon_inds].filled(np.nan)
                                                       
    return profiles

def read_dimensions(infile, varname):
    """
    """
    with Dataset(infile) as ds:
        dimensions = ds.variables[varname].shape
        
    return dimensions
    
    
def read_latlon(infile):
    """
    """
    with Dataset(infile) as ds:
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        
    return lat, lon

def save_random_profiles(outname, profiles, variable, height):
    """
    """
    profile_inds = np.arange(profiles.shape[0])
    
    if variable in varlist_3D():
        ds = xr.Dataset(
            {
                variable: (["profiles_tot", "levels"], profiles),
            },
            coords = {
                "profiles_tot": (["profiles_tot"], profile_inds),
                "levels": (["levels"], height)
            }
        )
    elif variable in varlist_2D():
        ds = xr.Dataset(
            {
                variable: (["profiles_tot"], profiles)
            },
            coords = {
                "profiles_tot": (["profiles_tot"], profile_inds)
            }
        )
        
    ds.to_netcdf(outname)
    
def read_random_profiles(infile):
    """
    """
    ds = xr.load_dataset(infile)
    height = ds.coords['levels']
    variable = list(ds.keys())[0]
    profiles = ds[variable].data
    
    return height, profiles

def varlist_2D():
    """
    """
    varlist = ['OLR', 'OLRC', 'STOA', 'IWV', 'CRH', 'TQI', 'TQC', 'TQG',\
               'TQS', 'TQR', 'SST', 'SURF_PRES', 'lat', 'lon', 'idx']
    
    return varlist

def varlist_3D():
    """
    """
    varlist = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'QS', 'QG', 'QR', 'RH',\
               'W', 'DRH_Dt_v', 'DRH_Dt_h', 'DRH_Dt_c', 'A_RH_v', 'A_QV_v',\
               'A_RH_h', 'A_QV_h', 'U', 'V', 'UV', 'dRH_dx', 'dRH_dy', 'dRH_dz']
    
    return varlist

def varlist_log_average():
    """
    """
    varlist = ['PRES', 'QV', 'H2O']
    
    return varlist
    
def create_random_indices(mask, num_timesteps, num_samples_per_timestep):
    """
    """
    
    mask_ind = np.where(mask)
    lat_not_masked = mask_ind[0]
    lon_not_masked = mask_ind[1]

    lat_inds = np.zeros((num_timesteps, num_samples_per_timestep)).astype(int)
    lon_inds = np.zeros((num_timesteps, num_samples_per_timestep)).astype(int)

    logger.info('Get indices')
    for t in range(num_timesteps):
        r = random.sample(list(zip(lat_not_masked, lon_not_masked)), num_samples_per_timestep)
        lat_inds[t], lon_inds[t] = zip(*r)
        
    return lat_inds, lon_inds

def read_random_indices(infile):
    """
    """
    ds = xr.load_dataset(infile)
    lat_inds = ds['lat_inds'].data
    lon_inds = ds['lon_inds'].data
    sort_inds = ds['sort_inds'].data
    
    return lat_inds, lon_inds, sort_inds

def save_random_indices(outfile, lat_inds, lon_inds, sort_inds):
    """
    """
    profile_inds_tot = np.arange(sort_inds.shape[0])
    profile_inds_timestep = np.arange(lat_inds.shape[1])
    timesteps = np.arange(lat_inds.shape[0])
    # Create xarray.Dataset
    ds = xr.Dataset(
        {
            "lat_inds": (["timesteps", "profiles_timestep"], lat_inds),
            "lon_inds": (["timesteps", "profiles_timestep"], lon_inds),
            "sort_inds": (["profiles_tot"], sort_inds)
        },
        coords = {
            "timesteps": (["timesteps"], timesteps),
            "profiles_timestep": (["profiles_timestep"], profile_inds_timestep),
            "profiles_tot": (["profiles_tot"], profile_inds_tot)
        }
    )
    # Save Dataset
    ds.to_netcdf(outfile)
    
def get_not_nan_mask(field):
    """
    """
    return np.logical_not(np.isnan(field))

def get_latlon_mask(lat, lon, lonlatbox):
    """
    """
    lat_reg_ind = np.logical_and(lat >= lonlatbox[2], lat <= lonlatbox[3])
    lon_reg_ind = np.logical_and(lon >= lonlatbox[0], lon <= lonlatbox[1])
    lat_mask = np.tile(lat_reg_ind, (len(lon), 1)).T
    lon_mask = np.tile(lon_reg_ind, (len(lat), 1))
    latlon_mask = np.logical_and(lon_mask, lat_mask)
    
    return latlon_mask

def get_ocean_mask(model, landmask_file):
    """
    """
    trop_lat = [-30, 30]
    ocean_mask = np.logical_not(np.squeeze(read_var(landmask_file, model, 'land_mask')))
    ocean_lat, ocean_lon = read_latlon(landmask_file)
    # cut out tropical region
    ocean_lat_ind = np.where(np.logical_and(ocean_lat >= trop_lat[0], ocean_lat <= trop_lat[1]))[0]
    ocean_mask = ocean_mask[ocean_lat_ind].astype(int)
    
    return ocean_mask

def combine_masks(mask_list):
    """
    """
    combined_mask = mask_list[0]
    for mask in mask_list[1:]:
        combined_mask = np.logical_and(combined_mask, mask)
        
    return combined_mask

def get_surface_ind(height):
    """
    """
    if height[0] < height[-1]:
        surface_ind = 0
    else:
        surface_ind = -1
    
    return surface_ind


def get_timesteps(model, timestep_param, timesteps_tot, sample_days):
    """
    """
    if timestep_param is not None:
        timesteps = timestep_param
        num_timesteps = len(timesteps)
    else:
        if sample_days.lower() == 'all':
            num_timesteps = timesteps_tot
            timesteps = np.arange(num_timesteps)
        elif sample_days.lower() == 'first10':
            num_timesteps = 80
            timesteps = np.arange(0, num_timesteps)
        elif sample_days.lower() == 'last10':
            num_timesteps = 80
            timesteps = np.arange(dimensions[0] - 80, dimensions[0])
            
    if model in ['MPAS', 'IFS', 'ARPEGE']:
        # For these models there is one timestep less for OLR and STOA 
        timesteps = timesteps[:-1]
        num_timesteps = num_timesteps - 1
            
    return timesteps, num_timesteps

def percentiles_from_number(num_percentiles):
    """
    """
    start = 100 / num_percentiles
    step = 100 / num_percentiles
    stop = 100 + 0.5 * step
    percentiles = np.arange(start, stop, step)
    
    return percentiles
    
    
def select_random_profiles_new(model, run, variables, time_period, data_dir, num_samples,
                               new_sampling_idx, sample_days='all', timesteps=None, 
                               filename_suffix='', **kwargs):
    """
    """

    # make sure that all variables needed to calculate IWV are in the list
    for v in ['TEMP', 'PRES', 'QV']:
        if v not in variables:
            variables.append(v)
            
    # longitudes and latitudes to sample from
    lonlatbox = [-180, 180, -29.9, 29.9]

    # lists of variables (2D and 3D)
    list_2D = varlist_2D()
    list_3D = varlist_3D()
    variables_2D = [v for v in variables if v in list_2D]
    variables_3D = [v for v in variables if v in list_3D]

    # filenames
    infiles = {}
    for variable in variables:
        infiles[variable] = filenames.preprocessed_output(data_dir, model, run, variable, num_samples, time_period)
    height_file = filenames.heightfile(data_dir, model)
    landmask_file = filenames.landmaskfile(data_dir)
    random_ind_file = filenames.random_ind(data_dir, model, run, num_samples, time_period)
    # suffix for outfiles
    if not filename_suffix and sample_days != 'all':
        filename_suffix = sample_days
    
    # Test file to get some general information (lats, lons, dimensions) from
    test_var = variables_3D[0]
    test_file = infiles[test_var]
    tot_timesteps_avail = read_dimensions(test_file, test_var)[0]

    logger.info('Read height')
    height = read_var(height_file, model, 'target_height')
    num_levels = len(height)
    surface_ind = get_surface_ind(height)

    if new_sampling_idx:
        logger.info('Create new sampling indices')
        # lats and lons
        lat, lon = read_latlon(test_file)

        # timesteps
        timesteps, num_timesteps = get_timesteps(model, timesteps, tot_timesteps_avail, sample_days)

        # number of samples per timestep
        num_samples_timestep = int(num_samples / num_timesteps)
        num_samples_exact = num_samples_timestep * num_timesteps

        # Dummy field to create NaN mask
        test_field = read_var_timestep(test_file, model, test_var, timestep=0)

        # Create masks
        not_nan_mask = get_not_nan_mask(test_field[surface_ind])
        ocean_mask = get_ocean_mask(model, landmask_file)
        latlon_mask = get_latlon_mask(lat, lon, lonlatbox)
        total_mask = combine_masks([not_nan_mask, ocean_mask, latlon_mask])

        # Random indices
        lat_inds, lon_inds = create_random_indices(total_mask, num_timesteps, num_samples_timestep)

    else:
        logger.info('Reas sampling indices from file')
        lat_inds, lon_inds, sort_inds = read_random_indices(random_ind_file)
        num_timesteps = lon_idx.shape[0]
        num_samples_timestep = lon_idx.shape[1]
        num_samples_exact = num_samples_timestep * num_timesteps


    logger.info('Select random profiles')
    profiles = {}

    # lons, lats
    profiles['lat'] = np.ones((num_samples_exact)) * np.nan
    profiles['lon'] = np.ones((num_samples_exact)) * np.nan    
    for j, t in enumerate(timesteps):
        start = j * num_samples_timestep
        end = start + num_samples_timestep
        profiles['lat'][start:end] = lat[lat_inds[j]]
        profiles['lon'][start:end] = lon[lon_inds[j]]

    # Variables
    for i, var in enumerate(variables):
        logger.info(var)
        if var in variables_2D:
            profiles[var] = np.ones((num_samples_exact)) * np.nan
        elif var in variables_3D:
            profiles[var] = np.ones((num_samples_exact, num_levels)) * np.nan

        for j, t in enumerate(timesteps):
            start = j * num_samples_timestep
            end = start + num_samples_timestep
            profiles[var][start:end] = read_var_timestep_latlon(infiles[var], model, var, t, lat_inds[t], lon_inds[t])

    if new_sampling_idx:
        logger.info('Calculate IWV')
        profiles['IWV'] = utils.calc_IWV(profiles['QV'].T, profiles['TEMP'].T, profiles['PRES'].T, height)
        logger.info('Get indices for IWV sorting')
        sort_inds = np.argsort(profiles['IWV'])  
        save_random_indices(random_ind_file, lat_inds, lon_inds, sort_inds)

    #FIXME Add QV_sat, IWV_sat, CRH, IWP, H_tropo, UV, dRH_dz 

    # sort by IWV and save output
    logger.info('Sort and save to files')
    for i, var in enumerate(variables + ['IWV', 'lon', 'lat']):
        outname = filenames.selected_profiles(data_dir, model, run, var, num_samples, time_period, filename_suffix)
        profiles_sorted = profiles[var][sort_inds]
        save_random_profiles(outname, profiles_sorted, var, height)

def select_random_profiles(model, run, num_samples_tot, infiles, outfiles, heightfile, landmaskfile,\
                               variables, lonlatbox, vinterp, data_dir, sample_days, new_sampling_idx, timesteps=None, **kwargs):
    """ Selects a subset of random profiles from (horizontally interpolated) model fields, sorts them by their integrated
    water vapor and saves them.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        num_samples_tot (int): Total number of profiles to select
        infiles (list of str): List of names of input files (each file contains one variable)
        outfiles (list of str): List of names for output files
        heightfile (str): Full path to file containing height vector
        landmaskfile (str): Full path to file containing land mask
        variables (list of str): names of variables (same order as in infiles!!)
        lonlatbox (list of floats): Boundaries of lon-lat box to select [lonmin, lonmax, latmin, latmax]
        vinterp (boolean): True, if data is already interpolated vertically 
        data_dir (str): Path to output directory
        sample_days (str): 'First10', 'Last10' or 'all'
        timesteps (1darray or None): Either a 1darray containing timesteps to use or None 
            (all timesteps are used; default)
    """
    
    logger.info(model)
    logger.info('Config')
    variables_2D = ['OLR', 'OLRC', 'STOA', 'IWV', 'CRH', 'TQI', 'TQC', 'TQG', 'TQS', 'TQR', 'lat', 'lon', 'timestep', 'SST', 'SURF_PRES']
    test_ind = [i for i in range(len(variables)) if variables[i] not in variables_2D][0]
    test_var = variables[test_ind]
    test_filename = infiles[test_ind]
    
    logger.info('Get dimensions')
    # height
    height = read_var(heightfile, model, 'target_height')
    num_levels = height.shape[0]
    if height[0] < height[-1]:
        surface = 0
    else:
        surface = -1
    if model == 'GEOS':
        surface = -2
    
    if new_sampling_idx:
        logger.info('Read lats/lons from file')
        # lon, lat
        with Dataset(test_filename) as ds:
            dimensions = ds.variables[test_var].shape
            test_field = ds.variables[test_var][0].filled(np.nan)
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]

        lat_reg_ind = np.logical_and(lat >= lonlatbox[2], lat <= lonlatbox[3])
        lon_reg_ind = np.logical_and(lon >= lonlatbox[0], lon <= lonlatbox[1])
        
        #lat = lat[lat_reg_ind]
        #lon = lon[lon_reg_ind]

        logger.info('NaN mask')
        if model == 'MPAS' and test_var in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
            nan_mask = np.logical_not(np.isnan(test_field[:, :, surface]))
        else:
            nan_mask = np.logical_not(np.isnan(test_field[surface]))

        # timesteps
        logger.info('Timesteps')
        if timesteps is not None:
            num_timesteps = len(timesteps)
        else:
            if sample_days.lower() == 'all':
                num_timesteps = dimensions[0]
                timesteps = np.arange(num_timesteps)
            elif sample_days.lower() == 'first10':
                num_timesteps = 80
                timesteps = np.arange(0, num_timesteps)
            elif sample_days.lower() == 'last10':
                num_timesteps = 80
                timesteps = np.arange(dimensions[0] - 80, dimensions[0])

        if model in ['MPAS', 'IFS', 'ARPEGE']:
            timesteps = timesteps[:-1]
            num_timesteps = num_timesteps - 1

        num_samples_timestep = int(num_samples_tot / num_timesteps)
        num_samples_tot = num_samples_timestep * num_timesteps

        #return nan_mask

        logger.info('Ocean mask')
        # get ocean_mask
        ocean_mask = np.logical_not(
            np.squeeze(read_var(landmaskfile, model, 'land_mask'))
        )
        ocean_lon = read_var(landmaskfile, model, 'lon')
        ocean_lat = read_var(landmaskfile, model, 'lat')
        ocean_lat_ind = np.where(np.logical_and(ocean_lat >= -30, ocean_lat <= 30))[0]
        #ocean_lat_ind = np.where(np.logical_and(ocean_lat >= lonlatbox[2], ocean_lat <= lonlatbox[3]))[0]
        #ocean_lon_ind = np.where(np.logical_and(ocean_lon >= lonlatbox[0], ocean_lon <= lonlatbox[1]))[0]
        ocean_mask = ocean_mask[ocean_lat_ind].astype(int)
    #    lat_ocean = np.where(ocean_mask)[0]
    #    lon_ocean = np.where(ocean_mask)[1]

        logger.info('Total mask')   
        ocean_nan_mask = np.logical_and(ocean_mask, nan_mask)
        lat_mask = np.tile(lat_reg_ind, (len(lon), 1)).T
        lon_mask = np.tile(lon_reg_ind, (len(lat), 1))
        lonlat_mask = np.logical_and(lon_mask, lat_mask)
        total_mask = np.where(np.logical_and(lonlat_mask, ocean_nan_mask))

        lat_not_masked = total_mask[0]
        lon_not_masked = total_mask[1]

        lat_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)
        lon_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)

        logger.info('Get indices')
        for t in range(num_timesteps):
            r = random.sample(list(zip(lat_not_masked, lon_not_masked)), num_samples_timestep)
            lat_inds[t], lon_inds[t] = zip(*r)

        # save indices
        if sample_days.lower() == 'all':
            nctools.array2D_to_netCDF(
                lat_inds, 'idx', '', (range(num_timesteps), range(num_samples_timestep)),
                ('timestep', 'profile_index'), outfiles[-2], overwrite=True
                    )
            nctools.array2D_to_netCDF(
                lon_inds, 'idx', '', (range(num_timesteps), range(num_samples_timestep)),
                ('timestep', 'profile_index'), outfiles[-3], overwrite=True
                    )
    else:
        logger.info('Read indices from files')
        with Dataset(filename_sort_idx) as ds:
            IWV_sort_idx = ds.variables['idx'][:].filled(np.nan).astype(int)

        with Dataset(filename_lat_idx) as ds:
            lat_idx = ds.variables['idx'][:].filled(np.nan).astype(int)

        with Dataset(filename_lon_idx) as ds:
            lon_idx = ds.variables['idx'][:].filled(np.nan).astype(int)
        
        logger.info('Timesteps')
        num_timesteps = lon_idx.shape[0]
        num_samples_timestep = lon_idx.shape[1]
        num_samples_tot = num_samples_timestep * num_timesteps
          
    profiles = {}
    profiles_sorted = {}
    #latlon = np.meshgrid(lat, lon)
    profiles['lat'] = np.ones((num_samples_tot)) * np.nan
    profiles['lon'] = np.ones((num_samples_tot)) * np.nan
    
    # select lats, lons
    for j, t in enumerate(timesteps):
        start = j * num_samples_timestep
        end = start + num_samples_timestep
        profiles['lat'][start:end] = lat[lat_inds[j]]
        profiles['lon'][start:end] = lon[lon_inds[j]]
    #return profiles, ocean_mask, nan_mask, total_mask, lat_inds, lon_inds, num_timesteps, num_samples_timestep, lon, lat
    logger.info('Read variables from files')
    for i, var in enumerate(variables):
        logger.info(var)
        if var in variables_2D:
            profiles[var] = np.ones((num_samples_tot)) * np.nan
            profiles_sorted[var] = np.ones((num_samples_tot)) * np.nan
            for j, t in enumerate(timesteps):
                start = j * num_samples_timestep
                end = start + num_samples_timestep
                with Dataset(infiles[i]) as ds:
                    if var == 'SST':
                        t_eff = t // 2
                    else:
                        t_eff = t
  
                    if model == 'NICAM' and var != 'SST':
                        profiles[var][start:end] = ds.variables[var][t_eff][0, lat_inds[j], lon_inds[j]].filled(np.nan)
                    elif model == 'IFS' and var == 'SURF_PRES':
                        profiles[var][start:end] = np.exp(ds.variables[var][t_eff][0, lat_inds[j], lon_inds[j]].filled(np.nan))
                    else:
                        profiles[var][start:end] = ds.variables[var][t_eff][lat_inds[j], lon_inds[j]].filled(np.nan)
        else:
            profiles[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
            profiles_sorted[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
            for j, t in enumerate(timesteps):
                start = j * num_samples_timestep
                end = start + num_samples_timestep
                with Dataset(infiles[i]) as ds:
                    if model == 'MPAS' and var in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
                        profiles[var][:, start:end] = ds.variables[var][t][lat_inds[j], lon_inds[j], :].filled(np.nan).transpose([1, 0])
                    else:
                        if model == 'SAM' and var == 'PRES':
                            pres_mean = np.expand_dims(ds.variables['p'][:], 1) * 1e2
                            pres_pert = ds.variables['PP'][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)
                            profiles[var][:, start:end] = pres_mean + pres_pert
                        else:
                            profiles[var][:, start:end] = ds.variables[var][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)
                        
        if model == 'SAM' and var in ['QV', 'QI', 'QC']:
            profiles[var] *= 1e-3
            
    logger.info('Calculate IWV')
    profiles['IWV'] = utils.calc_IWV(profiles['QV'], profiles['TEMP'], profiles['PRES'], height)
        
    if new_sampling_idx:
        # get indices to sort by IWV
        logger.info('Get indices for IWV sorting')
        IWV_sort_idx = np.argsort(profiles['IWV'])  
        
        # save indices
        if sample_days.lower() == 'all':
            logger.info('Save indices for IWV sorting')
            nctools.vector_to_netCDF(
                IWV_sort_idx, 'idx', '', range(num_samples_timestep * num_timesteps), 'profile_index', outfiles[-1], overwrite=True
                    )
    
    profiles['QV_sat'] = utils.rel_hum2spec_hum(
        np.ones(profiles['TEMP'].shape), 
        profiles['TEMP'], 
        profiles['PRES'], 
        phase='mixed'
    )
    profiles['IWV_sat'] = utils.calc_IWV(
        profiles['QV_sat'],
        profiles['TEMP'],
        profiles['PRES'],
        height
    )
    profiles['CRH'] = profiles['IWV'] / profiles['IWV_sat']
     
    # sort by IWV and save output
    logger.info('Save to files')
    for i, var in enumerate(variables + ['IWV', 'CRH', 'lon', 'lat']):
        if var in variables_2D:
            profiles_sorted[var] = profiles[var][IWV_sort_idx]
            nctools.vector_to_netCDF(
                profiles_sorted[var], var, '', range(num_samples_timestep * num_timesteps), 'profile_index', outfiles[i], overwrite=True
            )
        else:
            profiles_sorted[var] = profiles[var][:, IWV_sort_idx]
            nctools.array2D_to_netCDF(
                profiles_sorted[var], var, '', (height, range(num_samples_timestep * num_timesteps)), ('height', 'profile_index'), outfiles[i], overwrite=True
            )
            
def select_random_profiles_cloudsat(model, run, num_samples_tot, infiles, outfiles, heightfile, landmaskfile,\
                               variables, lonlatbox, vinterp, data_dir, sample_days, timesteps=None, **kwargs):
    """ Selects a subset of random profiles from (horizontally interpolated) model fields only from longitude regions with a local time corresponding to the cloudsat overpass time, sorts them by their integrated water vapor and saves them.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        num_samples_tot (int): Total number of profiles to select
        infiles (list of str): List of names of input files (each file contains one variable)
        outfiles (list of str): List of names for output files
        heightfile (str): Full path to file containing height vector
        landmaskfile (str): Full path to file containing land mask
        variables (list of str): names of variables (same order as in infiles!!)
        lonlatbox (list of floats): Boundaries of lon-lat box to select [lonmin, lonmax, latmin, latmax]
        vinterp (boolean): True, if data is already interpolated vertically 
        data_dir (str): Path to output directory
        timesteps (1darray or None): Either a 1darray containing timesteps to use or None 
            (all timesteps are used; default)
    """
    
    logger.info(model)
    logger.info('Config')
    variables_2D = ['OLR', 'OLRC', 'STOA', 'IWV', 'CRH', 'TQI', 'TQC', 'TQR', 'TQS', 'TQG', 'lat', 'lon', 'timestep']
    test_ind = [i for i in range(len(variables)) if variables[i] not in variables_2D][0]
    test_var = variables[test_ind]
    test_filename = infiles[test_ind]
    overpass_time = [9900, 53100] # 2:45 and 14:45
    time_delta = 19 * 60 # in seconds
    
    logger.info('Get dimensions')
    # height
    height = read_var(heightfile, model, 'target_height')
    num_levels = height.shape[0]
    if height[0] < height[-1]:
        surface = 0
    else:
        surface = -1
    if model == 'GEOS':
        surface = -2

    # timesteps
    logger.info('Timesteps')
    if timesteps is not None:
        time_array = get_time_array(model)[timesteps]
        num_timesteps = len(timesteps)
    else:
        time_array = get_time_array(model)
        timesteps = np.arange(len(time_array))
        num_timesteps = len(timesteps)
    num_samples_timestep = int(num_samples_tot / num_timesteps)

    logger.info('Read lats/lons from file')
    # lon, lat
    with Dataset(test_filename) as ds:
        dimensions = ds.variables[test_var].shape
        test_field = ds.variables[test_var][0].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)

    lon_reg_ind = np.where(np.logical_and(lon >= lonlatbox[0], lon <= lonlatbox[1]))[0]
    lon_reg = lon[lon_reg_ind]
    lat_reg_ind = np.where(np.logical_and(lat >= lonlatbox[2], lat <= lonlatbox[3]))[0]
    lat_reg = lat[lat_reg_ind]

    def lst_sec(utc, lon):
        timediff = pd.Timedelta(hours=np.deg2rad(lon)/np.pi*12)
        lst = utc + timediff
        lst_sec = (lst.hour * 60 + lst.minute) * 60 + lst.second
        return lst_sec
    vlst = np.vectorize(lst_sec)
    lst = np.empty((num_timesteps, len(lat_reg), len(lon_reg)), dtype=pd.Timestamp)
    lst_mask = []

    for t in range(num_timesteps):
        lst_lon = vlst(time_array[t], lon_reg)
        lst[t] = np.repeat(lst_lon[np.newaxis, :], len(lat_reg), axis=0)
        # Gitterzellen mit lsts, diezur CloudSat overpass time passen
        lst_mask.append(np.logical_or(np.logical_and(overpass_time[0] - time_delta <= lst[t], lst[t] <= overpass_time[0] + time_delta), np.logical_and(overpass_time[1] - time_delta <= lst[t], lst[t] <= overpass_time[1] + time_delta)))


    logger.info('NaN mask')
    nan_mask = []
    if model == 'MPAS' and test_var in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
        for t in range(num_timesteps):
            nan_mask = np.logical_not(np.isnan(test_field[lat_reg_ind][:, lon_reg_ind][:, :, surface]))
    else:
        for t in range(num_timesteps):
            nan_mask = np.logical_not(np.isnan(test_field[surface][lat_reg_ind][:, lon_reg_ind]))

    logger.info('Ocean mask')
    # get ocean_mask
    ocean_mask = np.logical_not(
        np.squeeze(read_var(landmaskfile, model, 'land_mask'))
    )

    ocean_lon = read_var(landmaskfile, model, 'lon')
    ocean_lat = read_var(landmaskfile, model, 'lat')
    ocean_lat_ind = np.where(np.logical_and(ocean_lat >= lonlatbox[2], ocean_lat <= lonlatbox[3]))[0]
    ocean_lon = read_var(landmaskfile, model, 'lon')
    ocean_lon_ind = np.where(np.logical_and(ocean_lon >= lonlatbox[0], ocean_lon <= lonlatbox[1]))[0]
    ocean_mask = ocean_mask[ocean_lat_ind][:, ocean_lon_ind].astype(int)

    logger.info('Total mask')
    total_mask = []
    lat_not_masked = []
    lon_not_masked = []
    for t in range(num_timesteps):
        total_mask.append(np.where(np.logical_and(np.logical_and(ocean_mask, nan_mask), lst_mask[t])))
        lat_not_masked.append(total_mask[t][0])
        lon_not_masked.append(total_mask[t][1])

    lat_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)
    lon_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)

    logger.info('Get indices')
    for t in range(num_timesteps):
        r = random.sample(list(zip(lat_not_masked[t], lon_not_masked[t])), num_samples_timestep)
        lat_inds[t], lon_inds[t] = zip(*r)

    profiles = {}
    profiles_sorted = {}
    #latlon = np.meshgrid(lat, lon)
    profiles['lat'] = np.ones((num_samples_tot)) * np.nan
    profiles['lon'] = np.ones((num_samples_tot)) * np.nan

    # select lats, lons
    for j, t in enumerate(timesteps):
        start = j * num_samples_timestep
        end = start + num_samples_timestep
        profiles['lat'][start:end] = lat[lat_inds[j]]
        profiles['lon'][start:end] = lon[lon_inds[j]]
    #return profiles, ocean_mask, nan_mask, total_mask, lat_inds, lon_inds, num_timesteps, num_samples_timestep, lon, lat
    logger.info('Read variables from files')
    for i, var in enumerate(variables):
        logger.info(var)
        if var in variables_2D:
            profiles[var] = np.ones((num_samples_tot)) * np.nan
            profiles_sorted[var] = np.ones((num_samples_tot)) * np.nan
            for j, t in enumerate(timesteps):
                start = j * num_samples_timestep
                end = start + num_samples_timestep
                with Dataset(infiles[i]) as ds:
                    if model == 'NICAM':
                        profiles[var][start:end] = ds.variables[var][t][0, lat_inds[j], lon_inds[j]].filled(np.nan)
                    else:
                        profiles[var][start:end] = ds.variables[var][t][lat_inds[j], lon_inds[j]].filled(np.nan)
        else:
            profiles[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
            profiles_sorted[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
            for j, t in enumerate(timesteps):
                start = j * num_samples_timestep
                end = start + num_samples_timestep
                with Dataset(infiles[i]) as ds:
                    if model == 'MPAS' and var in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
                        profiles[var][:, start:end] = ds.variables[var][t][lat_inds[j], lon_inds[j], :].filled(np.nan).transpose([1, 0])
                    else:
                        if model == 'SAM' and var == 'PRES':
                            pres_mean = np.expand_dims(ds.variables['p'][:], 1) * 1e2
                            pres_pert = ds.variables['PP'][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)
                            profiles[var][:, start:end] = pres_mean + pres_pert
                        else:
                            profiles[var][:, start:end] = ds.variables[var][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)

        if model == 'SAM' and var in ['QV', 'QI', 'QC']:
            profiles[var] *= 1e-3
            
    logger.info('Calculate IWV and CRH')
    # calculate IWV
    profiles['IWV'] = utils.calc_IWV(profiles['QV'], profiles['TEMP'], profiles['PRES'], height)
    # get indices to sort by IWV
    IWV_sort_idx = np.argsort(profiles['IWV'])
    
    profiles['QV_sat'] = utils.rel_hum2spec_hum(
        np.ones(profiles['TEMP'].shape), 
        profiles['TEMP'], 
        profiles['PRES'], 
        phase='mixed'
    )
    profiles['IWV_sat'] = utils.calc_IWV(
        profiles['QV_sat'],
        profiles['TEMP'],
        profiles['PRES'],
        height
    )
    profiles['CRH'] = profiles['IWV'] / profiles['IWV_sat']
            
    # sort by IWV and save output
    logger.info('Save to files')
    for i, var in enumerate(variables + ['IWV', 'CRH', 'lon', 'lat']):
        if var in variables_2D:
            profiles_sorted[var] = profiles[var][IWV_sort_idx]
            nctools.vector_to_netCDF(
                profiles_sorted[var], var, '', range(num_samples_timestep * num_timesteps), 'profile_index', outfiles[i], overwrite=True
            )
        else:
            profiles_sorted[var] = profiles[var][:, IWV_sort_idx]
            nctools.array2D_to_netCDF(
                profiles_sorted[var], var, '', (height, range(num_samples_timestep * num_timesteps)), ('height', 'profile_index'), outfiles[i], overwrite=True
            )

def advection_for_random_profiles(model, run, time_period, num_samples, data_dir, **kwargs):
    """ Calculate vertical and horizontal advection of QV and RH for randomly selected profiles.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        num_samples (num): number of randomly selected profiles
        data_dir (str): Path to output directory
    """
    #TODO Include wrong masks and set to NaN
    logger.info(f'{model}-{run}')
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")

    logger.info('Input and output variables and filenames')
    input_variables = ['U', 'V', 'W', 'QV', 'RH', 'TEMP', 'PRES'] # 'QC', 'QI'
    output_variables = ['A_QV_h', 'A_RH_h', 'A_QV_v', 'A_RH_v', 'DRH_Dt_h', 'DRH_Dt_v', 'dRH_dx', 'dRH_dy'] # 'A_T_h', 'A_T_v', 'DT_Dt_h', 'DT_Dt_v', 'A_QC_h', 'A_QC_v', 'A_QI_h', 'A_QI_v'
    if model in ['IFS', 'FV3', 'ARPEGE', 'GEOS', 'ERA5']:
        filename = '{}-{}_{}_hinterp_vinterp_merged_{}-{}.nc'
    else:
        filename = '{}-{}_{}_hinterp_merged_{}-{}.nc'
    filename_out = '{}-{}_{}_sample_{}_{}-{}.nc'
    filename_lat_idx = filename_out.format(model, run, 'ind_lat', num_samples, start_date, end_date)
    filename_lat_idx = os.path.join(data_dir, model, 'random_samples', filename_lat_idx)
    filename_lon_idx = filename_out.format(model, run, 'ind_lon', num_samples, start_date, end_date)
    filename_lon_idx = os.path.join(data_dir, model, 'random_samples', filename_lon_idx)
    filename_sort_idx = filename_out.format(model, run, 'sort_ind', num_samples, start_date, end_date)
    filename_sort_idx = os.path.join(data_dir, model, 'random_samples', filename_sort_idx)
    testfile = os.path.join(data_dir, model, filename.format(model, run, input_variables[0], start_date, end_date))
    heightfile = filelists.get_path2targetheightfile(model, data_dir)
    filenames = {}
    outnames = {}
    
    for var in input_variables:
        if model in ['ICON', 'MPAS'] and var == 'W':
            filenames[var] = os.path.join(data_dir, model,
                                          filename.format(model, run, 'WHL', start_date, end_date)
                                         )
        else:
            filenames[var] = os.path.join(data_dir, model,
                                          filename.format(model, run, var, start_date, end_date)
                                         )
    for var in output_variables:
        outnames[var] = os.path.join(data_dir, model, 'random_samples',
                                     filename_out.format(model, run, var, num_samples, start_date, end_date)
                                     )

    logger.info('Read lats and lons')
    with Dataset(testfile) as ds:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        if model == 'MPAS':
            num_levels = ds.variables[input_variables[0]].shape[3]
        else:
            num_levels = ds.variables[input_variables[0]].shape[1]
        
    with Dataset(heightfile) as ds:
        height = ds.variables['target_height'][:]

    logger.info('Read indices from files')
    with Dataset(filename_sort_idx) as ds:
        sort_idx = ds.variables['idx'][:].filled(np.nan).astype(int)

    with Dataset(filename_lat_idx) as ds:
        lat_idx = ds.variables['idx'][:].filled(np.nan).astype(int)

    with Dataset(filename_lon_idx) as ds:
        lon_idx = ds.variables['idx'][:].filled(np.nan).astype(int)

    num_timesteps = lon_idx.shape[0]
    num_samples_timestep = lon_idx.shape[1]
    num_samples_tot = num_samples_timestep * num_timesteps

    profiles = {}
    profiles['lat'] = np.ones((num_samples_tot)) * np.nan

    logger.info('Read latitudes from file')
    for t in range(num_timesteps):
        start = t * num_samples_timestep
        end = start + num_samples_timestep
        profiles['lat'][start:end] = lat[lat_idx[t]]

    logger.info('Calculate distance to next grid points')
    dx = 0.1
    dy = 0.1
    profiles['dx'] = 111000 * dx * np.cos(np.deg2rad(profiles['lat']))
    profiles['dy'] = 111000 * dy        

    logger.info('Allocate array for selected profiles')
    for var in list(set(input_variables) | set(output_variables)):
        profiles[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
        #profiles_sorted[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan

    logger.info('Read variables from files')
    for var in input_variables:
        for t in range(num_timesteps):
            start = t * num_samples_timestep
            end = start + num_samples_timestep
            with Dataset(filenames[var]) as ds:
                if model == 'SAM' and var == 'PRES':
                    pres_mean = np.expand_dims(ds.variables['p'][:], 1) * 1e2
                    pres_pert = ds.variables['PP'][t][:, lat_idx[t], lon_idx[t]].filled(np.nan)
                    profiles[var][:, start:end] = pres_mean + pres_pert
                elif model == 'MPAS' and var in ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'U', 'V']:
                    profiles[var][:, start:end] = ds.variables[var][t][lat_idx[t], lon_idx[t], :].filled(np.nan).T
                else:
                    profiles[var][:, start:end] = ds.variables[var][t][:, lat_idx[t], lon_idx[t]].filled(np.nan)
    if model == 'SAM':
        profiles['QV'] *= 1e-3

    if 'QC' in input_variables:
        logger.info('Set condensate to zero in unsaturated cases')
        unsat = profiles['RH'] < 1
        profiles['QC'][unsat] = 0.
        profiles['QI'][unsat] = 0.
        profiles['A_QC_v'] = -profiles['W'] * np.gradient(profiles['QC'], height, axis=0)
        profiles['A_QI_v'] = -profiles['W'] * np.gradient(profiles['QI'], height, axis=0)
    
    logger.info('Calculate vertical transport terms')
    R = typhon.constants.gas_constant_water_vapor
    cp = typhon.constants.isobaric_mass_heat_capacity
    L = typhon.constants.heat_of_vaporization
    rho = profiles['PRES'] / (profiles['TEMP'] * R) 
    #rho = utils.calc_density_moist_air(
    #    profiles['PRES'],
    #    profiles['TEMP'], 
    #    profiles['QV']
    #)
    
    profiles['A_RH_v'] = -profiles['W'] * np.gradient(profiles['RH'], height, axis=0)
    profiles['A_QV_v'] = -profiles['W'] * np.gradient(profiles['QV'], height, axis=0)
    profiles['DRH_Dt_v'] = utils.drh_dt(
            profiles['TEMP'], 
            profiles['PRES'], 
            profiles['RH'],
            profiles['W']  
        )
    
    if 'A_T_v' in output_variables:
        profiles['A_T_v'] = -profiles['W'] * np.gradient(profiles['TEMP'], height, axis=0)
    if 'DT_Dt_v' in output_variables:
        profiles['DT_Dt_v'] = -typhon.constants.g / cp * profiles['W']
    
    logger.info('Read QV and RH from files and calculate horizontal transport terms')

    for t in range(num_timesteps):

        start = t * num_samples_timestep
        end = start + num_samples_timestep
        lon_idx_next = lon_idx[t] + 1
        lon_idx_next[lon_idx_next == len(lon)] = 0
        lon_idx_before = lon_idx[t] - 1
        lon_idx_before[lon_idx_before < 0] = len(lon) - 1
        with Dataset(filenames['QV']) as ds:
            if model == 'MPAS':
                dQdx = (ds.variables['QV'][t][lat_idx[t], lon_idx_next, :]
                        - ds.variables['QV'][t][lat_idx[t], lon_idx_before, :]).T / profiles['dx'][start:end]
                dQdy = (ds.variables['QV'][t][lat_idx[t] + 1, lon_idx[t], :]
                        - ds.variables['QV'][t][lat_idx[t] - 1, lon_idx[t], :]).T / profiles['dy']
                
            else:
                dQdx = (ds.variables['QV'][t][:, lat_idx[t], lon_idx_next]
                        - ds.variables['QV'][t][:, lat_idx[t], lon_idx_before]) / profiles['dx'][start:end]
                dQdy = (ds.variables['QV'][t][:, lat_idx[t] + 1, lon_idx[t]]
                        - ds.variables['QV'][t][:, lat_idx[t] - 1, lon_idx[t]]) / profiles['dy']

            profiles['A_QV_h'][:, start:end] = -1 * (profiles['U'][:, start:end] * dQdx + profiles['V'][:, start:end] * dQdy)
            
        with Dataset(filenames['RH']) as ds:
            RH_next_lon = ds.variables['RH'][t][:, lat_idx[t], lon_idx_next]
            RH_before_lon = ds.variables['RH'][t][:, lat_idx[t], lon_idx_before]
            RH_next_lat = ds.variables['RH'][t][:, lat_idx[t] + 1, lon_idx[t]]
            RH_before_lat = ds.variables['RH'][t][:, lat_idx[t] - 1, lon_idx[t]]
            dRHdx = (RH_next_lon
                     - RH_before_lon) / profiles['dx'][start:end]
            dRHdy = (RH_next_lat
                    - RH_before_lat) / profiles['dy']

            profiles['A_RH_h'][:, start:end] = -1 * (profiles['U'][:, start:end] * dRHdx + profiles['V'][:, start:end] * dRHdy)
            profiles['dRH_dx'][:, start:end] = dRHdx
            profiles['dRH_dy'][:, start:end] = dRHdy
            
        with Dataset(filenames['TEMP']) as ds:
            if model == 'MPAS':
                dTdx = (ds.variables['TEMP'][t][lat_idx[t], lon_idx_next, :]
                         - ds.variables['TEMP'][t][lat_idx[t], lon_idx_before, :]).T / profiles['dx'][start:end]
                dTdy = (ds.variables['TEMP'][t][lat_idx[t] + 1, lon_idx[t], :]
                        - ds.variables['TEMP'][t][lat_idx[t] - 1, lon_idx[t], :]).T / profiles['dy']
            else:
                dTdx = (ds.variables['TEMP'][t][:, lat_idx[t], lon_idx_next]
                         - ds.variables['TEMP'][t][:, lat_idx[t], lon_idx_before]) / profiles['dx'][start:end]
                dTdy = (ds.variables['TEMP'][t][:, lat_idx[t] + 1, lon_idx[t]]
                        - ds.variables['TEMP'][t][:, lat_idx[t] - 1, lon_idx[t]]) / profiles['dy']
            
        with Dataset(filenames['PRES']) as ds:
            if model == 'SAM':
                pname = 'PP'
            else:
                pname = 'PRES'
                
            if model == 'MPAS':
                dPdx = (ds.variables[pname][t][lat_idx[t], lon_idx_next, :]
                        - ds.variables[pname][t][lat_idx[t], lon_idx_before, :]).T / profiles['dx'][start:end]
                dPdy = (ds.variables[pname][t][lat_idx[t] + 1, lon_idx[t], :]
                        - ds.variables[pname][t][lat_idx[t] - 1, lon_idx[t], :]).T / profiles['dy']

            else:
                dPdx = (ds.variables[pname][t][:, lat_idx[t], lon_idx_next]
                        - ds.variables[pname][t][:, lat_idx[t], lon_idx_before]) / profiles['dx'][start:end]
                dPdy = (ds.variables[pname][t][:, lat_idx[t] + 1, lon_idx[t]]
                        - ds.variables[pname][t][:, lat_idx[t] - 1, lon_idx[t]]) / profiles['dy']

            dPdt = dPdx * profiles['U'][:, start:end] + dPdy * profiles['V'][:, start:end]
            dTdP = -1 / rho[:, start:end] / cp

            profiles['DRH_Dt_h'][:, start:end] = (profiles['RH'][:, start:end] / profiles['PRES'][:, start:end] - profiles['RH'][:, start:end] * L / R / (profiles['TEMP'][:, start:end] ** 2) * dTdP) * dPdt
            
            if 'A_T_h' in output_variables:
                profiles['DT_Dt_h'][:, start:end] = dTdP * dPdt
                profiles['A_T_h'][:, start:end] = -(dTdx * profiles['U'][:, start:end] + dTdy * profiles['V'][:, start:end])
            
        if 'QI' in input_variables:
            with Dataset(filenames['QI']) as ds:
                if model == 'MPAS':
                    QI_next_lon = ds.variables['QI'][t][lat_idx[t], lon_idx_next, :].T
                    QI_before_lon = ds.variables['QI'][t][lat_idx[t], lon_idx_before, :].T
                    QI_next_lat = ds.variables['QI'][t][lat_idx[t] + 1, lon_idx[t], :].T
                    QI_before_lat = ds.variables['QI'][t][lat_idx[t] - 1, lon_idx[t], :].T
                    QI_next_lon[RH_next_lon < 1.] = 0.
                    QI_before_lon[RH_before_lon < 1.] = 0.
                    QI_next_lat[RH_next_lat < 1.] = 0.
                    QI_before_lat[RH_before_lat < 1.] = 0.
                    dQIdx = (QI_next_lon
                            - QI_before_lon) / profiles['dx'][start:end]
                    dQIdy = (QI_next_lat
                            - QI_before_lat) / profiles['dy']

                else:
                    QI_next_lon = ds.variables['QI'][t][:, lat_idx[t], lon_idx_next]
                    QI_before_lon = ds.variables['QI'][t][:, lat_idx[t], lon_idx_before]
                    QI_next_lat = ds.variables['QI'][t][:, lat_idx[t] + 1, lon_idx[t]]
                    QI_before_lat = ds.variables['QI'][t][:, lat_idx[t] - 1, lon_idx[t]]
                    QI_next_lon[RH_next_lon < 1.] = 0.
                    QI_before_lon[RH_before_lon < 1.] = 0.
                    QI_next_lat[RH_next_lat < 1.] = 0.
                    QI_before_lat[RH_before_lat < 1.] = 0.
                    dQIdx = (QI_next_lon
                            - QI_before_lon) / profiles['dx'][start:end]
                    dQIdy = (QI_next_lat
                            - QI_before_lat) / profiles['dy']

                profiles['A_QI_h'][:, start:end] = -1 * (profiles['U'][:, start:end] * dQIdx + profiles['V'][:, start:end] * dQIdy)
                
        if 'QC' in input_variables:
            with Dataset(filenames['QC']) as ds:
                if model == 'MPAS':
                    QC_next_lon = ds.variables['QC'][t][lat_idx[t], lon_idx_next, :]
                    QC_before_lon = ds.variables['QC'][t][lat_idx[t], lon_idx_before, :]
                    QC_next_lat = ds.variables['QC'][t][lat_idx[t] + 1, lon_idx[t], :]
                    QC_before_lat = ds.variables['QC'][t][lat_idx[t] - 1, lon_idx[t], :]
                    QC_next_lon[RH_next_lon < 1.] = 0.
                    QC_before_lon[RH_before_lon < 1.] = 0.
                    QC_next_lat[RH_next_lat < 1.] = 0.
                    QC_before_lat[RH_before_lat < 1.] = 0.
                    dQCdx = (QC_next_lon
                            - QC_before_lon) / profiles['dx'][start:end]
                    dQCdy = (QC_next_lat
                            - QC_before_lat) / profiles['dy']

                else:
                    QC_next_lon = ds.variables['QC'][t][:, lat_idx[t], lon_idx_next]
                    QC_before_lon = ds.variables['QC'][t][:, lat_idx[t], lon_idx_before]
                    QC_next_lat = ds.variables['QC'][t][:, lat_idx[t] + 1, lon_idx[t]]
                    QC_before_lat = ds.variables['QC'][t][:, lat_idx[t] - 1, lon_idx[t]]
                    QC_next_lon[RH_next_lon < 1.] = 0.
                    QC_before_lon[RH_before_lon < 1.] = 0.
                    QC_next_lat[RH_next_lat < 1.] = 0.
                    QC_before_lat[RH_before_lat < 1.] = 0.
                    dQCdx = (QC_next_lon
                            - QC_before_lon) / profiles['dx'][start:end]
                    dQCdy = (QC_next_lat
                            - QC_before_lat) / profiles['dy']

                profiles['A_QC_h'][:, start:end] = -1 * (profiles['U'][:, start:end] * dQCdx + profiles['V'][:, start:end] * dQCdy)

    if model == 'SAM':
         profiles['A_QV_h'] *= 1e-3
            
    logger.info('Save results to files')
    for var in output_variables:
        profiles_sorted = profiles[var][:, sort_idx]
        nctools.array2D_to_netCDF(
            profiles_sorted, var, '', (height, range(num_samples_tot)), ('height', 'profile_index'), outnames[var], overwrite=True
                )
    
def average_random_profiles(model, run, time_period, variables, num_samples, sample_days, data_dir, log_average, **kwargs):
    """ Average randomly selected profiles in IWV percentile bins and IWV bins, separately for different
    ocean basins. Output is saved as .pkl files.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        variables (list of str): names of variables
        num_samples (num): number of randomly selected profiles
        sample_days (str): 'all', 'first10' or 'last10'
    """

    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'QS', 'QG', 'QR', 'RH', 'W', 'DRH_Dt_v', 'DRH_Dt_h', 'DRH_Dt_c', 'A_RH_v', 'A_QV_v', 'A_RH_h', 'A_QV_h', 'U', 'V', 'UV', 'dRH_dx', 'dRH_dy', 'dRH_dz']
    variables_2D = ['OLR', 'IWV', 'STOA', 'OLRC', 'STOAC', 'H_tropo', 'IWP', 'TQI', 'TQC', 'TQG', 'TQS', 'TQR', 'SST', 'SURF_PRES']
    extra_variables = ['QV', 'dRH_dz']
    datapath = f'{data_dir}/{model}/random_samples/'
    filenames = '{}-{}_{}_sample_{}_{}-{}{}{}.nc'
    perc_values = np.arange(2., 100.5, 2.0)
    num_percs = len(perc_values)
    iwv_bin_bnds = np.arange(0, 101, 1)
    ic_thres = {
        'QI': 0.001 * 1e-6,
        'QC': 0.001 * 1e-6
    }
    stats = ['mean', 'std', 'median', 'min', 'max', 'quart25', 'quart75']
    if sample_days == 'all':
        sample_days_str = ''
    else:
        sample_days_str = '_'+sample_days
    if log_average:
        log_average_str = '_log'
    else:
        log_average_str = ''
    
    logger.info('Read data from files')
    filename = filenames.format(model, run, variables_3D[0], num_samples, start_date, end_date, sample_days_str, '')
    filename = os.path.join(datapath, filename)
    with(Dataset(filename)) as ds:
        height = ds.variables['height'][:].filled(np.nan)
    num_levels = len(height)
    
    bins = np.arange(1, len(iwv_bin_bnds)) 
    bin_count = np.zeros(len(iwv_bin_bnds) - 1)
    
    profiles_sorted = {}
    for var in variables+['lon', 'lat']:
        logger.info(var)
        filename = filenames.format(model, run, var, num_samples, start_date, end_date, sample_days_str, '')
        filename = os.path.join(datapath, filename)
        with(Dataset(filename)) as ds:
            profiles_sorted[var] = ds.variables[var][:].filled(np.nan)
    num_profiles = len(profiles_sorted[var])
            
    logger.info('Calc additional variables: IWP, H_tropo etc.')
    # RH tendency 
    
    if 'IWP' in extra_variables:
        profiles_sorted['IWP'] = utils.calc_IWP(
            profiles_sorted['QI'], 
            profiles_sorted['TEMP'], 
            profiles_sorted['PRES'], 
            profiles_sorted['QV'], 
            height
        )
        outname = filenames.format(model, run, 'IWP', num_samples, start_date, end_date, sample_days_str, '')
        nctools.vector_to_netCDF(
            profiles_sorted['IWP'], 'IWP', '', range(num_samples), 'profile_index', outname, overwrite=True
        )
        
    if 'H_tropo' in extra_variables:    
        profiles_sorted['H_tropo'] = utils.tropopause_height(
            profiles_sorted['TEMP'], 
            height
        )
        
    if 'UV' in extra_variables:
        profiles_sorted['UV'] = np.sqrt(profiles_sorted['U'] ** 2 + profiles_sorted['V'] ** 2)
    
    if 'dRH_dz' in extra_variables:
        profiles_sorted['dRH_dz'] = np.gradient(profiles_sorted['RH'], height, axis=0)
    # create mask to exclude profiles with no given longitude       
    nan_mask = np.isnan(profiles_sorted['lon'])
    # create mask to exclude profiles with unrealistic advection values
    tropo = np.where(height < 20000)
    if model == 'SAM':
        thres_a_qv = 4e-3
    else:
        thres_a_qv = 3e-5
    if 'A_QV_h' in variables:
        #wrong_mask_1 = np.any(np.abs(profiles_sorted['A_QV_h'][tropo]) > thres_a_qv, axis=0) 
        wrong_mask_1 = np.logical_or(np.any(np.abs(profiles_sorted['A_QV_h'][tropo]) > thres_a_qv, axis=0),
                                     np.any(np.abs(profiles_sorted['A_QV_h'] / profiles_sorted['QV']) > 0.01, axis=0))

        nan_mask = np.logical_or(nan_mask, wrong_mask_1).astype(int)
        
    if 'DRH_Dt_h' in variables:
        wrong_mask_2 = np.logical_or(np.any(profiles_sorted['DRH_Dt_h'][tropo] < -0.001, axis=0), 
                                     np.any(profiles_sorted['DRH_Dt_h'][tropo] > 0.001, axis=0))
    
        nan_mask = np.logical_or(nan_mask, wrong_mask_2).astype(int)
    
    # set masked profiles to nan
    for var in variables+['lon', 'lat']+extra_variables:
        if var in variables_3D:
            profiles_sorted[var][:, np.where(nan_mask)] = np.nan
        else:
            profiles_sorted[var][np.where(nan_mask)] = np.nan

    perc = {s: {} for s in stats+['percentiles']}
    binned = {s: {} for s in stats+['count']}
    
    logger.info('Binning to percentile bins')
    perc['percentiles'] = np.asarray([np.percentile(profiles_sorted['IWV'], p) for p in perc_values])
    splitted_array = {}
    for var in variables+extra_variables:
        logger.info(var)
        splitted_array[var] = {}
        if var in variables_2D: 
            axis=0
            splitted_array[var] = np.array_split(profiles_sorted[var], num_percs, axis=axis)
            
        else:
            axis=1
            splitted_array[var] = np.array_split(profiles_sorted[var], num_percs, axis=axis)
        
        if var in ['PRES', 'QV'] and log_average == True:
            # Average pressure in log-space
            perc['mean'][var] = np.asarray([np.exp(np.nanmean(np.log(a), axis=axis)) for a in splitted_array[var]])
        else:
            perc['mean'][var] = np.asarray([np.nanmean(a, axis=axis) for a in splitted_array[var]])
        perc['std'][var] = np.asarray([np.nanstd(a, axis=axis) for a in splitted_array[var]])
        perc['min'][var] = np.asarray([np.nanmin(a, axis=axis) for a in splitted_array[var]])
        perc['max'][var] = np.asarray([np.nanmax(a, axis=axis) for a in splitted_array[var]])
        perc['median'][var] = np.asarray([np.nanmedian(a, axis=axis) for a in splitted_array[var]])
        perc['quart25'][var] = np.asarray([np.nanpercentile(a, 25, axis=axis) for a in splitted_array[var]])
        perc['quart75'][var] = np.asarray([np.nanpercentile(a, 75, axis=axis) for a in splitted_array[var]])


    # determine in-cloud ice/water content and cloud fraction in each bin
    for condensate, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
        perc['mean'][content] = np.asarray(
            [np.ma.average(a, weights=(a > ic_thres[condensate]), axis=1).filled(0) for a in splitted_array[condensate]]
        )
        perc['mean'][fraction] = np.asarray(
            [np.sum(a * (a > ic_thres[condensate]), axis=1) / a.shape[1] for a in splitted_array[condensate]]
        )

    logger.info('Binning to IWV bins')
    bin_idx = np.digitize(profiles_sorted['IWV'], iwv_bin_bnds)
    binned['count'] = np.asarray([len(np.where(bin_idx == bi)[0]) for bi in bins])
    binned_profiles = {}
    for var in variables+extra_variables:
        logger.info(var)
        if var in variables_2D:
            binned_profiles[var] = [profiles_sorted[var][bin_idx == bi] for bi in bins]
            axis=0
        else:
            binned_profiles[var] = [profiles_sorted[var][:, bin_idx == bi] for bi in bins]
            axis=1

        if var in ['PRES', 'QV']:
            binned['mean'][var] = np.asarray([np.exp(np.nanmean(np.log(p), axis=axis)) for p in binned_profiles[var]])
        else:
            binned['mean'][var] = np.asarray([np.nanmean(p, axis=axis) for p in binned_profiles[var]])
        binned['std'][var] = np.asarray([np.nanstd(p, axis=axis) for p in binned_profiles[var]])
        binned['median'][var] = np.asarray([np.nanmedian(p, axis=axis) for p in binned_profiles[var]])
        binned['min'][var] = np.asarray(
            [np.nanmin(p, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
        )
        binned['max'][var] = np.asarray(
            [np.nanmax(p, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
        )
        binned['quart25'][var] = np.asarray(
            [np.nanpercentile(p, 25, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
        )
        binned['quart75'][var] = np.asarray(
            [np.nanpercentile(p, 75, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
        )

    # determine in-cloud ice/water content and cloud fraction in each bin    
    for condensate, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
        binned['mean'][content] = np.asarray(
            [np.ma.average(a, weights=(a > ic_thres[condensate]), axis=1).filled(0) for a in binned_profiles[condensate]]
        )
        binned['mean'][fraction] = np.asarray(
            [np.sum(a * (a > ic_thres[condensate]), axis=1) / a.shape[1] for a in binned_profiles[condensate]]
        )
        
    logger.info('Write output')
    #output files
    outname_perc = f'{model}-{run}_{start_date}-{end_date}_{num_percs}_perc_means_{num_samples}{sample_days_str}{log_average_str}_1exp.pkl'
    outname_binned = f'{model}-{run}_{start_date}-{end_date}_bin_means_{num_samples}{sample_days_str}{log_average_str}_1exp.pkl'
    
    with open(os.path.join(datapath, outname_perc), 'wb' ) as outfile:
        pickle.dump(perc, outfile) 
    with open(os.path.join(datapath, outname_binned), 'wb' ) as outfile:
        pickle.dump(binned, outfile)

def average_random_profiles_per_basin(model, run, time_period, variables, num_samples, sample_days, data_dir, **kwargs):
    """ Average randomly selected profiles in IWV percentile bins and IWV bins, separately for different
    ocean basins. Output is saved as .pkl files.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        variables (list of str): names of variables
        num_samples (num): number of randomly selected profiles
        sample_days (str): 'all', 'first10' or 'last10'
    """

    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'QR', 'QS', 'QG', 'RH', 'W', 'A_QV', 'A_RH', 'DRH_Dt_v', 'DRH_Dt_h', 'DRH_Dt_c', 'A_RH_v', 'A_QV_v', 'A_RH_h', 'A_QV_h', 'U', 'V']
    variables_2D = ['OLR', 'IWV', 'STOA', 'OLRC', 'STOAC', 'H_tropo', 'IWP', 'TQI', 'TQC', 'TQG', 'TQS', 'TQR']
    datapath = f'{data_dir}/{model}/random_samples/'
    filenames = '{}-{}_{}_sample_{}_{}-{}{}{}.nc'
    perc_values = np.arange(2, 100.5, 2.0)
    num_percs = len(perc_values)
    iwv_bin_bnds = np.arange(0, 101, 1)
    ic_thres = {
        'QI': 0.001 * 1e-6,
        'QC': 0.001 * 1e-6
    }
    if sample_days == 'all':
        sample_days_str = ''
    else:
        sample_days_str = '_'+sample_days
    ocean_basins = ['Pacific', 'Atlantic', 'Indic']
    stats = ['mean', 'std', 'median', 'min', 'max', 'quart25', 'quart75']
    
    logger.info('Read data from files')
    filename = filenames.format(model, run, variables_3D[0], num_samples, start_date, end_date, sample_days_str, '')
    filename = os.path.join(datapath, filename)
    with(Dataset(filename)) as ds:
        height = ds.variables['height'][:].filled(np.nan)
    num_levels = len(height)
    
    bins = np.arange(len(iwv_bin_bnds) - 1) 
    bin_count = np.zeros(len(iwv_bin_bnds) - 1)
    
    profiles_sorted = {}
    for var in variables+['lon', 'lat']:
        logger.info(var)
        filename = filenames.format(model, run, var, num_samples, start_date, end_date, sample_days_str, '')
        filename = os.path.join(datapath, filename)
        with(Dataset(filename)) as ds:
            profiles_sorted[var] = ds.variables[var][:].filled(np.nan)
    num_profiles = len(profiles_sorted[var])
            
    logger.info('Calc additional variables: T_QV, IWP and H_tropo')
    profiles_sorted['T_QV'] = -profiles_sorted['W'] * np.gradient(profiles_sorted['QV'], height, axis=0)
    
    profiles_sorted['IWP'] = np.ones(num_profiles) * np.nan
    profiles_sorted['IWP'] = utils.calc_IWP(
        profiles_sorted['QI'], 
        profiles_sorted['TEMP'], 
        profiles_sorted['PRES'], 
        profiles_sorted['QV'], 
        height
    )    
    profiles_sorted['H_tropo'] = utils.tropopause_height(
        profiles_sorted['TEMP'], 
        height
    )
            
    # create mask to exclude profiles with no given longitude       
    nan_mask = np.isnan(profiles_sorted['lon'])
    # create mask to exclude profiles with unrealistic advection values
    tropo = np.where(height < 20000)
    if 'A_QV_h' in variables:
        wrong_mask_1 = np.logical_or(np.any(profiles_sorted['A_QV_h'][tropo] < -1e-5, axis=0),
                                  np.any(profiles_sorted['A_QV_h'][tropo] > 1e-5, axis=0))

        nan_mask = np.logical_or(nan_mask, wrong_mask_1).astype(int)
        
    if 'DRH_Dt_h' in variables:
        wrong_mask_2 = np.logical_or(np.any(profiles_sorted['DRH_Dt_h'][tropo] < -0.001, axis=0),
                                  np.any(profiles_sorted['DRH_Dt_h'][tropo] > 0.001, axis=0))

        nan_mask = np.logical_or(nan_mask, wrong_mask_2).astype(int)
    
    # set masked profiles to nan
    for var in variables+['lon', 'lat']+extra_variables:
        if var in variables_3D:
            profiles_sorted[var][:, np.where(nan_mask)] = np.nan
        else:
            profiles_sorted[var][np.where(nan_mask)] = np.nan
    
    logger.info('Split profiles into corresponding ocean basins')
    # determine ocean basin for each profile
    basins = []
    for lon, lat in zip(profiles_sorted['lon'], profiles_sorted['lat']):
        basins.append(ocean_basin(lon, lat))
    basins = np.array(basins)
    
    # split profiles into basins
    profiles = {}
    for basin in ocean_basins:
        logger.info(basin)
        profiles[basin] = {}
        for var in variables+['lon', 'lat', 'IWP', 'H_tropo', 'T_QV']:
            if var in variables_3D:
                profiles[basin][var] = profiles_sorted[var][:, basins == basin]
            else:
                profiles[basin][var] = profiles_sorted[var][basins == basin]

    perc = {}
    binned = {}
    logger.info('Binning to percentile bins')
    for b in ocean_basins:
        logger.info(b)
        perc[b] = {s: {} for s in stats+['percentiles']}
        perc[b]['percentiles'] = np.asarray([np.percentile(profiles[b]['IWV'], p) for p in perc_values])
        splitted_array = {}
        for var in variables+['H_tropo', 'IWP', 'T_QV']:
            logger.info(var)
            splitted_array[var] = {}
            if var in variables_2D: 
                splitted_array[var] = np.array_split(profiles[b][var], num_percs)
                axis=0
            else:
                splitted_array[var] = np.array_split(profiles[b][var], num_percs, axis=1)
                axis=1
                
            if var in ['PRES', 'QV']:
                perc[b]['mean'][var] = np.asarray([np.exp(np.nanmean(np.log(a), axis=axis)) for a in splitted_array[var]])
            else:
                perc[b]['mean'][var] = np.asarray([np.nanmean(a, axis=axis) for a in splitted_array[var]])
            perc[b]['std'][var] = np.asarray([np.nanstd(a, axis=axis) for a in splitted_array[var]])
            perc[b]['min'][var] = np.asarray([np.nanmin(a, axis=axis) for a in splitted_array[var]])
            perc[b]['max'][var] = np.asarray([np.nanmax(a, axis=axis) for a in splitted_array[var]])
            perc[b]['median'][var] = np.asarray([np.nanmedian(a, axis=axis) for a in splitted_array[var]])
            perc[b]['quart25'][var] = np.asarray([np.nanpercentile(a, 25, axis=axis) for a in splitted_array[var]])
            perc[b]['quart75'][var] = np.asarray([np.nanpercentile(a, 75, axis=axis) for a in splitted_array[var]])
        

        # determine in-cloud ice/water content and cloud fraction in each bin
        for condensate, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
            perc[b]['mean'][content] = np.asarray(
                [np.ma.average(a, weights=(a > ic_thres[condensate]), axis=1).filled(0) for a in splitted_array[condensate]]
            )
            perc[b]['mean'][fraction] = np.asarray(
                [np.sum(a * (a > ic_thres[condensate]), axis=1) / a.shape[1] for a in splitted_array[condensate]]
            )

    logger.info('Binning to IWV bins')
    for b in ocean_basins:
        bin_idx = np.digitize(profiles[b]['IWV'], iwv_bin_bnds)
        binned[b] = {s: {} for s in stats+['count']}
        binned[b]['count'] = np.asarray([len(np.where(bin_idx == bi)[0]) for bi in bins])
        binned_profiles = {}
        for var in variables+['H_tropo', 'IWP', 'T_QV']:
            logger.info(var)
            if var in variables_2D+['IWP', 'H_tropo']:
                binned_profiles[var] = [profiles[b][var][bin_idx == bi] for bi in bins]
                axis=0
            else:
                binned_profiles[var] = [profiles[b][var][:, bin_idx == bi] for bi in bins]
                axis=1
            
            if var in ['PRES', 'QV']:
                binned[b]['mean'][var] = np.asarray([np.exp(np.nanmean(np.log(p), axis=axis)) for p in binned_profiles[var]])
            else:
                binned[b]['mean'][var] = np.asarray([np.nanmean(p, axis=axis) for p in binned_profiles[var]])
            binned[b]['std'][var] = np.asarray([np.nanstd(p, axis=axis) for p in binned_profiles[var]])
            binned[b]['median'][var] = np.asarray([np.nanmedian(p, axis=axis) for p in binned_profiles[var]])
            binned[b]['min'][var] = np.asarray(
                [np.nanmin(p, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
            )
            binned[b]['max'][var] = np.asarray(
                [np.nanmax(p, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
            )
            binned[b]['quart25'][var] = np.asarray(
                [np.nanpercentile(p, 25, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
            )
            binned[b]['quart75'][var] = np.asarray(
                [np.nanpercentile(p, 75, axis=axis) if p.size else np.ones(num_levels) * np.nan for p in binned_profiles[var]]
            )
            
        # determine in-cloud ice/water content and cloud fraction in each bin    
        for condensate, content, fraction in zip(['QI', 'QC'], ['ICQI', 'ICQC'], ['CFI', 'CFL']):
            binned[b]['mean'][content] = np.asarray(
                [np.ma.average(a, weights=(a > ic_thres[condensate]), axis=1).filled(0) for a in binned_profiles[condensate]]
            )
            binned[b]['mean'][fraction] = np.asarray(
                [np.sum(a * (a > ic_thres[condensate]), axis=1) / a.shape[1] for a in binned_profiles[condensate]]
            )
        
    logger.info('Write output')
    #output files
    outname_perc = f'{model}-{run}_{start_date}-{end_date}_{num_percs}_perc_means_basins_{num_samples}{sample_days_str}_0exp.pkl'
    outname_binned = f'{model}-{run}_{start_date}-{end_date}_bin_means_basins_{num_samples}{sample_days_str}_0exp.pkl'
    
    with open(os.path.join(datapath, outname_perc), 'wb' ) as outfile:
        pickle.dump(perc, outfile) 
    with open(os.path.join(datapath, outname_binned), 'wb' ) as outfile:
        pickle.dump(binned, outfile)

def get_time_array(model):
    """ Return array containing time stamps for files containing the period 
    2016-08-10 to 2016-09-10.
    """
    if model == 'ICON':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
    elif model == 'NICAM':
        time_array = pd.date_range("2016-08-10 3:00:00", "2016-09-09 0:00:00", freq='3h')
    elif model == 'GEOS':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
    elif model == 'SAM':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
    elif model == 'UM':
        time_array = pd.date_range("2016-08-10 3:00:00", "2016-09-09 0:00:00", freq='3h')
    elif model == 'IFS':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
    elif model == 'FV3':
        time_array = pd.date_range("2016-08-10 3:00:00", "2016-09-09 0:00:00", freq='3h')
    elif model == 'MPAS':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
    elif model == 'ARPEGE':
        time = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
        exclude_times = list(pd.date_range("2016-08-11 00:00:00", "2016-08-12 00:00:00", freq='3h'))
        exclude_times.append(pd.to_datetime('2016-08-10 00:00:00'))
        exclude_times.append(pd.to_datetime('2016-08-10 15:00:00'))
        time_array = [t for t in time if t not in exclude_times]
    elif model == 'ERA5':
        time_array = pd.date_range("2016-08-10 0:00:00", "2016-09-08 21:00:00", freq='3h')
         
    return time_array
        
def ocean_basin(lon, lat):
    """ Returns the ocean basin for given longitude and latitude in the tropics.
    
    Parameters:
        lon (float): Longitude (between -180 and 180)
        lat (float): Latitude (between -30 and 30)
    """
    if lon < -70. and lon > -84. and lat < 9.:
        basin = 'Pacific'
    elif lon < -84. and lon > -90. and lat < 14.:
        basin = 'Pacific'
    elif lon < -90. and lon > -100. and lat < 18.:
        basin = 'Pacific'
    elif lon > 145. or lon < -100.:
        basin = 'Pacific'
    elif lon < 145. and lon > 100. and lat > 0.:
        basin = 'Pacific'
    elif lon > 20. and lon < 100.:
        basin = 'Indic'
    elif lon < 145. and lon > 100. and lat < 0.:
        basin = 'Indic'
    elif lon < 20. and lon > -70.:
        basin = 'Atlantic'
    elif lon < -70. and lon > -84. and lat > 9.:
        basin = 'Atlantic'
    elif lon < -84. and lon > -90. and lat > 14.:
        basin = 'Atlantic'
    elif lon < -90. and lon > -100. and lat > 18.:
        basin = 'Atlantic'
    
    return basin