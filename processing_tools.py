import numpy as np
import pandas as pd
import os
import glob
import logging
import json
import typhon
import random
import pickle
import filelists
import analysis_tools as atools
from time import sleep
from scipy.interpolate import interp1d
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from moisture_space import utils
import netCDF_tools as nctools

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def config():
    """ Reads specifications for processing from confing.json and returns a dictionary
    with specifications.
    """
    with open('/mnt/lustre02/work/um0878/users/tlang/work/dyamond/processing/config.json') as handle:
        config = json.loads(handle.read())
    
    return config

def preprocess(model, infile, tempfile, outfile, option_1, option_2, numthreads, **kwargs):
    """ Performs preprocessing steps (before horizontal interpolation). This function just 
    calls the model-specific function for preprocessing.
    
    Parameters:
        model (str): Name of model
        infile (str): Path to input file
        tempfile (str): Path to temporary output file
        outfile (str): Path to output file
        option_1 (str), option_2 (str): options for preprocessing
        numthreads (int): Number of OpenMP threads for cdo
    """
    
    if model == 'IFS':
        preprocess_IFS(infile, tempfile, outfile, option_1, option_2, numthreads)
    elif model == 'MPAS':
        preprocess_MPAS(infile, tempfile, outfile, option_1, numthreads)
    elif model == 'ERA5':
        preprocess_ERA5(infile, outfile, option_1, option_2, numthreads)
        
    
def preprocess_IFS(infile, tempfile, outfile, option_selvar, option_nzlevs, numthreads, **kwargs):
    """ Perform processing steps required before horizontal interpolation for the IFS model.
    
    Parameters:
        infile (str): Name of input file
        tempfile (str): Name for temporary files
        outfile (str): Name for preprocessed output file
        option_selvar (str): Name of variable to select from input file
        option_nzlevs (str): Number of vertical levels
        numthreads (int): Number of OpenMP threads for cdo
    """
    logger.info('Preprocessing IFS...')
    filename, file_extension = os.path.splitext(tempfile)
    tempfile_1 = tempfile
    tempfile_2 = filename+'_2'+file_extension
    #tempfile_3 = filename+'_3'+file_extension
    
    # for OLR and IWV less pre-processing steps are needed than for the other variables
    if option_nzlevs == '':
        cmd_1 = f'cdo --eccodes select,name={option_selvar} {infile} {tempfile_1}'
        #cmd_2 = f'grib_set -s editionNumber=1 {tempfile_1} {tempfile_2}'
        #cmd_3 = f'rm {tempfile_1}'
        cmd_2 = f'cdo -P {numthreads} -R -f nc4 copy {tempfile_1} {outfile}'
        cmd_3 = f'rm {tempfile_1}'
        
        logger.info(cmd_1)
        os.system(cmd_1)
        logger.info(cmd_2)
        os.system(cmd_2)
        logger.info(cmd_3)
        os.system(cmd_3)
        
    else:
        cmd_1 = f'cdo --eccodes select,name={option_selvar} {infile} {tempfile_1}'
        cmd_2 = f'grib_set -s numberOfVerticalCoordinateValues={option_nzlevs} {tempfile_1} {tempfile_2}'
        cmd_3 = f'rm {tempfile_1}'
        #cmd_4 = f'grib_set -s editionNumber=1 {tempfile_2} {tempfile_3}'
        #cmd_5 = f'rm {tempfile_2}'
        #cmd_6 = f'cdo -P {numthreads} -R -f nc4 copy {tempfile_3} {outfile}'
        cmd_4 = f'cdo -P {numthreads} -f nc4 -setgridtype,regular {tempfile_2} {outfile}'
        cmd_5 = f'rm {tempfile_2}'

        logger.info(cmd_1)
        os.system(cmd_1)
        logger.info(cmd_2)
        os.system(cmd_2)
        logger.info(cmd_3)
        os.system(cmd_3)
        logger.info(cmd_4)
        os.system(cmd_4)
        logger.info(cmd_5)
        os.system(cmd_5)


def preprocess_MPAS(infile, tempfile, outfile, option_selvar, numthreads, **kwargs):
    """ Perform processing steps required before horizontal interpolation for the MPAS model.
    
    Parameters:
        infile (str): Name of input file
        tempfile (str): Name for temporary files
        outfile (str): Name for preprocessed output file
        numthreads (int): Number of OpenMP threads for cdo
    """
    logger.info('Preprocessing MPAS...')
    gridfile = '/work/ka1081/DYAMOND/PostProc/GridsAndWeights/MPAS-3.75km_gridinfo.nc'
    cmd_1 = f'cdo -P 4 setattribute,*@axis="txz" -selname,{option_selvar} {infile} {tempfile}'
    cmd_2 = f'cdo -P 4 setgrid,mpas:{gridfile} {tempfile} {outfile}'
    cmd_3 = f'rm {tempfile}'
    
    logger.info(cmd_1)
    os.system(cmd_1)
    logger.info(cmd_2)
    os.system(cmd_2)
    logger.info(cmd_3)
    os.system(cmd_3)
    
def preprocess_ERA5(infile, outfile, option_grid, option_splitday, numthreads, **kwargs):
    """ Perform processing steps required before horizontal interpolation for the ERA5 reanalysis.
    """
    logger.info('Preprocessing ERA5...')
    
    if option_grid == 'gaussian':
        # Change file format from GRIB2 to NetCDF4
        # -R option: Change from Gaussian reduced to regular Gaussian grid
        cmd = f'cdo -P {numthreads} -f nc4 -{option_splitday} -setgridtype,regular {infile} {outfile}'
    elif option_grid == 'spectral':
        # Some variables (e.g. temperature, vertical velocity) are given on a spectral grid,
        # which has to be transformed to a regular Gaussian grid first.
        cmd = f'cdo -P {numthreads} -f nc4 sp2gp,linear {infile} {outfile}' 
     
    logger.info(cmd)
    os.system(cmd)
    
def preprocess_ARPEGE_1(preprocess_dir, infile_2D, infile_3D, outfileprefix, merge_list_3D, tempfile_list_3D,\
                        filelist_2D, tempfile_list_2D, variables, **kwargs):
    """ Preprocessing of ARPEGE raw data (steps needed before horizontal interpolation).
    
    Parameters: 
        preprocess_dir (str): full path to directory where preprocessing is performed 
            (gribsplit can only write output into the directory where it runs)
        infile_2D (str): full path to file containing ARPEGE 2D raw data for one time step
        infile_3D (str): full path to file containing ARPEGE 3D raw data for one time step
        outfileprefix (str): prefix for temporary files (should contain date and time)
        merge_list_3D (list of list of str): lists of filenames of 3D variables produced by gribsplit,
            that have to be merged (containing data for individual model levels), 
            one list for each variable
        tempfile_list_3D (list of str): names for files containing merged 3D variables 
            (one name for every variable)
        filelist_2D (list of str): list of filenames of 2D variables produced by gribsplit
        tempfile_list_2D (list of str): list of filenames of 2D variables (files produced by
            gribsplit are renamed)
    """
    variables_3D = ['GH', 'TEMP', 'QV', 'QI', 'QC', 'W', 'U', 'V']
    variables_2D = ['SURF_PRES', 'OLR', 'STOA']
    
    # split file containing 3D variables
    if merge_list_3D:
        cmd_1 = f'{preprocess_dir}/mygribsplit {infile_3D} {outfileprefix}'
        logger.info(cmd_1)
        os.system(cmd_1)
    
    # split file containing 2D variables
    if filelist_2D:
        cmd_2 = f'{preprocess_dir}/mygribsplit {infile_2D} {outfileprefix}'
        logger.info(cmd_2)
        os.system(cmd_2)
     
    i = 0 # index for 3D variables
    j = 0 # index for 2D variables
    for var in variables:
        # merge individual model levels for each 3D variable
        if var in variables_3D:
            mergelist_str = ' '.join(merge_list_3D[i])
            cmd_3 = f'cdo merge {mergelist_str} {tempfile_list_3D[i]}'
            i += 1
        
        # rename files containing 2D variables
        elif var in variables_2D:
            cmd_3 = f'mv {filelist_2D[j]} {tempfile_list_2D[j]}'
            j += 1
            
        logger.info(cmd_3)
        os.system(cmd_3)
    
    # remove files that are not needed
    cmd_4 = f'rm {outfileprefix}.t*.l*.grb.*'
    logger.info(cmd_4)
    os.system(cmd_4)
    
def interpolate_horizontally(infile, target_grid, weights, outfile, options, numthreads,
                             lonlatbox=[-180, 180, -30, 30], **kwargs):
    """ Horizontal Interpolation to another target grid with the CDO command remap 
    that uses pre-calculated interpolation weights. The selection of a lon-lat box is 
    performed at the same time.
    
    Parameters:
        infile (str): path to file with input 2D or 3D field
        target_grid (str): path to file with target grid information
        weights (str): path to file with pre-calculated interpolation weights
        outfile (str): output file (full path)
        numthreads (int): number of OpenMP threads for cdo 
        lonlatbox (list of num): boundaries of lon-lat box to select 
            [lonmin, lonmax, latmin, latmax]
    """
    lon1 = lonlatbox[0]
    lon2 = lonlatbox[1]
    lat1 = lonlatbox[2]
    lat2 = lonlatbox[3]
    
    filename, file_extension = os.path.splitext(infile)
    if file_extension == '.nc' or file_extension == '.nc4':
        to_netCDF = ''
    else:
        to_netCDF = '-f nc4'
    
    if options != '':
        options += ' '
        
    cmd = f'cdo --verbose -O {to_netCDF} -P {numthreads} sellonlatbox,{lon1},{lon2},{lat1},{lat2} -remap,{target_grid},{weights} {options}{infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)
    
def calc_relative_humidity(temp_file, qv_file, pres_file, timestep, model, run, time_period, temp_dir, **kwargs):
    """ Calculate relative humidity from temperature, specific humidity and pressure.
    Note: Only one timestep is selected from input files, output files contain one time step per file and have to be
    merged afterwards. 
    
    Parameters:
        temp_file (str): File containing temperature (all timesteps)
        qv_file (str): File containing specific humidity (all timesteps)
        pres_file (str): File containing pressure (all timesteps)
        timestep (int): Time step to select
        model (str): Name of model
        run (str): Name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        temp_dir (str): Directory for output files
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    if model == 'ARPEGE':
        exclude_times = list(pd.date_range("2016-08-11 00:00:00", "2016-08-12 00:00:00", freq='3h'))
        exclude_times.append(pd.to_datetime('2016-08-10 00:00:00'))
        exclude_times.append(pd.to_datetime('2016-08-10 15:00:00'))
        time = [t for t in time if t not in exclude_times]
    # variable names
    temp_name = 'TEMP'
    qv_name = 'QV'
    if model == 'SAM':
        pres_name = 'PP'
    else:
        pres_name = 'PRES'
    
    # Read variables from files
    logger.info('Read variables from files')
    # temperature
    with Dataset(temp_file) as ds:
        temp = ds.variables[temp_name][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    
    # pressure (for SAM mean pressure and perturbation)
    with Dataset(pres_file) as ds:
        if model == 'SAM':
            pres_pert = ds.variables[pres_name][timestep].filled(np.nan)
            pres_mean = ds.variables['p'][:].filled(np.nan)
        else:
            pres = ds.variables[pres_name][timestep].filled(np.nan)
            
    # Specific humidity        
    with Dataset(qv_file) as ds:
        qv = ds.variables[qv_name][timestep].filled(np.nan)
    
    # for SAM, pressure has to be calculated from mean pressure and pressure perturbation
    if model == 'SAM':
        qv = qv * 1e-3
        pres = np.zeros(pres_pert.shape)
        for i in range(pres_pert.shape[1]):
            for j in range(pres_pert.shape[2]):
                pres[:, i, j] = pres_mean * 1e2 + pres_pert[:, i, j]
    
    # Calculate RH
    logger.info('Calc VMR')
    vmr = typhon.physics.specific_humidity2vmr(qv)
    logger.info('Calc e_eq')
    e_eq = typhon.physics.e_eq_mixed_mk(temp)
    logger.info('Calc RH')
    rh = vmr * pres / e_eq
    rh = np.expand_dims(rh, axis=0)
    if model == 'MPAS':
        rh = rh.transpose(0, 3, 1, 2)
    
    # Save RH to file 
    logger.info('Save RH to file')
    height = np.arange(rh.shape[1])
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_RH_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    nctools.latlonheightfield_to_netCDF(height, lat, lon, rh, 'RH', '[]', outname, time_dim=True, time_var=timestep*3, overwrite=True)

def calc_vertical_velocity(omega_file, temp_file, qv_file, pres_file, heightfile, timestep, model, run, time_period, temp_dir, **kwargs):
    """ Calculate vertical velocity w from pressure velocity omega (requires temperature, specific humidity and pressure).
    Note: Only one timestep is selected from input files, output files contain one time step per file and have to be
    merged afterwards. 
    
    Needed for ERA5 only 
    
    Parameters:
        omega_file (str): File containing pressure velocity omega (all timesteps)
        temp_file (str): File containing temperature (all timesteps)
        qv_file (str): File containing specific humidity (all timesteps)
        pres_file (str): File containing pressure (all timesteps)
        timestep (int): Time step to select
        model (str): Name of model
        run (str): Name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        temp_dir (str): Directory for output files
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    # read variables   
    logger.info('Read variables from files')
    # temperature
    with Dataset(temp_file) as ds:
        temp = ds.variables['TEMP'][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    # omega
    with Dataset(omega_file) as ds:
        omega = ds.variables['OMEGA'][timestep].filled(np.nan)
    # pressure    
    with Dataset(pres_file) as ds:
        pres = ds.variables['PRES'][timestep].filled(np.nan)
    # specific humidity        
    with Dataset(qv_file) as ds:
        qv = ds.variables['QV'][timestep].filled(np.nan)
    
    # calculate vertical velocity
    logger.info('Calculate vertical velocity')
    g = typhon.constants.g
    # mixing ratio
    r = typhon.physics.specific_humidity2mixing_ratio(qv) 
    # virtual temperature
    virt_temp = temp * (1 + 0.61 * r) 
    # air density (from virtual temperature)
    rho = typhon.physics.density(pres, virt_temp)
    # vertical velocity
    w = -omega / rho / g
    
    # save to file
    logger.info('Save omega to file')
    w = np.expand_dims(w, axis=0)
    height = np.arange(w.shape[1])
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_W_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    logger.info('Save file')
    print(outname)
    nctools.latlonheightfield_to_netCDF(height, lat, lon, w, 'W', '[]', outname, time_dim=True, time_var=timestep*3, overwrite=True)
         
    
def interpolate_vertically_per_timestep(infile, height_file, target_height_file, timestep, model, run, variable, time_period, temp_dir, **kwargs):
    """ Perform vertical interpolation for one timestep included in the input files. Every input file contains one variable.
    Note: Only one timestep is selected from input files, output files contain one time step per file and have to be
    merged afterwards.
    
    Parameters:
        infiles (list of str): List of input files, one for every variable (in the same order as in the variable variables)
        height_file (str): Path to file containing geometric heights for every model level for this timestep
        target_height_file (str): Path to file containing target heights
        timestep (int): Timestep to process
        model (str): Name of model
        run (str): Name of run
        variable (str): Variable to interpolate  
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    if model == 'ARPEGE':
        exclude_times = list(pd.date_range("2016-08-11 00:00:00", "2016-08-12 00:00:00", freq='3h'))
        exclude_times.append(pd.to_datetime('2016-08-10 00:00:00'))
        exclude_times.append(pd.to_datetime('2016-08-10 15:00:00'))
        time = [t for t in time if t not in exclude_times]
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_{variable}_hinterp_vinterp_{date_str}_{hour_str}.nc'
    outname = os.path.join(temp_dir, outname)    
    units = filelists.get_variable_units()
    
    # Read variables from files
    logger.info('Read variables from files')
    # target height
    with Dataset(target_height_file) as dst:
        target_height = dst.variables['target_height'][:].filled(np.nan)
    # height     
    with Dataset(height_file) as dsh:
        heightname = 'H'
        var_height = dsh.variables[heightname][timestep].filled(np.nan)
    # lon, lat and variable to interpolate            
    with Dataset(infile) as ds:
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        field = ds.variables[variable][timestep].filled(np.nan)

    nheight, nlat, nlon = field.shape
    ntargetheight = len(target_height)
    field_interp = np.ones((ntargetheight, nlat, nlon)) * np.nan

    # interpolate
    logger.info('Interpolation')
    for i in range(nlat):
        print(f'Lat: {i} of {nlat}')
        for j in range(nlon):
            field_interp[:, i, j] = interp1d(
                var_height[:, i, j],
                field[:, i, j],
                bounds_error=False,
                fill_value='extrapolate')(target_height)
    
    # save interpolated field to netCDF file
    logger.info('Save to file')
    field_interp = np.expand_dims(field_interp, axis=0)
    nctools.latlonheightfield_to_netCDF(target_height, lat, lon, field_interp,\
                                variable, units[variable], outname, time_dim=True, time_var=timestep*3, overwrite=True)  
        
def interpolation_to_halflevels_per_timestep(infile, model, run, variable, timestep, time_period, data_dir, temp_dir, **kwargs):
    """ Perform vertical interpolation for a field given on full model levels to half levels for one timestep included
    in the input file. 
    Note: Only one timestep is selected from input files, output files contain one time step per file and have to be
    merged afterwards.
    
    Parameters:
        infile (str): Path to input file
        model (str): Name of model
        run (str): Name of model run
        variable (str): Variable to interpolate 
        timestep (int): Timestep to process
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        temp_dir (str): Directory for output files
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_{variable}HL_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    units = get_variable_units()
    
    # Read variables from file
    logger.info('Read from file')
    varname = variable    
    with Dataset(infile) as ds:
        field = np.squeeze(ds.variables[varname][timestep].filled(np.nan))
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    # Dimensions are different for MPAS
    if model == 'MPAS':
        field = field.transpose(2, 0, 1)

    # Interpolation
    logger.info('Interpolation')
    field_interp = field[:-1] + 0.5 * np.diff(field, axis=0)    
    field_interp = np.expand_dims(field_interp, 0)
    height = np.arange(field_interp.shape[1])
    # Save to file
    logger.info('Save to file')
    nctools.latlonheightfield_to_netCDF(height, lat, lon, field_interp, variable, units[variable], outname, time_dim=True, time_var=timestep*3, overwrite=True)
            
def calc_level_pressure_from_surface_pressure(surf_pres_file, timestep, model, run, temp_dir, time_period, **kwargs):
    """ Calculate pressure at IFS model levels from surface pressure for one timestep contained in 
    surf_pres_file.
    Note: Only one timestep is selected from input files, output files contain one time step per file and have to be
    merged afterwards.
    
    Parameters:
        surf_pres_file (str): Path to file that contains (logarithm of) surface pressure for every timestep
        timestep (int): Timestep for which to calculate pressure
        temp_dir (str): Path to directory to save output files
        model (str): Name of model
        run (str): Name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    if model == 'ARPEGE':
        exclude_times = list(pd.date_range("2016-08-11 00:00:00", "2016-08-12 00:00:00", freq='3h'))
        exclude_times.append(pd.to_datetime('2016-08-10 00:00:00'))
        exclude_times.append(pd.to_datetime('2016-08-10 15:00:00'))
        time = [t for t in time if t not in exclude_times]
        
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_PRES_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    
    # Read surface pressure from file
    logger.info('Read surface pressure from file')
    with Dataset(surf_pres_file) as ds:
        # calc surface pressure from logarithm of surface pressure
        if model == 'IFS':
            surf_pres = np.exp(ds.variables['SURF_PRES'][timestep].filled(np.nan))
        else:
            surf_pres = ds.variables['SURF_PRES'][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    
    # Calculate pressure
    logger.info('Calculate pressure')
    # Scaling parameters
    a, b = get_pressure_scaling_parameters(model)
    if len(surf_pres.shape) == 2:
        surf_pres = np.expand_dims(surf_pres, axis=0)
    pres = np.ones((len(a), surf_pres.shape[1], surf_pres.shape[2])) * np.nan    
    for la in range(surf_pres.shape[1]):
        for lo in range(surf_pres.shape[2]):
            if model == 'IFS':
                pres[:, la, lo] = np.flipud(surf_pres[:, la, lo] * b + a)
            if model == 'ERA5':
                pres[:, la, lo] = np.flipud(surf_pres[:, la, lo] * b + a)
            else:
                pres[:, la, lo] = surf_pres[:, la, lo] * b + a
            #return(a[ilev-1]+b[ilev-1]*surfacepressure)
            
    # Save to file
    logger.info('Save pressure to file')
    height = np.arange(pres.shape[0])
    pres = np.expand_dims(pres, axis=0)
    nctools.latlonheightfield_to_netCDF(height, lat, lon, pres, 'PRES', 'Pa', outname, time_dim=True, time_var=timestep*3, overwrite=True)
    
def calc_height_from_pressure(pres_file, temp_file, z0_file, timestep, model, run, time_period, temp_dir, **kwargs):
    """ Calculate level heights from level pressure and layer temperature assuming hydrostatic equilibrium (Needed
    for IFS, SAM, FV3 and ARPEGE).
    
    Parameters:
        pres_file (str): Path to file containing pressures for every model level
        temp_file (str): Path to file containing temperatures for every model level
        z0_file (str): Path to file containing orography (as geopotential height)
        timestep (int): Timestep in pres_file and temp_file to process
        model (str): Name of model
        run (str): Name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        temp_dir (str): Path to directory for output files
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    if model == 'ARPEGE':
        exclude_times = list(pd.date_range("2016-08-11 00:00:00", "2016-08-12 00:00:00", freq='3h'))
        exclude_times.append(pd.to_datetime('2016-08-10 00:00:00'))
        exclude_times.append(pd.to_datetime('2016-08-10 15:00:00'))
        time = [t for t in time if t not in exclude_times]
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_H_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    
    # variable names
    temp_name = 'TEMP'
    pres_name = 'PRES'
    
    # Read variables from file
    logger.info('Read from file')
    # temperature, lat, lon
    with Dataset(temp_file) as ds:
        temp = ds.variables[temp_name][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        
    with Dataset(pres_file) as ds:
        # IFS: logarithm of surface pressure is given
        if model == 'IFS':
            surf_pres = np.exp(ds.variables[pres_name][timestep].filled(np.nan))
        # SAM: mean pressure and pressure perturbation are given
        if model == 'SAM':
            pres_pert = ds.variables[pres_name][timestep].filled(np.nan)
            pres_mean = ds.variables['p'][:].filled(np.nan)
        # FV3: only one pressure vector is given (same for every grid cell)
        if model == 'FV3':
            pres_vector = ds.variables['pfull'][:].filled(np.nan)
            pres = np.zeros((len(pres_vector), len(lat), len(lon)))
            for la in range(len(lat)):
                for lo in range(len(lon)):
                    pres[:, la, lo] = pres_vector
        else:
            pres = ds.variables[pres_name][timestep].filled(np.nan)
            
    # orography
    with Dataset(z0_file) as ds:
        if model in ['IFS', 'ERA5']:
            z0 = ds.variables['z'][0][0].filled(np.nan) / typhon.constants.g
        elif model == 'FV3':
            z0 = ds.variables['z'][:].filled(np.nan)
        elif model == 'ARPEGE':
            z0 = ds.variables['GH'][timestep][0].filled(np.nan) / typhon.constants.g 
            # z0 = np.ones((len(lat), len(lon))) * 17. (lowest model level is at 17 meters above sea level)
            
    # Calculate heights
    logger.info('Calculate heights')
    height = np.flipud(pressure2height(np.flipud(pres), np.flipud(temp), z0))

    # Save to file
    logger.info('Save heights to file')
    h = np.arange(height.shape[0])
    height = np.expand_dims(height, axis=0)    
    nctools.latlonheightfield_to_netCDF(h, lat, lon, height, 'H', 'm', outname, time_dim=True, time_var=timestep*3, overwrite=True)


def calc_net_radiation_flux(infiles, tempfile, outfile, flux_names):
    """ Calculates net radiation flux from upward and downward fluxes using CDOs only. Files containing
    upward and downward fluxes are merged first, then the net flux is calculated.
    
    Parameters:
        infiles (list of str): Name of input files containing 1. downward and 2. upward fluxes
        tempfile (str): Name of temporary file containing both downward and upward fluxes
        outfile (str): Name of output file containing net fluxes
        flux_names (list of str): Variable names of 1. downward, 2. upward and 3. net fluxes 
            (e.g. ['SDTOA', 'SUTOA', 'STOA'])
    """
    cmd_1 = f'cdo --verbose merge {infiles[0]} {infiles[1]} {tempfile}'
    cmd_2 = f'cdo --verbose expr,{flux_names[2]}={flux_names[0]}-{flux_names[1]} {tempfile} {outfile}'
    cmd_3 = f'rm {tempfile}'
    
    logger.info(cmd_1)
    os.system(cmd_1)
    logger.info(cmd_2)
    os.system(cmd_2)
    logger.info(cmd_3)
    os.system(cmd_3)
    
def deaccumulate_fields(model, infile, variable):
    """ Deaccumulate radiation fields, i.e. transform radiation given as accumulated energy [J] to 
    power [W m**-2].
    
    Parameters:
        model: model name
        infile: full path to input file
        variable: name of variable
    """
   
    if model == 'IFS':
        with Dataset(infile) as ds:
            field = ds.variables[variable][:].filled(np.nan)
            lat = ds.variables['lat'][:].filled(np.nan)
            lon = ds.variables['lon'][:].filled(np.nan)
            time = ds.variables['time'][:].filled(np.nan)
    if model == 'MPAS':
        with Dataset(infile) as ds:
            time = ds.variables['xtime'][:].filled(np.nan)
            field = ds.variables[variable][:].filled(np.nan)
            lat = ds.variables['lat'][:].filled(np.nan)
            lon = ds.variables['lon'][:].filled(np.nan)
        
    field_deacc = np.zeros((field.shape[0] - 1, field.shape[1], field.shape[2]))
    for t in range(field.shape[0] - 1):
        field_deacc[t] = (field[t + 1] - field[t]) / (3600 * 3) * -1
        
    time_new = time[:-1] + np.diff(time) * 0.5
    
    infilename, infile_extension = os.path.splitext(infile)
    filename_new = infilename+'_acc'+infile_extension
    os.system(f'mv {infile} {filename_new}')
        
    outname = infile
    nctools.latlonfield_to_netCDF(lat, lon, field_deacc, variable, 'W m^-2', outname, time_dim=True, time_var=time_new, overwrite=True)
    
def pressure2height(p, T, z0=None):
    """Convert pressure to height based on the hydrostatic equilibrium.

    Parameters:
        p (ndarray): Pressure [Pa].
        T (ndarray): Temperature [K].
        z0 (ndarray): Height of lowest pressure level [m].

    Returns:
        ndarray: Height [m].
    """
     
    layer_depth = np.diff(p, axis=0)
    rho = typhon.physics.thermodynamics.density(p, T)
    rho_layer = 0.5 * (rho[:-1] + rho[1:])

    z = np.cumsum(-layer_depth / (rho_layer * typhon.constants.g), axis=0) + z0
    z = np.vstack((np.expand_dims(z0, axis=0), z))
    
    return z  
        
def merge_timesteps(infiles, outfile, numthreads, **kwargs):
    """ Merge files containing several time steps to one file containing all time steps.
    
    Parameter:
        infiles (list of str): list of paths to input files to be merged
        outfile (str): name of output file
        numthreads (int): number of OpenMP threads for cdo
    """
    infiles_str = ' '.join(infiles)
    cmd = f'cdo -O -P {numthreads} mergetime {infiles_str} {outfile}'
    logger.info(cmd)
    os.system(cmd)

def average_timesteps(infile, outfile, options, numthreads, **kwargs):
    """
    Calculate temporal average of all time steps contained in one file.
    
    Parameter:
        infile (str): path to input file containing several time steps
        outfile (str): name for file containing time average
        numthreads (int): number of OpenMP threads for cdo
    """
    if options != '':
        options = options + ' '
    cmd = f'cdo -O -P {numthreads} timmean {options}{infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)

def select_lonlatbox(infile, outfile, lonlatbox, numthreads, **kwargs):
    """ Select a lon-lat box from the fields contained in one file.
    
    Parameters:
        infile (str): path to input file containing several time steps
        outfile (str): name for file containing time average
        lonlatbox (list): boundaries of latitude longitude box: [lonmin lonmax latmin latmax]
        numthreads (int): number of OpenMP threads for cdo
    """
    lon_1 = lonlatbox[0]
    lon_2 = lonlatbox[1]
    lat_1 = lonlatbox[2]
    lat_2 = lonlatbox[3]
    cmd = f'cdo -O -P {numthreads} sellonlatbox,{lon_1},{lon_2},{lat_1},{lat_2} {infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)

def select_random_profiles(model, run, num_samples_tot, infiles, outfiles, heightfile, landmaskfile,\
                               variables, lonlatbox, vinterp, data_dir, sample_days, timesteps=None, **kwargs):
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
    variables_2D = ['OLR', 'OLRC', 'STOA', 'IWV', 'lat', 'lon', 'timestep']
    test_ind = [i for i in range(len(variables)) if variables[i] not in variables_2D][0]
    test_var = variables[test_ind]
    test_filename = infiles[test_ind]
    
    logger.info('Get dimensions')
    # height
    height = atools.read_var(heightfile, model, 'target_height')
    num_levels = height.shape[0]
    if height[0] < height[-1]:
        surface = 0
    else:
        surface = -1
    if model == 'GEOS':
        surface = -2
    
    print(surface)
    logger.info('Read lats/lons from file')
    # lon, lat
    with Dataset(test_filename) as ds:
        dimensions = ds.variables[test_var].shape
        test_field = ds.variables[test_var][0].filled(np.nan)
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        
    lat_reg_ind = np.where(np.logical_and(lat >= lonlatbox[2], lat <= lonlatbox[3]))[0]
    lon_reg_ind = np.where(np.logical_and(lon >= lonlatbox[0], lon <= lonlatbox[1]))[0]
    lat = lat[lat_reg_ind]
    lon = lon[lon_reg_ind]
    
    logger.info('NaN mask')
    if model == 'MPAS' and test_var in ['TEMP', 'PRES', 'QV', 'QI', 'QC']:
        nan_mask = np.logical_not(np.isnan(test_field[lat_reg_ind][:, lon_reg_ind][:, :, surface]))
    else:
        nan_mask = np.logical_not(np.isnan(test_field[surface][lat_reg_ind][:, lon_reg_ind]))
    
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

    #return nan_mask
    
    logger.info('Ocean mask')
    # get ocean_mask
    ocean_mask = np.logical_not(
        np.squeeze(atools.read_var(landmaskfile, model, 'land_mask'))
    )
    ocean_lon = atools.read_var(landmaskfile, model, 'lon')
    ocean_lat = atools.read_var(landmaskfile, model, 'lat')
    ocean_lat_ind = np.where(np.logical_and(ocean_lat >= lonlatbox[2], ocean_lat <= lonlatbox[3]))[0]
    ocean_lon_ind = np.where(np.logical_and(ocean_lon >= lonlatbox[0], ocean_lon <= lonlatbox[1]))[0]
    ocean_mask = ocean_mask[ocean_lat_ind][:, ocean_lon_ind].astype(int)
#    lat_ocean = np.where(ocean_mask)[0]
#    lon_ocean = np.where(ocean_mask)[1]

    logger.info('Total mask')    
    total_mask = np.where(np.logical_and(ocean_mask, nan_mask))
    lat_not_masked = total_mask[0]
    lon_not_masked = total_mask[1]
    
    lat_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)
    lon_inds = np.zeros((num_timesteps, num_samples_timestep)).astype(int)

    logger.info('Get indices')
    for t in range(num_timesteps):
        r = random.sample(list(zip(lat_not_masked, lon_not_masked)), num_samples_timestep)
        lat_inds[t], lon_inds[t] = zip(*r)
    
    # save indices
    nctools.array2D_to_netCDF(
        lat_inds, 'idx', '', (range(num_timesteps), range(num_samples_timestep)),
        ('timestep', 'profile_index'), outfiles[-2], overwrite=True
            )
    nctools.array2D_to_netCDF(
        lon_inds, 'idx', '', (range(num_timesteps), range(num_samples_timestep)),
        ('timestep', 'profile_index'), outfiles[-3], overwrite=True
            )
    
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
                    if model == 'MPAS' and var in ['TEMP', 'PRES', 'QV', 'QI', 'QC']:
                        profiles[var][:, start:end] = ds.variables[var][t][lat_inds[j], lon_inds[j], :].filled(np.nan).transpose([1, 0])
                    else:
                        if model == 'SAM' and var == 'PRES':
                            pres_mean = np.expand_dims(ds.variables['p'][:], 1) * 1e2
                            pres_pert = ds.variables['PP'][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)
                            profiles[var][:, start:end] = pres_mean + pres_pert
                            print(profiles[var][:, start:end])
                        else:
                            profiles[var][:, start:end] = ds.variables[var][t][:, lat_inds[j], lon_inds[j]].filled(np.nan)
                        
        if model == 'SAM' and var in ['QV', 'QI', 'QC']:
            profiles[var] *= 1e-3
            
    logger.info('Calculate IWV and sort')
    # calculate IWV
    profiles['IWV'] = utils.calc_IWV(profiles['QV'], profiles['TEMP'], profiles['PRES'], height)
    # get indices to sort by IWV
    IWV_sort_idx = np.argsort(profiles['IWV'])
    
    # save indices
    nctools.vector_to_netCDF(
        IWV_sort_idx, 'idx', '', range(num_samples_timestep * num_timesteps), 'profile_index', outfiles[-1], overwrite=True
            )
            
    # sort by IWV and save output
    for i, var in enumerate(variables + ['IWV', 'lon', 'lat']):
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
    """ Calculate horizontal advection of QV and RH for randomly selected profiles.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        num_samples (num): number of randomly selected profiles
        data_dir (str): Path to output directory
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")

    logger.info('Input and output variables and filenames')
    input_variables = ['U', 'V', 'QV', 'RH']
    output_variables = ['U', 'V', 'A_QV_h', 'A_RH_h']
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
    for var in ['U', 'V', 'A_QV_h', 'A_RH_h']:
        profiles[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan
        #profiles_sorted[var] = np.ones((num_levels, num_samples_timestep * num_timesteps)) * np.nan

    logger.info('Read U and V from files')
    for var in ['U', 'V']:
        for t in range(num_timesteps):
            start = t * num_samples_timestep
            end = start + num_samples_timestep
            with Dataset(filenames[var]) as ds:
                profiles[var][:, start:end] = ds.variables[var][t][:, lat_idx[t], lon_idx[t]].filled(np.nan)

    logger.info('Read QV and RH from files and calculate advection')
    for t in range(num_timesteps):
        print(t)
        start = t * num_samples_timestep
        end = start + num_samples_timestep
        lon_idx_next = lon_idx[t] + 1
        lon_idx_next[lon_idx_next == len(lon)] = 0
        lon_idx_before = lon_idx[t] - 1
        lon_idx_before[lon_idx_before < 0] = len(lon) - 1
        with Dataset(filenames['QV']) as ds:
            dQdx = (ds.variables['QV'][t][:, lat_idx[t], lon_idx_next]
                    - ds.variables['QV'][t][:, lat_idx[t], lon_idx_before]) / profiles['dx'][start:end]
            dQdy = (ds.variables['QV'][t][:, lat_idx[t] + 1, lon_idx[t]]
                    - ds.variables['QV'][t][:, lat_idx[t] - 1, lon_idx[t]]) / profiles['dy']

            profiles['A_QV_h'][:, start:end] = profiles['U'][:, start:end] * dQdx
            + profiles['V'][:, start:end] * dQdy

        with Dataset(filenames['RH']) as ds:
            dRHdx = (ds.variables['RH'][t][:, lat_idx[t], lon_idx_next]
                     - ds.variables['RH'][t][:, lat_idx[t], lon_idx_before]) / profiles['dx'][start:end]
            dRHdy = (ds.variables['RH'][t][:, lat_idx[t] + 1, lon_idx[t]]
                    - ds.variables['RH'][t][:, lat_idx[t] - 1, lon_idx[t]]) / profiles['dy']

            profiles['A_RH_h'][:, start:end] = profiles['U'][:, start:end] * dRHdx
            + profiles['V'][:, start:end] * dRHdy

    logger.info('Save results to files')
    for var in ['A_QV_h', 'A_RH_h', 'U', 'V']:
        profiles_sorted = profiles[var][:, sort_idx]
        nctools.array2D_to_netCDF(
                    profiles_sorted, var, '', (height, range(num_samples_tot)), ('height', 'profile_index'), outnames[var], overwrite=True
                )
    
    
def average_random_profiles(model, run, time_period, variables, num_samples, sample_days, data_dir, **kwargs):
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
    variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'RH', 'W', 'A_QV', 'A_RH', 'DRH_Dt', 'A_RH_h', 'A_QV_h', 'U', 'V']
    variables_2D = ['OLR', 'IWV', 'STOA', 'OLRC', 'STOAC', 'H_tropo', 'IWP']
    extra_variables = ['A_QV', 'A_RH', 'DRH_Dt']
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
            
    logger.info('Calc additional variables: IWP, H_tropo and T_QV')
    # RH tendency 
    if 'DRH_Dt' in extra_variables:
        logger.info('DRH_Dt')
        profiles_sorted['DRH_Dt'] = utils.drh_dt(
            profiles_sorted['TEMP'], 
            profiles_sorted['PRES'], 
            profiles_sorted['QV'],
            profiles_sorted['RH'],
            profiles_sorted['W'],
            height   
        )
    # RH advection
    if 'A_RH' in extra_variables:
        profiles_sorted['A_RH'] = -profiles_sorted['W'] * np.gradient(profiles_sorted['RH'], height, axis=0)

    # spec. hum. advection
    if 'A_QV' in extra_variables:
        profiles_sorted['A_QV'] = -profiles_sorted['W'] * np.gradient(profiles_sorted['QV'], height, axis=0)
    
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
           
    nan_mask = np.isnan(profiles_sorted['lon'])
    
    for var in variables+['lon', 'lat']+extra_variables:
        if var in variables_3D:
            profiles_sorted[var] = profiles_sorted[var][:, np.logical_not(nan_mask)]
        else:
            profiles_sorted[var] = profiles_sorted[var][np.logical_not(nan_mask)]

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
        
        if var == 'PRES':
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

        if var == 'PRES':
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
    outname_perc = f'{model}-{run}_{start_date}-{end_date}_{num_percs}_perc_means_{num_samples}{sample_days_str}_1exp.pkl'
    outname_binned = f'{model}-{run}_{start_date}-{end_date}_bin_means_{num_samples}{sample_days_str}_1exp.pkl'
    
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
    variables_3D = ['TEMP', 'PRES', 'QV', 'QI', 'QC', 'RH', 'W', 'T_QV', 'U', 'V', 'A_RH_h', 'A_QV_h']
    variables_2D = ['OLR', 'IWV', 'STOA', 'OLRC', 'STOAC', 'H_tropo', 'IWP']
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
            
    nan_mask = np.isnan(profiles_sorted['lon'])
    for var in variables+['lon', 'lat', 'IWP', 'H_tropo', 'T_QV']:
        if var in variables_3D:
            profiles_sorted[var] = profiles_sorted[var][:, np.logical_not(nan_mask)]
        else:
            profiles_sorted[var] = profiles_sorted[var][np.logical_not(nan_mask)]
    
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
                
            if var == 'PRES':
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
            
            if var == 'PRES':
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
    
def get_pressure_scaling_parameters(model):
    """
    """
    if model in ['IFS', 'ERA5']:
        
        a = np.array([0, 0, 3.757813, 22.83594, 62.78125, 122.1016, 202.4844, 302.4766, 424.4141, 568.0625, 734.9922, 926.5078, 1143.25, 1387.547, 1659.477, 1961.5, 2294.242, 2659.141, 3057.266, 3489.234, 3955.961, 4457.375, 4993.797, 5564.383, 6168.531, 6804.422, 7470.344, 8163.375, 8880.453, 9617.516, 10370.18, 11133.3, 11901.34, 12668.26, 13427.77, 14173.32, 14898.45, 15596.7, 16262.05, 16888.69, 17471.84, 18006.93, 18489.71, 18917.46, 19290.23, 19608.57, 19874.03, 20087.09, 20249.51, 20361.82, 20425.72, 20442.08, 20412.31, 20337.86, 20219.66, 20059.93, 19859.39, 19620.04, 19343.51, 19031.29, 18685.72, 18308.43, 17901.62, 17467.61, 17008.79, 16527.32, 16026.12, 15508.26, 14975.62, 14432.14, 13881.33, 13324.67, 12766.87, 12211.55, 11660.07, 11116.66, 10584.63, 10065.98, 9562.683, 9076.4, 8608.525, 8159.354, 7727.412, 7311.869, 6911.871, 6526.947, 6156.074, 5798.345, 5452.991, 5119.895, 4799.149, 4490.817, 4194.931, 3911.49, 3640.468, 3381.744, 3135.119, 2900.391, 2677.348, 2465.771, 2265.432, 2076.096, 1897.519, 1729.449, 1571.623, 1423.77, 1285.61, 1156.854, 1037.201, 926.3449, 823.9678, 729.7441, 643.3399, 564.4135, 492.616, 427.5925, 368.9824, 316.4207, 269.5396, 227.9689, 191.3386, 159.2794, 131.4255, 107.4157, 86.89588, 69.52058, 54.95546, 42.87924, 32.98571, 24.98572, 18.60893, 13.60542, 9.746966, 6.827977, 4.666084, 3.102241, 2.000365])
        
        b = np.array([1, 0.99763, 0.995003, 0.991984, 0.9885, 0.984542, 0.980072, 0.975078, 0.969513, 0.963352, 0.95655, 0.949064, 0.94086, 0.931881, 0.922096, 0.911448, 0.8999, 0.887408, 0.873929, 0.859432, 0.843881, 0.827256, 0.809536, 0.790717, 0.770798, 0.749797, 0.727739, 0.704669, 0.680643, 0.655736, 0.630036, 0.603648, 0.576692, 0.549301, 0.521619, 0.4938, 0.466003, 0.438391, 0.411125, 0.384363, 0.358254, 0.332939, 0.308598, 0.285354, 0.263242, 0.242244, 0.222333, 0.203491, 0.185689, 0.16891, 0.153125, 0.138313, 0.124448, 0.111505, 0.099462, 0.088286, 0.077958, 0.068448, 0.059728, 0.051773, 0.044548, 0.038026, 0.032176, 0.026964, 0.022355, 0.018318, 0.014816, 0.011806, 0.009261, 0.007133, 0.005378, 0.003971, 0.002857, 0.001992, 0.001353, 0.00089, 0.000562, 0.00034, 0.000199, 0.000112, 0.000059, 0.000024, 0.000007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if model == 'IFS':
            return a[0:113], b[0:113]
        elif model == 'ERA5':
            return a, b
        
    elif model == 'ARPEGE':
        a = np.array([1222.8013005419998, 1151.081140022, 1047.558699711, 916.5939116075, 770.0440539174999, 622.9077824315, 486.6332968815, 368.10595326199996, 270.203518668, 192.8273379845, 133.998038074, 90.79660478299999, 60.0589440985, 38.817746126, 24.5329218185, 15.1694680875, 9.1799852255, 5.437809149, 3.152733994, 1.7886128665, 0.9924604774999999, 0.5382769345, 0.2851399245, 0.147395167, 0.07427774100000001, 0.036453698, 0.0174054565, 0.008077108, 0.0036395365000000002, 0.001591072, 0.000674332, 0.00027691500000000003, 0.0001101345, 4.24125e-05, 1.5813e-05, 5.7085e-06, 1.996e-06, 6.760000000000001e-07, 2.22e-07, 7.05e-08, 2.15e-08, 6.5e-09, 2e-09, 5e-10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #surface pressure?
        b = np.array([0.033519443, 0.043141488500000005, 0.0538864905, 0.065179446, 0.07655079000000001, 0.087755708, 0.098617487, 0.109037617, 0.11898940250000001, 0.1285059975, 0.1376670545, 0.1465862685, 0.15540085, 0.1642631035, 0.173333806, 0.1827769155, 0.19275514300000002, 0.2034260255, 0.2149382755, 0.2274282975, 0.241016859, 0.2558059515, 0.271875918, 0.289282932, 0.3080569055, 0.3281999095, 0.34968516949999995, 0.3724566975, 0.39642960250000003, 0.4214911215, 0.447502403, 0.47430106299999997, 0.501704532, 0.529514206, 0.5575204085000001, 0.585508159, 0.6132637475, 0.6405821055, 0.6672749535, 0.6931797075, 0.7181691185000001, 0.7421616105, 0.7651322825, 0.787124532, 0.808262252, 0.828762547, 0.848636437, 0.8675177970000001, 0.8851629525, 0.9015718930000001, 0.916744609, 0.9306810925, 0.9433813345, 0.9548453275, 0.9650730654999999, 0.974064542, 0.981819751, 0.988338689, 0.9936213524999999, 0.9979768075]) #surface pressure?

    return a, b
        

    
    
 
