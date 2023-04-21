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
from time import sleep
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from moisture_space import utils
import netCDF_tools as nctools

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def config():
    """ Reads specifications for processing from confing.json and returns a dictionary
    with specifications.
    """
    with open('/mnt/lustre02/work/um0878/users/tlang/work/dyamond/processing/preprocessing/preprocessing_config.json') as handle:
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
    elif model == 'FV3':
        preprocess_FV3(infile, outfile, option_1, numthreads)
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
    
def preprocess_FV3(infiles, outfile, option_gridspec, numthreads, **kwargs):
    """ Perform preprocessing steps for FV3: Combine 6 subtiles of the grid (only needed for variables U and V)
    
    Parameters: 
        infiles (str): Name of input files
        outfile (str): Name of output file
        numthreads (int): Number of OpenMP threads for cdo
    """
    cmd = f'cdo -P {numthreads} -setgrid,{option_gridspec} -collgrid,gridtype=unstructured {infiles} {outfile}'
    
    logger.info(cmd)
    os.system(cmd)
    
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
    
def interpolate_to_pressure_levels(infile, pres_file, surf_pres_file, outfile, pressure_levels, var, numthreads, **kwargs):
    """ Interpolate vertically to common pressure levels.
    
    Parameters:
        infile (str): Path to file containing variable to interpolate
        pres_file (str): Path to file containing pressure for all grid points
        surf_pres_file (str): Path to file containing surface pressure
        outfile (str): Path to output file containing interpolated data
        pressure_levels (array): Pressure levels to interpolate on
        var (str): Variable to interpolate
        numthreads (int): Number of OpenMP threads for cdo 
    """
    pressure_levels_str = ','.join([str(l) for l in pressure_levels])
    cmd = f'cdo --verbose -P {numthreads} -f nc4 -O selvar,{var} -ap2pl,{pressure_levels_str} -merge [ {infile} {pres_file} {surf_pres_file} ] {outfile}'
    
    print(cmd)
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
    
def calc_relative_humidity_ncl(qv_file, temp_file, out_file):
    """ Calculate relative humidity using the ncl function "relhum"
    """
    
    cmd = f'ncl \'filename_q=\"{qv_file}\"\' \'filename_t=\"{temp_file}\"\' \'filename_out=\"{out_file}\"\' /mnt/lustre02/work/mh1126/m300773/OTS/calc_rh.ncl'
    logger.info(cmd)
    os.system(cmd)

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
    units = filelists.get_variable_units()
    
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
        # FV3:
        else:
            pres = ds.variables[pres_name][timestep].filled(np.nan)
            
    # orography
    with Dataset(z0_file) as ds:
        if model in ['IFS', 'ERA5']:
            z0 = ds.variables['z'][0][0].filled(np.nan) / typhon.constants.g
            #z0 = np.ones((len(lat), len(lon))) * 10. #lowest model level at 10 m (first half level)
        elif model == 'FV3':
            z0 = np.ones((len(lat), len(lon))) * 14. # lowest model level at about 14m 
            #z0 = ds.variables['z'][:].filled(np.nan)
        elif model == 'ARPEGE':
            z0 = ds.variables['GH'][timestep][0].filled(np.nan) / typhon.constants.g + 17.# lowest level from geopotential
            #z0 = np.ones((len(lat), len(lon))) * 17. #(lowest model level is at 17 meters above sea level)
            
    # Calculate heights
    logger.info('Calculate heights')
    if model == 'IFS':
        height = np.flipud(pressure2height(pres, np.flipud(temp), z0))
    else:
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
    cmd = f'cdo --verbose -O -P {numthreads} mergetime {infiles_str} {outfile}'
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
    
    
def get_pressure_scaling_parameters(model):
    """
    """
    if model in ['IFS', 'ERA5']:
        
        with Dataset('/mnt/lustre02/work/mh1126/m300773/DYAMOND/IFS/IFS-B_param.nc') as ds:
            b = ds.variables['b_mid'][:].filled(np.nan)[::-1]
        with Dataset('/mnt/lustre02/work/mh1126/m300773/DYAMOND/IFS/IFS-A_param.nc') as ds:
            a = ds.variables['a_mid'][:].filled(np.nan)[::-1]
            
        if model == 'IFS':
            return a[0:113], b[0:113]
        elif model == 'ERA5':
            return a, b
        
    elif model == 'ARPEGE':
        a = np.array([1222.8013005419998, 1151.081140022, 1047.558699711, 916.5939116075, 770.0440539174999, 622.9077824315, 486.6332968815, 368.10595326199996, 270.203518668, 192.8273379845, 133.998038074, 90.79660478299999, 60.0589440985, 38.817746126, 24.5329218185, 15.1694680875, 9.1799852255, 5.437809149, 3.152733994, 1.7886128665, 0.9924604774999999, 0.5382769345, 0.2851399245, 0.147395167, 0.07427774100000001, 0.036453698, 0.0174054565, 0.008077108, 0.0036395365000000002, 0.001591072, 0.000674332, 0.00027691500000000003, 0.0001101345, 4.24125e-05, 1.5813e-05, 5.7085e-06, 1.996e-06, 6.760000000000001e-07, 2.22e-07, 7.05e-08, 2.15e-08, 6.5e-09, 2e-09, 5e-10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #surface pressure?
        b = np.array([0.033519443, 0.043141488500000005, 0.0538864905, 0.065179446, 0.07655079000000001, 0.087755708, 0.098617487, 0.109037617, 0.11898940250000001, 0.1285059975, 0.1376670545, 0.1465862685, 0.15540085, 0.1642631035, 0.173333806, 0.1827769155, 0.19275514300000002, 0.2034260255, 0.2149382755, 0.2274282975, 0.241016859, 0.2558059515, 0.271875918, 0.289282932, 0.3080569055, 0.3281999095, 0.34968516949999995, 0.3724566975, 0.39642960250000003, 0.4214911215, 0.447502403, 0.47430106299999997, 0.501704532, 0.529514206, 0.5575204085000001, 0.585508159, 0.6132637475, 0.6405821055, 0.6672749535, 0.6931797075, 0.7181691185000001, 0.7421616105, 0.7651322825, 0.787124532, 0.808262252, 0.828762547, 0.848636437, 0.8675177970000001, 0.8851629525, 0.9015718930000001, 0.916744609, 0.9306810925, 0.9433813345, 0.9548453275, 0.9650730654999999, 0.974064542, 0.981819751, 0.988338689, 0.9936213524999999, 0.9979768075]) #surface pressure?

    return a, b
        

    
    
 
