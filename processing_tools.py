import numpy as np
import pandas as pd
import os
import glob
import logging
import json
import typhon
from cdo import Cdo
from subprocess import call, check_output
from concurrent.futures import ProcessPoolExecutor, as_completed, wait # for launching parallel tasks
from scipy.interpolate import interp1d
from netCDF4 import Dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def config():
    """ Reads specifications for processing from confing.json and returns a dictionary
    with specifications.
    """
    with open('config.json') as handle:
        config = json.loads(handle.read())
    
    return config

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
    filename, file_extension = os.path.splitext(tempfile)
    tempfile_1 = tempfile
    tempfile_2 = filename+'_2'+file_extension
    tempfile_3 = filename+'_3'+file_extension
    
    cmd_1 = f'cdo --eccodes select,name={option_selvar} {infile} {tempfile_1}'
    cmd_2 = f'grib_set -s numberOfVerticalCoordinateValues={option_nzlevs} {tempfile_1} {tempfile_2}'
    cmd_3 = f'rm {tempfile_1}'
    cmd_4 = f'grib_set -s editionNumber=1 {tempfile_2} {tempfile_3}'
    cmd_5 = f'rm {tempfile_2}'
    cmd_6 = f'cdo -P {numthreads} -R -f nc4 copy {tempfile_3} {outfile}'
    
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
    logger.info(cmd_6)
    os.system(cmd_6)
    
def interpolate_horizontally(infile, target_grid, weights, outfile, options, numthreads, **kwargs):
    """ Horizontal Interpolation to another target grid with the CDO command remap 
    that uses pre-calculated interpolation weights.
    
    Parameters:
        infile (str): path to file with input 2D or 3D field
        target_grid (str): path to file with target grid information
        weights (str): path to file with pre-calculated interpolation weights
        outfile (str): output file (full path)
        numthreads (int): number of OpenMP threads for cdo 
    """
    filename, file_extension = os.path.splitext(infile)
    if file_extension == '.nc' or file_extension == '.nc4':
        to_netCDF = ''
    else:
        to_netCDF = '-f nc4'
    
    if options != '':
        options = options + ' '
        
    cmd = f'cdo --verbose -O {to_netCDF} -P {numthreads} remap,{target_grid},{weights} {options}{infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)
    
def calc_relative_humidity(temp_file, qv_file, pres_file, timestep, model, run, time_period, temp_dir, **kwargs):
    """
    """
    varnames = get_modelspecific_varnames(model)
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    temp_name = varnames['TEMP']
    qv_name = varnames['QV']
    if model == 'IFS':
        pres_name = 'SURF_PRES'
    elif model == 'SAM':
        pres_name = 'PP'
    else:
        pres_name = varnames['PRES']
            
    with Dataset(temp_file) as ds:
        temp = ds.variables[temp_name][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    with Dataset(pres_file) as ds:
        if model == 'IFS':
            surf_pres = np.exp(ds.variables[pres_name][timestep].filled(np.nan))
        if model == 'SAM':
            pres_pert = ds.variables[pres_name][timestep].filled(np.nan)
            pres_mean = ds.variables['p'][:].filled(np.nan)
        else:
            pres = ds.variables[pres_name][timestep].filled(np.nan)
    with Dataset(qv_file) as ds:
        qv = ds.variables[qv_name][timestep].filled(np.nan)
    
    if model == 'IFS':
        pres = np.ones(temp.shape) * np.nan
        a, b = get_IFS_pressure_scaling_parameters()
        for la in range(temp.shape[1]):
            for lo in range(temp.shape[2]):            
                pres[:, la, lo] = np.flipud(surf_pres[:, la, lo] * b + a)
    
    if model == 'SAM':
        qv = qv * 1e-3
        pres = np.zeros(pres_pert.shape)
        for i in range(pres_pert.shape[1]):
            for j in range(pres_pert.shape[2]):
                pres[:, i, j] = pres_mean * 1e2 + pres_pert[:, i, j]
                        
    logger.info('Calc VMR')
    vmr = typhon.physics.specific_humidity2vmr(qv)
    logger.info('Calc e_eq')
    e_eq = typhon.physics.e_eq_mixed_mk(temp)
    logger.info('Calc RH')
    rh = vmr * pres / e_eq
    rh = np.expand_dims(rh, axis=0)
    
    height = np.arange(temp.shape[0])
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_RH_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    logger.info('Save file')
    latlonheightfield_to_netCDF(height, lat, lon, rh, 'RH', '[]', outname, time_dim=True, time_var=timestep*3, overwrite=True)
    
def interpolate_vertically(infile, outfile, height_file, target_height_file, variable, **kwargs):
    """ Interpolates field from from height levels given in height_file to new height levels 
    given in target_height_file.
    
    Parameters:
        infile (str): path to input file
        outfile (str): path to output file containing interpolated field
        height_file (str): path to file containing geometric heights corresponding to model levels
        target_height_file (str): path to file containing target heights
        variable (str): Name of variable to be processed (e.g. 'TEMP' or 'PRES')
    """
    # read variables from files
    units = get_variable_units()
    varnames = get_modelspecific_varnames(model)
    varname = varnames[variable]
    with Dataset(infile) as ds:
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        field = ds.variables[varname][0].filled(np.nan)

    with Dataset(target_height_file) as dst:
        target_height = dst.variables['target_height'][:].filled(np.nan)

    with Dataset(height_file) as dsh:
        var_height = dsh.variables['height'][:].filled(np.nan)
    
    nheight, nlat, nlon = field.shape
    ntargetheight = len(target_height)
    field_interp = np.ones((ntargetheight, nlat, nlon)) * np.nan
    
    # interpolate
    for i in range(nlat):
        logger.info(f'Lat: {i} of {nlat}')
        for j in range(nlon):
            field_interp[:, i, j] = interp1d(
                var_height[:, i, j],
                field[:, i, j],
                bounds_error=False,
                fill_value=np.nan)(target_height)
    
    # save interpolated field to netCDF file
    latlonheightfield_to_netCDF(target_height, lat, lon, field_interp,\
                                variable, units[variable], outfile, overwrite=True)  
    
def interpolate_vertically_per_timestep(infiles, outfiles, height_file, target_height_file, timestep, model, variables, **kwargs):
    """ Perform vertical interpolation for one timestep included in the input files. Every input file contains one variable.
    Outupt files contain only one timestep.
    
    Parameters:
        infiles (list of str): List of input files, one for every variable (in the same order as in the variable variables)
        outfiles (list of outfiles): List of output files, one for every variable (output files contain only one timestep)
        height_file (str): File containing geometric heights for every model level for this timestep
        timestep (int): Timestep to process
        model (str): Name of model
        variables (list of str): List of variables to interpolate       
    """
    varnames = get_modelspecific_varnames(model)
    units = get_variable_units()
    
    with Dataset(target_height_file) as dst:
        target_height = dst.variables['target_height'][:].filled(np.nan)
        
    with Dataset(height_file) as dsh:
        heightname = varnames['H']
        var_height = dsh.variables[heightname][timestep].filled(np.nan)
        
    for i, var in enumerate(variables):
        infile = infiles[i]
        outfile = outfiles[i]
        varname = varnames[var]
        
        with Dataset(infile) as ds:
            lat = ds.variables['lat'][:].filled(np.nan)
            lon = ds.variables['lon'][:].filled(np.nan)
            field = ds.variables[varname][timestep].filled(np.nan)
            
        nheight, nlat, nlon = field.shape
        ntargetheight = len(target_height)
        field_interp = np.ones((ntargetheight, nlat, nlon)) * np.nan
            
        # interpolate
        for i in range(nlat):
            logger.info(f'Lat: {i} of {nlat}')
            for j in range(nlon):
                field_interp[:, i, j] = interp1d(
                    var_height[:, i, j],
                    field[:, i, j],
                    bounds_error=False,
                    fill_value=np.nan)(target_height)

        # save interpolated field to netCDF file
        latlonheightfield_to_netCDF(target_height, lat, lon, field_interp,\
                                    var, units[var], outfile, overwrite=True)  
            
def calc_level_pressure_from_surface_pressure_IFS(surf_pres_file, timestep, temp_dir, model, run, time_period, **kwargs):
    """ Calculate pressure at IFS model levels from surface pressure for one timestep contained in 
    surf_pres_file.
    
    Parameters:
        surf_pres_file (str): Path to file that contains (logarithm of) surface pressure for every timestep
        timestep (int): Timestep for which to calculate pressure
        temp_dir (str): Path to directory to save output files
        model (str): Name of model
        run (str): Name of model run
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    with Dataset(surf_pres_file) as ds:
        surf_pres = np.exp(ds.variables['SURF_PRES'][timestep].filled(np.nan))
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
    
    date_str = time[timestep].strftime('%m%d')
    hour_str = time[timestep].strftime('%H')
    outname = f'{model}-{run}_PRES_{date_str}_{hour_str}_hinterp.nc'
    outname = os.path.join(temp_dir, outname)
    a, b = get_IFS_pressure_scaling_parameters()
    pres = np.ones((len(a), surf_pres.shape[1], surf_pres.shape[2])) * np.nan    
    for la in range(surf_pres.shape[1]):
        logger.info(la)
        for lo in range(surf_pres.shape[2]):            
            pres[:, la, lo] = np.flipud(surf_pres[:, la, lo] * b + a)
    logger.info('Save pressure file')
    height = np.arange(pres.shape[0])
    pres = np.expand_dims(pres, axis=0)
    logger.info(pres.shape)
    latlonheightfield_to_netCDF(height, lat, lon, pres, 'PRES', 'Pa', outname, time_dim=True, time_var=timestep*3, overwrite=True)
    
def calc_height_from_pressure_IFS(pres_file, temp_file, z0_file, out_file, timestep, model, temp_dir, **kwargs):
    """ Calculate level heights from level pressure and layer temperature assuming hydrostatic equilibrium.
    """
    varnames = get_modelspecific_varnames(model)
    temp_name = varnames['TEMP']
    pres_name = varnames['PRES']
    
    logger.info('Load data')
    with Dataset(temp_file) as ds:
        temp = ds.variables[temp_name][timestep].filled(np.nan)
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        
    with Dataset(pres_file) as ds:
        if model == 'IFS':
            surf_pres = np.exp(ds.variables[pres_name][timestep].filled(np.nan))
        if model == 'SAM':
            pres_pert = ds.variables[pres_name][timestep].filled(np.nan)
            pres_mean = ds.variables['p'][:].filled(np.nan)
        else:
            pres = ds.variables[pres_name][timestep].filled(np.nan)
    
    with Dataset(z0_file) as ds:
        z0 = ds.variables['height'][-1].filled(np.nan)
    
    logger.info('Calculate heights')
    height = pressure2height(pres, temp, z0)
    h = np.arange(height.shape[0])
    height = np.expand_dims(height, axis=0)
    logger.info('Save file')
    latlonheightfield_to_netCDF(h, lat, lon, height, 'H', 'm', out_file, time_dim=True, time_var=timestep*3, overwrite=True)
    
def pressure2height(p, T, z0=None):
    r"""Convert pressure to height based on the hydrostatic equilibrium.
    .. math::
       z = \int -\frac{\mathrm{d}p}{\rho g}
    Parameters:
        p (ndarray): Pressure [Pa].
        T (ndarray): Temperature [K].
            If ``None`` the standard atmosphere is assumed.
    See also:
        .. autosummary::
            :nosignatures:
            standard_atmosphere
    Returns:
        ndarray: Relative height above lowest pressure level [m].
    """
    if p[0, 0, 0] < p[-1, 0, 0]:
        np.flipud(p)
        np.flipud(T)
         
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

def average_timesteps(infile, outfile, numthreads, **kwargs):
    """
    Calculate temporal average of all time steps contained in one file.
    
    Parameter:
        infile (str): path to input file containing several time steps
        outfile (str): name for file containing time average
        numthreads_timeaverage (int): number of OpenMP threads for cdo
    """
    cmd = f'cdo -O -P {numthreads} timmean {infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)
    
def get_modelspecific_varnames(model):
    """ Returns dictionary with variable names for a specific model.
    
    Parameters:
        model (string): Name of model
    """
    
    if model == 'ICON':
        varname = {
            'TEMP': 't',
            'QV': 'q',
            'PRES': 'pres',
            'RH': 'RH'
        }
    elif model == 'NICAM':
        varname = {
            'TEMP': 'ms_tem',
            'QV': 'ms_qv',
            'PRES': 'ms_pres', 
            'RH': 'RH'
        }
    elif model == 'GEOS':
        varname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P',
            'RH': 'RH',
            'H': 'H'
        }    
    elif model == 'IFS':
        varname = {
            'TEMP': 'TEMP',
            'QV': 'QV',
            'PRES': 'PRES',
            'RH': 'RH'
        }
    elif model == 'SAM':
        varname = {
            'TEMP': 'TABS',
            'QV': 'QV',
            'PRES': 'PRES',
            'RH': 'RH'
        }
    elif model == 'UM':
        varname = {
            'TEMP': 'air_temperature',
            'QV': 'specific_humidity',
            'PRES': 'air_pressure',
            'RH': 'RH'
        }
    elif model == 'MPAS':
        varname = {
            'TEMP': 'temperature',
            'QV': 'qv',
            'PRES': 'pressure',
            'RH': 'RH'
        }

    else:
        print('Modelspecific variable names for Model {model} have not been implemented yet.')
        return
        
    return varname

def get_variable_units():
    """ Returns dictionary with units for different variables. 
    """
    
    varunit = {
        'TEMP': 'K',
        'QV': 'kg kg**-1',
        'PRES': 'Pa',
        'RH': '-'
    }
    
    return varunit        
def get_path2grid(grid_res, **kwargs):
    """ Returns path to file with grid information for a regular lat-lon grid with a
    resolution specified in grid_res. Note that grid files only exist for 0.1 and 0.2
    degrees so far. 
    
    Parameters:
        grid_res (float): grid resolution in degree  
    """
    
    gridname = str(grid_res).ljust(4, '0')
    grid = f'/work/ka1081/Hackathon/GrossStats/{gridname}_grid.nc'
    
    if not os.path.isfile(grid):
        logger.warning('Warning: A grid file for this resolution does not exist yet.')
    
    return grid

def get_path2weights(model, run, grid_res, **kwargs):
    """ Returns path to file containing pre-calculated interpolation weights for a given
    model run and a given target grid resolution.
    
    Parameter:
        model (str): name of model
        run (str): name of model run
        grid_res (float): resolution of target grid in degrees.
    """
    grid_dir = '/work/ka1081/DYAMOND/PostProc/GridsAndWeights'
    if model == 'ICON':
        if run == '5.0km_1':
            weights = 'ICON_R2B09_0.10_grid_wghts.nc'
        elif run == '5.0km_2':
            weights = 'ICON_R2B09-mpi_0.10_grid_wghts.nc'
        elif run == '2.5km':
            weights = 'ICON_R2B10_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "5.0km_1", "5.0km_2", "2.5km".')
            return None
        
    elif model == 'NICAM':
        if run == '3.5km':
            weights = 'NICAM-3.5km_0.10_grid_wghts.nc'
        elif run == '7.0km':
            weights = 'NICAM-7.0km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "3.5km" and "7.0km".')
            return None
    
    elif model == 'GEOS':
        if run == '3.0km' or run == '3.0km-MOM':
            weights = 'GEOS-3.25km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "3.0km" and "3.0km-MOM".')
    
    elif model == 'IFS':
        if run == '4.0km':
            weights = 'ECMWF-4km_0.10_grid_wghts.nc'
        elif run == '9.0km':
            weights = 'ECMWF-9km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "4.0km" and "9.0km".') 
            
    elif model == 'SAM':
        if run == '4.0km':
            weights = 'SAM_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "4.0km".')
    
    elif model == 'UM':
        if run == '5.0km':
            weights = 'UM-5km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "5.0km".')
            
    
    elif model == 'MPAS':
        if run == '3.75km':
            weights = 'MPAS-3.75km_0.10_grid_wghts.nc'
        elif run == '7.5km':
            weights = 'MPAS-7.5km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "3.75km" and "7.5km".')
            

    # other models...
    else:
        logger.error('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
        return None
    
    path2weights = os.path.join(grid_dir, weights)
    if not os.path.isfile(path2weights):
        logger.warning('Warning: There are no pre-calculated weights for this model run and target grid yet.')

    return path2weights

def get_path2heightfile(model, grid_res, **kwargs):
    """ Returns path to file containing geometrical heights corresponding to model levels for a given model.
   
    Parameters:
        model (str): name of model
        grid_res (num): resolution of grid in degrees
    """
    gridname = str(grid_res).ljust(4, '0')
    return f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{model}-{gridname}deg_heights.nc'

def get_interpolationfilelist(model, run, variables, time_period, temp_dir, **kwargs):
    """ Returns list of all raw output files from a specified DYAMOND model run corresponding to a given
    time period and given variables. A list of filenames for output files needed e.g. for
    horizontal interpolation is also returned.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period in the format YYYY-mm-dd
        temp_dir (str): path to directory for output files
    """
    
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    raw_files = []
    out_files = []
    options = []
        
    if model == 'ICON':
        # dictionary containing endings of filenames containing the variables
        var2suffix = {
            'TEMP': 'atm_3d_t_ml_',
            'QV': 'atm_3d_qv_ml_',
            'PRES': 'atm_3d_pres_ml_'
                }
        for var in variables:
            # filenames of raw files, grid weights and the output file
            suffix = var2suffix[var]
            if (run=='5.0km_1'):
                stem   = '/work/ka1081/DYAMOND/ICON-5km/nwp_R2B09_lkm1006_{}'.format(suffix)
            elif (run=='5.0km_2'):
                stem   = '/work/ka1081/DYAMOND/ICON-5km_2/nwp_R2B09_lkm1013_{}'.format(suffix)
            elif (run=='2.5km'):
                stem   = '/work/ka1081/DYAMOND/ICON-2.5km/nwp_R2B10_lkm1007_{}'.format(suffix)
            else:
                print (f'Run {run} not supported for {model}.\nSupported runs are: "5.0km_1", "5.0km_2", "2.5km".')
                exit

            for i in np.arange(time.size):
                raw_file = stem + time[i].strftime("%Y%m%d") + 'T000000Z.grb'
                time_str = time[i].strftime("%m%d")
                out_file = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                out_file = os.path.join(temp_dir, out_file)

                raw_files.append(raw_file)
                out_files.append(out_file)
                options.append('')

    elif model == 'NICAM':
        # dictionary containing filenames containing the variables
        var2filename = {
            'TEMP': 'ms_tem.nc',
            'QV': 'ms_qv.nc',
            'PRES': 'ms_pres.nc'
        }
                
        for var in variables:
            # paths to raw files
            filename = var2filename[var]
            if (run=='3.5km'):
                stem   = '/work/ka1081/DYAMOND/NICAM-3.5km'
            elif (run=='7.0km'):
                stem   = '/work/ka1081/DYAMOND/NICAM-7km'
            else:
                print (f'Run {run} not supported for {model}.\nSupported runs are: "3.5km" and "7.0km".')
                exit

            # append filenames for this variable to raw_files and out_files
            for i in np.arange(time.size):
                dayfolder = glob.glob(os.path.join(stem, time[i].strftime("%Y%m%d")+'*'))[0]
                raw_file = os.path.join(dayfolder, filename)
                time_str = time[i].strftime("%m%d")
                out_file = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                out_file = os.path.join(temp_dir, out_file)

                raw_files.append(raw_file)
                out_files.append(out_file)
                options.append('')

    elif model == 'GEOS':
        var2dirname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P', 
            'H': 'H'
        }
        # GEOS output is one file per time step (3-hourly)
        for var in variables:
            varname = var2dirname[var]
            if run == '3.0km':
                stem = f'/mnt/lustre02/work/ka1081/DYAMOND/GEOS-3km/inst/inst_03hr_3d_{varname}_Mv'
            else:
                print (f'Run {run} not supported for {model}.\nSupported runs are: "3.0km".')  
                
            for i in np.arange(time.size):
                for h in np.arange(0, 24, 3):
                    date_str = time[i].strftime("%Y%m%d")
                    hour_str = f'{h:02d}'
                    hour_file = f'DYAMOND.inst_03hr_3d_{varname}_Mv.{date_str}_{hour_str}00z.nc4'
                    hour_file = os.path.join(stem, hour_file)
                    date_str = time[i].strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(hour_file)
                    out_files.append(out_file)
                    options.append('')
#        for var in variables:
#            if run == '3.0km' or run == '3.0km-MOM':
#                stem = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/GEOS'
#            else:
#                print (f'Run {run} not supported for {model}.\nSupported runs are: "3.0km" and "3.0km-MOM".')    
#            for i in np.arange(time.size):
#                date_str = time[i].strftime("%m%d")
#                day_file = f'GEOS-{run}_{var}_{date_str}.nc'
#                day_file = os.path.join(stem, day_file)
#                temp = f'{model}-{run}_{var}_{date_str}_hinterp.nc'
#                out_file = os.path.join(temp_dir, temp)
#                raw_files.append(day_file)
#                out_files.append(out_file)

    elif model == 'IFS':
        stem = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/IFS'
        var2varnumber = {
            'TEMP': 130,
            'QV': 133,
            'SURF_PRES': 152            
        }
        var2unit = {
            'TEMP': 'K',
            'QV': 'kg kg^-1',
            'SURF_PRES': 'Pa'
        }
    
        for var in variables:
            option = f'-setattribute,{var}@unit={var2unit[var]} -chname,var{var2varnumber[var]},{var}'
            for i in np.arange(time.size):
                for h in np.arange(0, 24, 3):
                    date_str = time[i].strftime("%m%d")
                    hour_str = f'{h:02d}'
                    hour_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    hour_file = os.path.join(stem, hour_file)
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(hour_file)
                    out_files.append(out_file)
                    options.append(option)
                    
    elif model == 'SAM':
        ref_date = pd.Timestamp('2016-08-01-00')
        timeunit_option = '-settunits,days' 
        grid_option = '-setgrid,/mnt/lustre02/work/ka1081/DYAMOND/SAM-4km/OUT_2D/DYAMOND_9216x4608x74_7.5s_4km_4608_0000002400.CWP.2D.nc'
        var2filename = {
            'TEMP': 'TABS',
            'QV': 'QV',
            'PRES': 'PP'
        }
        
        for var in variables:
            stem = '/mnt/lustre02/work/ka1081/DYAMOND/SAM-4km/OUT_3D'
            for i in np.arange(time.size):
                for h in np.arange(0, 24, 3):
                    date_str = time[i].strftime('%Y-%m-%d')
                    hour_str = f'{h:02d}'
                    
                    timeaxis_option = f'-settaxis,{date_str},{hour_str}:00:00,3h'
                    option = ' '.join([timeaxis_option, timeunit_option, grid_option])
                    
                    timestamp = pd.Timestamp(f'{date_str}-{hour_str}')
                    secstr = int((timestamp - ref_date).total_seconds() / 7.5)
                    secstr = f'{secstr:010}'
                    hour_file = f'DYAMOND_9216x4608x74_7.5s_4km_4608_{secstr}_{var2filename[var]}.nc'
                    hour_file = os.path.join(stem, hour_file)
                    date_str = time[i].strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)

                    raw_files.append(hour_file)
                    out_files.append(out_file)
                    options.append(option)
                    
    elif model == 'UM':
        stem = '/mnt/lustre02/work/ka1081/DYAMOND/UM-5km'
        
        var2dirname = {
            'TEMP': 'ta',
            'QV': 'hus',
            'PRES': 'phalf'
        }
        
        for var in variables:
            for i in np.arange(time.size):
                date_str = time[i].strftime('%Y%m%d')
                varstr = var2dirname[var]
                day_file = f'{varstr}_3hr_HadGEM3-GA71_N2560_{date_str}.nc'
                day_file = os.path.join(stem, varstr, day_file)
                
                date_str = time[i].strftime('%m%d')
                out_file = f'{model}-{run}_{var}_{date_str}_hinterp.nc'
                out_file = os.path.join(temp_dir, out_file)
                
                raw_files.append(day_file)
                out_files.append(out_file)
                options.append('')
        

    elif model == 'MPAS':
        if run == '3.75km':
            stem = '/mnt/lustre02/work/ka1081/DYAMOND/MPAS-3.75km'
        elif run == '7.5km':
            stem = '/mnt/lustre02/work/ka1081/DYAMOND/MPAS-7.5km'
        else:
            print (f'Run {run} not supported for {model}.\nSupported runs are: "3.75km" and "7.5km".')
        
        varnames = get_modelspecific_varnames(model)
        for var in variables:
            option = f'-setgrid,mpas:/work/ka1081/DYAMOND/PostProc/GridsAndWeights/MPAS_x1.41943042.grid.nc -selgrid,1 -selname,{varnames[var]}'
            for i in np.arange(time.size):
                for h in np.arange(0, 24, 3):
                    date_str = time[i].strftime("%Y-%m-%d")
                    hour_str = f'{h:02d}'
                    hour_file = f'history.{date_str}_{hour_str}.00.00.nc'
                    hour_file = os.path.join(stem, hour_file)
                    date_str = time[i].strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(hour_file)
                    out_files.append(out_file)
                    options.append(option)
                        
    else:
        logger.error('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
        return None
              
    return raw_files, out_files, options
                
def get_preprocessingfilelist(model, run, variables, time_period, temp_dir, data_dir, **kwargs):
    """ Returns list of all raw output files from a specified DYAMOND model run corresponding to a given
    time period and given variables. A list of filenames for output files needed for
    pre-processing of these models is also returned.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period in the format YYYY-mm-dd
        temp_dir (str): path to directory for output files
        data_dir (str): path to save output files
    """
    
    infile_list = []
    tempfile_list = []
    outfile_list = []
    option_1_list = []
    option_2_list = []
    
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    ref_time = pd.date_range('2016-08-01 00:00:00', '2018-09-10 00:00:00', freq='3h')
    hour_start = np.where(ref_time == time[0])[0][0] * 3
    
    if model == 'IFS':
        var2filename = {
            'TEMP': 'gg_mars_out_ml_upper_sh',
            'QV': 'mars_out_ml_moist',
            'SURF_PRES': 'gg_mars_out_sfc_ps_orog'
        }
        var2variablename = {
            'TEMP': 't',
            'QV': 'q',
            'SURF_PRES': 'lnsp'
        }
        var2numzlevels = {
            'TEMP': 113,
            'QV': 113,
            'SURF_PRES': 1
        }
        

        if (run=='4.0km'):
            stem   = '/work/ka1081/DYAMOND/IFS-4km'
        elif (run=='9.0km'):
            stem   = '/work/ka1081/DYAMOND/IFS-9km'
            
        for var in variables:
            hour_con = hour_start - 3
            option_1 = f'{var2variablename[var]}'
            option_2 = f'{var2numzlevels[var]}'
            for i in np.arange(time.size-1):
                for h in np.arange(0, 24, 3):
                    hour_con = hour_con + 3
                    filename = var2filename[var]
                    in_file = f'{filename}.{hour_con}'
                    in_file = os.path.join(stem, in_file)
                    if var == 'TEMP' or var == 'SURF_PRES':
                        in_file = in_file+'.grib'
                    
                    date_str = time[i].strftime("%m%d")
                    hour_str = f'{h:02d}'
                    temp_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.grb'
                    temp_file = os.path.join(temp_dir, temp_file)
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                    out_file = os.path.join(data_dir, model, out_file)
                    
                    infile_list.append(in_file)
                    tempfile_list.append(temp_file)
                    outfile_list.append(out_file)
                    option_1_list.append(option_1)
                    option_2_list.append(option_2)
    else:
        logger.error('The model specified for preprocessing does not exist or has not been implemented yet.')
        
    return infile_list, tempfile_list, outfile_list, option_1_list, option_2_list
                    

def get_mergingfilelist(model, run, variables, time_period, temp_dir, data_dir, **kwargs):
    """ Returns a list of filenames of horizontally interpolated DYAMOND output needed for time merging.
    For each variable, the list contains a list of filenames. 
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variables
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        temp_dir (str): path to directory with temporary files
        data_dir (str): path to directory to save files        
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    infile_list = []
    outfile_list = []
    
    for v, var in enumerate(variables):
        infile_list.append([])
        outfile_name = os.path.join(data_dir, model.upper(), f'{model}-{run}_{var}_hinterp_merged.nc')
        outfile_list.append(outfile_name)
        for i in np.arange(time.size):
            if model == 'GEOS' or model == 'IFS' or model == 'SAM' or var == 'RH':
                for h in np.arange(0, 24, 3):
                    infile_name = os.path.join(temp_dir, f'{model}-{run}_{var}_{time[i].strftime("%m%d")}_{h:02d}_hinterp.nc')
                    infile_list[v].append(infile_name)    
            else:
                infile_name = os.path.join(temp_dir, f'{model}-{run}_{var}_{time[i].strftime("%m%d")}_hinterp.nc')
                infile_list[v].append(infile_name)
     
    return infile_list, outfile_list

def get_averagingfilelist(model, run, variables, data_dir, **kwargs):
    """ Returns list of filenames of horizontally interpolated and merged as well as 
    list of filenames of horizontally interpolated, merged and temporally averaged 
    DYAMOND output files needed for time averaging. 
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variables
        data_dir (str): path to directory to save files         
    """
    infile_list = []
    outfile_list = []
    
    for var in variables:
        infile_name = os.path.join(data_dir, model.upper(), f'{model}-{run}_{var}_hinterp_merged.nc')
        outfile_name = os.path.join(data_dir, model.upper(), f'{model}-{run}_{var}_hinterp_timeaverage.nc')
        infile_list.append(infile_name)
        outfile_list.append(outfile_name)
        
    return infile_list, outfile_list

def get_vinterpolationfilelist(model, run, variables, data_dir, **kwargs):
    """ Returns list of filenames of horizontally interpolated and temporally averaged as well as 
    list of filenames of horizontally interpolated, temporally averaged and vertically averaged
    DYAMOND output files needed for vertical interpolation.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variables
        data_dir (str): path to directory to save files    
    """
    infile_list = []
    outfile_list = []
    
    for var in variables:
        infile_name = os.path.join(data_dir, model.upper(), f'{model}-{run}_{var}_hinterp_timeaverage.nc')
        outfile_name = os.path.join(data_dir, model.upper(), f'{model}-{run}_{var}_hinterp_timeaverage_vinterp.nc')
        infile_list.append(infile_name)
        outfile_list.append(outfile_name)
        
    return infile_list, outfile_list

def get_vinterpolation_per_timestep_filelist(model, run, variables, time_period, data_dir, **kwargs):
    """
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    infile_list = []
    outfile_list = []
    heightfile = f'{model}-{run}_H_hinterp_merged.nc'
    heightfile = os.path.join(data_dir, model, heightfile)
    
    for var in variables:
        infile = f'{model}-{run}_{var}_hinterp_merged.nc'
        infile = os.path.join(data_dir, model, infile)
        infile_list.append(infile)
        #outfile_sublist.append(f'{model}-{run}_{var}_hinterp_merged_vinterp.nc')
    for i in np.arange(time.size):
        date_str = time[i].strftime("%m%d")
        for h in np.arange(0, 24, 3):
            hour_str = f'{h:02d}'
            outfile_sublist = []
            for var in variables:
                outfile = f'{model}-{run}_{var}_hinterp_vinterp_{date_str}_{hour_str}.nc'
                outfile = os.path.join(data_dir, model, outfile)
                outfile_sublist.append(outfile)
            outfile_list.append(outfile_sublist)
    
    return infile_list, outfile_list, heightfile

def get_height_calculation_filelist(model, run, variables, time_period, data_dir, **kwargs):
    """
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    outfile_list = []
    temp_file = f'{model}-{run}_TEMP_hinterp_merged.nc'
    temp_file = os.path.join(data_dir, model, temp_file)
    pres_file = f'{model}-{run}_PRES_hinterp_merged.nc'
    pres_file = os.path.join(data_dir, model, pres_file)
        
    for i in np.arange(time.size):
        date_str = time[i].strftime("%m%d")
        for h in np.arange(0, 24, 3):
            hour_str = f'{h:02d}'
            for var in variables:
                outfile = f'{model}-{run}_H_hinterp_{date_str}_{hour_str}.nc'
                outfile = os.path.join(data_dir, model, outfile)
                outfile_list.append(outfile)
    
    return pres_file, temp_file, outfile_list
    

def get_rhcalculation_filelist(model, run, data_dir, **kwargs):
    """ Returns filelist needed to calculate the relative humidity for every timestep.
    """
    
    infile_list = []
    
    variables = ['TEMP', 'QV', 'PRES']
    for var in variables:
        if model == 'IFS' and var == 'PRES':
            file = f'{model}-{run}_SURF_{var}_hinterp_merged.nc'
        else:
            file = f'{model}-{run}_{var}_hinterp_merged.nc'
        file = os.path.join(data_dir, model, file)
        infile_list.append(file)
    
    return infile_list

def get_IFS_pressure_scaling_parameters():
    a = np.array([0,0,3.757813,22.835938,62.78125,122.101563,202.484375,302.476563,424.414063,568.0625,734.992188,926.507813,1143.25,1387.546875,1659.476563,1961.5,2294.242188,2659.140625,3057.265625,3489.234375,3955.960938,4457.375,4993.796875,5564.382813,6168.53125,6804.421875,7470.34375,8163.375,8880.453125,9617.515625,10370.17578,11133.30469,11901.33984,12668.25781,13427.76953,14173.32422,14898.45313,15596.69531,16262.04688,16888.6875,17471.83984,18006.92578,18489.70703,18917.46094,19290.22656,19608.57227,19874.02539,20087.08594,20249.51172,20361.81641,20425.71875,20442.07813,20412.30859,20337.86328,20219.66406,20059.93164,19859.39063,19620.04297,19343.51172,19031.28906,18685.71875,18308.43359,17901.62109,17467.61328,17008.78906,16527.32227,16026.11523,15508.25684,14975.61523,14432.13965,13881.33106,13324.66895,12766.87305,12211.54785,11660.06738,11116.66211,10584.63184,10065.97852,9562.682617,9076.400391,8608.525391,8159.354004,7727.412109,7311.869141,6911.870605,6526.946777,6156.074219,5798.344727,5452.990723,5119.89502,4799.149414,4490.817383,4194.930664,3911.490479,3640.468262,3381.743652,3135.119385,2900.391357,2677.348145,2465.770508,2265.431641,2076.095947,1897.519287,1729.448975,1571.622925,1423.770142,1285.610352,1156.853638,1037.201172,926.34491,823.967834,729.744141,643.339905])
    b = np.array([1,0.99763,0.995003,0.991984,0.9885,0.984542,0.980072,0.975078,0.969513,0.963352,0.95655,0.949064,0.94086,0.931881,0.922096,0.911448,0.8999,0.887408,0.873929,0.859432,0.843881,0.827256,0.809536,0.790717,0.770798,0.749797,0.727739,0.704669,0.680643,0.655736,0.630036,0.603648,0.576692,0.549301,0.521619,0.4938,0.466003,0.438391,0.411125,0.384363,0.358254,0.332939,0.308598,0.285354,0.263242,0.242244,0.222333,0.203491,0.185689,0.16891,0.153125,0.138313,0.124448,0.111505,0.099462,0.088286,0.077958,0.068448,0.059728,0.051773,0.044548,0.038026,0.032176,0.026964,0.022355,0.018318,0.014816,0.011806,0.009261,0.007133,0.005378,0.003971,0.002857,0.001992,0.001353,0.00089,0.000562,0.00034,0.000199,0.000112,0.000059,0.000024,0.000007,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return a, b
        
def latlonheightfield_to_netCDF(height, latitude, longitude, field, varname, varunit, outname, time_dim=False, time_var=None, overwrite=True):
    """ Saves a field with dimensions (height, latitude, longitude) to a NetCDF file.
    
    Parameters:
        height (1D-array, dimension nheight): height levels in m
        latitude (1D-array, dimension nlat): latitudes in deg North
        longitude (1D-array, dimension nlon): longitudes in deg East
        field (3D-array, dimensions (nheight, nlat, nlon)): field with variable to save
        varname (string): name of variable (e.g. 't' or 'q')
        varunit (string): unit of variable
        outname (string): name for output file
        time (bool): if True, a dimension for time (with length 1) is added
        overwrite (boolean): if True, exisiting files with the same filename are overwritten
    """
    if (os.path.isfile(outname)) and overwrite:
        os.remove(outname)
    elif (os.path.isfile(outname)) and overwrite==False:
        print('File {} exists. Not overwriting.'.format(outname))
        return

    with Dataset(outname, 'w') as f:
        lat = f.createDimension('lat', len(latitude))
        lon = f.createDimension('lon', len(longitude))
        zlev = f.createDimension('zlev', len(height))

        lat = f.createVariable('lat','f4',('lat',))
        lon = f.createVariable('lon','f4',('lon',))
        zlev = f.createVariable('zlev','f4',('zlev',))
        
        if time_dim:
            time = f.createDimension('time', 1)
        if time_var is not None:
            time = f.createVariable('time', 'f4', ('time',))

        lat.units = 'degrees north'
        lon.units= 'degrees east'
        zlev.units = 'm'

        lat[:] = latitude[:]
        lon[:] = longitude[:]
        zlev[:] = height[:]
        if time_var is not None:
            time.units = 'hours'
            time = time_var
        
        if time_dim:
            v = f.createVariable(varname,'f4',dimensions=('time','zlev','lat','lon'))
            f[varname][:,:,:,:]=field[:,:,:,:]

            
        else:
            v = f.createVariable(varname,'f4',dimensions=('zlev','lat','lon'))
            f[varname][:,:,:]=field[:,:,:] 
        v.units = varunit
        

def vector_to_netCDF(vector, varname, varunit, dimensionvar, dimensionname, outname, overwrite=True):
    """ Saves a 1D-Array to a NetCDF file.
    
    Parameters:
        vector (1D Array): Vector with variable to save
        varname (string): Name of variable to save
        varunit (string): Unit of variable to save
        dimensionvar (1D Array): Grid the variable is given on (e.g. latitudes, longitudes, altitudes) 
        dimensionname (string): Name of dimension associated with vector (e.g. 'lat', 'lon', 'height')
        outname (string): Name of output file
        overwrite (boolean): If True, exisiting files with the same filename are overwritten
        
    """
    if (os.path.isfile(outname)) and overwrite:
        os.remove(outname)
    elif (os.path.isfile(outname)) and overwrite==False:
        print('File {} exists. Not overwriting.'.format(outname))
        return
    
    with Dataset(outname, 'w') as f:
        print(dimensionvar)
        dim = f.createDimension(dimensionname, len(dimensionvar))
        dimvar = f.createVariable(dimensionname, 'f4', (dimensionname,))    
        dimvar[:] = dimensionvar[:]

        f.createVariable(varname, 'f4', dimensions=(dimensionname))
        f[varname][:] = vector[:]


    
    
 
