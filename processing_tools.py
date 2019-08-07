import numpy as np
import pandas as pd
import os
import glob
import logging
import json
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

def interpolate_horizontally(infile, target_grid, weights, outfile, numthreads, **kwargs):
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
        
    cmd = f'cdo --verbose -O {to_netCDF} -P {numthreads} remap,{target_grid},{weights} {infile} {outfile}'
    logger.info(cmd)
    os.system(cmd)
    
def interpolate_vertically(infile, varname, height_file, target_height_file, outfile, out_varname, **kwargs):
    """ Interpolates field from from height levels given in height_file to new height levels 
    given in target_height_file.
    
    Parameters:
        infile (str): path to input file
        varname (str): name of variable in input file
        height_file (str): path to file containing geometric heights corresponding to model levels
        target_height_file (str): path to file containing target heights
        outfile (str): path to output file containing interpolated field
        out_varname (str): name of variable in output file
    """
    # read variables from files
    with Dataset(infile) as ds:
        lat = ds.variables['lat'][:].filled(np.nan)
        lon = ds.variables['lon'][:].filled(np.nan)
        field = ds.variables[varname][0].filled(np.nan)
        varunit = ds.variables[varname].units

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
                                out_varname, varunit, outfile, overwrite=True)    

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
            'PRES': 'pres'
        }
    elif model == 'NICAM':
        varname = {
            'TEMP': 'ms_tem',
            'QV': 'ms_qv',
            'PRES': 'ms_pres'
        }
    elif model == 'GEOS':
        varname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P'
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
        'PRES': 'Pa'
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
    grid_dir = '/work/ka1081/Hackathon/GrossStats/'
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
                temp = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                out_file = os.path.join(temp_dir, temp)

                raw_files.append(raw_file)
                out_files.append(out_file)

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
                temp = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                out_file = os.path.join(temp_dir, temp)

                raw_files.append(raw_file)
                out_files.append(out_file)

    elif model == 'GEOS':
        var2dirname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P'
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
                
    else:
        logger.error('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
        return None
              
    return raw_files, out_files

#def get_hoursmergingfilelist(model, run, variables, time_period, data_dir, **kwargs):
#    """ Returns a list of filenames of 3-hourly raw DYAMOND output needed to merge these files together before 
#    horizontal interpolation. This is needed for the models GEOS, #FIXME.
#    """
#    
#    time = pd.date_range(time_period[0], time_period[1], freq='1D')
#    out_dir = os.path.join(data_dir, model.upper())
#    in_files = []
#    out_files = []
#    
#    if model == 'GEOS':
#        var2dirname = {
#            'TEMP': 'T',
#            'QV': 'QV',
#            'PRES': 'P'
#        }
#    
#        for v, var in enumerate(variables):
#            print(f'v = {v}')
#            varname = var2dirname[var]
#            if run == '3.0km':
#                stem = f'/mnt/lustre02/work/ka1081/DYAMOND/GEOS-3km/inst/inst_03hr_3d_{varname}_Mv'
#            elif run == '3.0km-MOM':
#                stem = f'/mnt/lustre02/work/ka1081/DYAMOND/GEOS-3km-MOM/inst/inst_03hr_3d_{varname}_Mv'
#            else:
#                print (f'Run {run} not supported for GEOS.\nSupported runs are: "3.0km" and "3.0km-MOM".')    
#            for i in np.arange(time.size):
#                print(f'i = {i}')
#                in_files.append([])
#                print('appended')
#                date_str = time[i].strftime("%m%d")
#                temp = f'GEOS-{run}_{var}_{date_str}.nc'
#                out_file = os.path.join(out_dir, temp)
#                out_files.append(out_file)
#                for h in np.arange(0, 24, 3): 
#                    date_str = time[i].strftime("%Y%m%d")
#                    hour_str = f'{h:02d}'
#                    hour_file = f'DYAMOND.inst_03hr_3d_{varname}_Mv.{date_str}_{hour_str}00z.nc4'
#                    hour_file = os.path.join(stem, hour_file)
#                    in_files[v * time.size + i].append(hour_file)
#    else: 
#         logger.error('The model specified for time merging of 3-hourly output does not exist or has not been implemented yet.') 
#                
#    return in_files, out_files
                


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
            if model == 'GEOS':
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

def latlonheightfield_to_netCDF(height, latitude, longitude, field, varname, varunit, outname, overwrite=True):
    """ Saves a field with dimensions (height, latitude, longitude) to a NetCDF file.
    
    Parameters:
        height (1D-array, dimension nheight): height levels in m
        latitude (1D-array, dimension nlat): latitudes in deg North
        longitude (1D-array, dimension nlon): longitudes in deg East
        field (3D-array, dimensions (nheight, nlat, nlon)): field with variable to save
        varname (string): name of variable (e.g. 't' or 'q')
        varunit (string): unit of variable
        outname (string): name for output file
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

        lat.units = 'degrees north'
        lon.units= 'degrees east'
        zlev.units = 'm'

        lat[:] = latitude[:]
        lon[:] = longitude[:]
        zlev[:] = height[:]

        v = f.createVariable(varname,'f4',dimensions=('zlev','lat','lon'))
        v.units = varunit
        f[varname][:,:,:]=field[:,:,:] 

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


    
    
 
