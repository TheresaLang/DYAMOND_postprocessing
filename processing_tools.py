import numpy as np
import pandas as pd
import os
import glob
#import multiprocessing as mp
from cdo import Cdo
from subprocess import call, check_output
from concurrent.futures import ProcessPoolExecutor, as_completed, wait # for launching parallel tasks
from scipy.interpolate import interp1d
from netCDF4 import Dataset


def horizontal_interpolation(model, run, variable, grid_res, time_period, outfile, temp_dir='/scratch/m/m300773/'):
    """ Interpolate raw model output to a regular lat-lon grid using the SCRIP method. For Details, see
    
    P.W. Jones (1999), First- and Second-Order Conservative Remapping Schemes for Grids in Spherical Coordinates,
    Monthly Weather Review (127), 2204-2210, doi.org/10.1175/1520-0493%281999%29127%3C2204%3AFASOCR%3E2.0.CO%3B2.
    
    Parameters:
    model (string): Model
    run (string): Model run (e.g. '2.5km' or '5.0km_2')
    variables (string): Model output variable to process
    grid_res (string): Resolution of target lat-lon grid (e.g. 0.1 for 0.1 deg resolution)
    time_period (list of strings): Start and end time of time period in the format YYYY-mm-dd
    temp_dir (string): Path to directory to save temporary files
    """
    # if outfile exists, do not overwrite
    if (os.path.isfile(outfile)): 
        print (outfile + ' exists, not overwriting.')
        return
    
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    gridname = str(grid_res).ljust(4, '0')
    grid = '/work/ka1081/Hackathon/GrossStats/{}_grid.nc'.format(gridname)
    
    cdo = Cdo()
    # max. number of processes used for execution
    nprocs = 15
    pool   = ProcessPoolExecutor(nprocs)
    tasks = []
    
    if model == 'ICON':
        # dictionary containing endings of filenames containing the variables
        var2suffix = {
            'TEMP': 'atm_3d_t_ml_',
            'QV': 'atm_3d_qv_ml_',
            'PRES': 'atm_3d_pres_ml_'
                }
        
        # filenames of raw files, grid weights and the output file
        suffix   = var2suffix[variable]
        if (run=='5.0km_1'):
            weight   = '/work/ka1081/Hackathon/GrossStats/ICON_R2B09_0.10_grid_wghts.nc'
            stem   = '/work/ka1081/DYAMOND/ICON-5km/nwp_R2B09_lkm1006_{}'.format(suffix)
        elif (run=='5.0km_2'):
            weight   = '/work/ka1081/Hackathon/GrossStats/ICON_R2B09-mpi_0.10_grid_wghts.nc'
            stem   = '/work/ka1081/DYAMOND/ICON-5km_2/nwp_R2B09_lkm1013_{}'.format(suffix)
        elif (run=='2.5km'):
            weight   = '/work/ka1081/Hackathon/GrossStats/ICON_R2B10_0.10_grid_wghts.nc'
            stem   = '/work/ka1081/DYAMOND/ICON-2.5km/nwp_R2B10_lkm1007_{}'.format(suffix)
        else:
            print ('Run' + run + 'not supported.\nSupported runs are: "5.0km_1", "5.0km_2", "2.5km".')
            exit

        # interpolate fields associated with all specified time steps and write to temporary files
        for i in np.arange(time.size):
            raw_file = stem  + time[i].strftime("%Y%m%d") + 'T000000Z.grb'
            temp = '{}-{}_{}_{}_{}.nc'.format(model, run, variable, gridname, time[i].strftime("%m%d"))
            temp_file = os.path.join(temp_dir, temp)
            # option -O: existing files are overwritten
            arg  = 'cdo -O -P 2 -f nc4 remap,{},{} {} {}'.format(grid, weight, raw_file, temp_file)
            print(arg)
            task = pool.submit(call, arg, shell=True)
            tasks.append(task)                      
        wait(tasks)
    
    elif model == 'NICAM':
        # dictionary containing endings of filenames containing the variables
        var2filename = {
            'TEMP': 'ms_tem.nc',
            'QV': 'ms_qv.nc',
            'PRES': 'ms_pres.nc'
        }
            
        # filenames of raw files, grid weights and the output file
        filename = var2filename[variable]
        if (run=='3.5km'):
            weight   = '/work/ka1081/Hackathon/GrossStats/NICAM-3.5km_0.10_grid_wghts.nc'
            stem   = '/work/ka1081/DYAMOND/NICAM-3.5km'
        elif (run=='7.0km'):
            weight   = '/work/ka1081/Hackathon/GrossStats/NICAM-7.0km_0.10_grid_wghts.nc'
            stem   = '/work/ka1081/DYAMOND/NICAM-7km'
        else:
            print ('Run' + run + 'not supported.\nSupported runs are: "3.5km" and "7.0km".')
            exit
            
        # interpolate fields associated with all specified time steps and write to temporary files
        for i in np.arange(time.size):
            dayfolder = glob.glob(os.path.join(stem, time[i].strftime("%Y%m%d")+'*'))[0]
            raw_file = os.path.join(dayfolder, filename)
            temp = '{}-{}_{}_{}_{}.nc'.format(model, run, variable, gridname, time[i].strftime("%m%d"))
            temp_file = os.path.join(temp_dir, temp)
            # option -O: existing files are overwritten
            arg  = 'cdo -O -P 2 remap,{},{} {} {}'.format(grid, weight, raw_file, temp_file)
            print(arg)
            task = pool.submit(call, arg, shell=True)
            tasks.append(task)                      
        wait(tasks)

    # other models ...
    else:
        print('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
        return

    # merge all timesteps into one file
    print('merge time...')
    cdo.mergetime(
        input='{}{}-{}_{}_{}_????.nc'.format(temp_dir, model, run, variable, gridname),
        output=outfile, options='-P 4'
    )
    # remove temporary files
    os.system('{}{}-{}_{}_{}_????.nc'.format(temp_dir, model, run, variable, gridname))
                

    
def temporal_averaging(infile, outfile, overwrite=False):
    """ Performs a temporal averaging of all timesteps contained in one file.
    
    Parameters:
        infile (string): full path to file containing fields that should be averaged
        outfile (string): full path to output file containing the averaged field
        overwrite (boolean): If true, existing files will be overwritten
    """
    
    if overwrite == False and os.path.isfile(outfile):
        print('file {} exists, not overwriting.'.format(outfile))
    else:
        cdo = Cdo()
        cdo.timmean(input=infile, output=outfile, options='-P 8')
        
def vertical_interpolation(infile, outfile, model, variable, overwrite=False):
    """ Performs a vertical interpolation for a 3D model field (that is already interpolated
        horizontally to a regular latitude-longitude grid. 
        
        Parameters:
            infile (string): full path to file containing field that should be vertically interpolated
            outfile (string): full path to output file containing interpolated field
            model (string): model that produced the data
            variable (string): variable that is interpolated
    """
    
    if overwrite == False and os.path.isfile(outfile):
        print('file {} exists, not overwriting.'.format(outfile))
    else:
        varname = get_modelspecific_names(model)    
        varunit = get_variable_units()
        targetheight_file = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/ICON/target_height.nc'
        
        with Dataset(infile) as ds:
            lat = ds.variables['lat'][:].filled(np.nan)
            lon = ds.variables['lon'][:].filled(np.nan)
            field = ds.variables[varname[variable]][0].filled(np.nan)

        with Dataset(targetheight_file) as dst:
            target_height = dst.variables['target_height'][:].filled(np.nan)
        
        if model == 'ICON':
            # path to file containing heights for every grid cell and level
            heightfile = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/ICON/ICON-5.0km_2_Z.0.10deg.nc'
            heightname = 'HHL'

            with Dataset(heightfile) as dsh:
                height = dsh.variables[heightname][0].filled(np.nan)
            
            # variables like temperature are given on half levels in ICON
            var_levels = height[14:, :, :] + np.abs(np.diff(height[13:, :, :], axis=0)) * 0.5

        elif model == 'NICAM':
            heightfile = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/NICAM/geometric_height_of_model_levels.nc'
            heightname = 'zlev'
            
            # flip height levels and field so that the first entry corresponds to the uppermost level (as in ICON)
            with Dataset(heightfile) as dsh:
                height = height = dsh.variables[heightname][::-1].filled(np.nan)
            field = np.flip(field, axis=0)
            
            #FIXME: Are variables like temperature, qv etc. given on full or half levels in NICAM?
            var_levels = height 
                    
        # split array into chunks
        nchunks = 10
        chunkaxis = 2
        field_chunks = np.split(field, nchunks, axis=chunkaxis)
        # split height levels into chunks, if they are 3D
        if len(var_levles.shape) == 3:
            var_levels_chunks = np.split(var_levels, 10, axis=2)
        # if they are 1D, they do not have to be split
        else:
            var_levels_chunks = var_levels
        
        # collect all data needed for interpolation in a list consisting of dictionaries
        interpolation_arguments = get_interpolation_arguments(field_chunks, var_levels_chunks, target_height) 
        # perform interpolation for chunks in parallel
        with ProcessPoolExecutor(15) as pool:
            field_interp_chunks = pool.map(interpolate_to_target_height_unpack, interpolation_arguments)
        # merge chunks back together
        field_interp = np.dstack(field_interp_chunks)        
        # write interpolated field no netCDF file
        name = varname[variable]
        unit = varunit[variable]
        latlonheightfield_to_netCDF(target_height, lat, lon, field_interp,\
                                    name, unit , outfile, overwrite=True)
        
def get_interpolation_arguments(field_chunks, height_chunks, target_height):
    """ Merges chunks into a list of dictionaries that can be passed to the interpolation function.
    """
    interpolation_arguments = []
    for i in range(len(field_chunks)):
        arg = {}
        arg['field'] = field_chunks[i]
        # heights can be either 3D or 1D
        if len(height_chunks) > 1:
            arg['height'] = height_chunks[i]
        else:
            arg['height'] = height_chunks
        arg['target_height'] = target_height
        interpolation_arguments.append(arg)
    return interpolation_arguments  

def interpolate_to_target_height_unpack(args):
    """ Function to unpack arguments to pass to interpolation function.
    """
    return interpolate_to_target_height(
        field=args['field'],
        height=args['height'],
        target_height=args['target_height']
    )

def interpolate_to_target_height(field, height, target_height):
    """ Interpolates field from from height levels given by height to new height levels given in target_height.
    """
    nheight, nlat, nlon = field.shape
    field_interp = np.ones(field.shape) * np.nan

    for i in range(nlat):
        for j in range(nlon):
            field_interp[:, i, j] = interp1d(
                height[:, i, j],
                field[:, i, j],
                bounds_error=False,
                fill_value=np.nan)(target_height)
            
    print('interpolated.')
    return field_interp

def latlonheightfield_to_netCDF(height, latitude, longitude, field, varname, varunit, outname, overwrite=True):
    """ Saves a field with dimensions (height, latitude, longitude) to a NetCDF file.
    
    Parameters:
        height (1D-array, dimension nheight): height levels in m
        latitude (1D-array, dimension nlat): latitudes in deg North
        longitude (1D-array, dimension nlon): longitudes in deg North
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

def get_modelspecific_names(model):
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

    else:
        print('Model {} does not exist or is not implemented yet.')
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
    
    
 
