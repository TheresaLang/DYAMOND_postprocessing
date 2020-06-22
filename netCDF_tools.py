import numpy as np
import os
import typhon
from netCDF4 import Dataset

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

def latlonfield_to_netCDF(latitude, longitude, field, varname, varunit, outname, time_dim=False, time_var=None, overwrite=True):
    """ Saves a field with dimensions (latitude, longitude) to a NetCDF file.
    
    Parameters:
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

        lat = f.createVariable('lat','f4',('lat',))
        lon = f.createVariable('lon','f4',('lon',))
        
        if time_dim:
            if time_var is not None:
                time = f.createDimension('time', len(time_var))
                time = f.createVariable('time', 'f4', ('time',))
            else:
                time = f.createDimension('time', 1)

        lat.units = 'degrees north'
        lon.units= 'degrees east'

        lat[:] = latitude[:]
        lon[:] = longitude[:]
        
        if time_var is not None:
            time.units = 'hours'
            time = time_var
        
        if time_dim:
            v = f.createVariable(varname,'f4',dimensions=('time','lat','lon'))
            f[varname][:,:,:]=field[:,:,:]
           
        else:
            v = f.createVariable(varname,'f4',dimensions=('lat','lon'))
            f[varname][:,:]=field[:,:] 
        v.units = varunit
        
def array2D_to_netCDF(array, varname, varunit, dimensionvars, dimensionnames, outname, overwrite=True):
    """ Saves a 1D-Array to a NetCDF file.
    
    Parameters:
        vector (1D Array): Vector with variable to save
        varname (string): Name of variable to save
        varunit (string): Unit of variable to save
        dimensionvar (Tuple of 1D Arrays): Grid the variable is given on (e.g. latitudes, longitudes, altitudes) 
        dimensionname (Tuple of strings): Name of dimension associated with vector (e.g. 'lat', 'lon', 'height')
        outname (string): Name of output file
        overwrite (boolean): If True, exisiting files with the same filename are overwritten
        
    """
    if (os.path.isfile(outname)) and overwrite:
        os.remove(outname)
    elif (os.path.isfile(outname)) and overwrite==False:
        print('File {} exists. Not overwriting.'.format(outname))
        return
    
    with Dataset(outname, 'w') as f:
        dim0 = f.createDimension(dimensionnames[0], len(dimensionvars[0]))
        dimvar0 = f.createVariable(dimensionnames[0], 'f4', (dimensionnames[0],))    
        dimvar0[:] = dimensionvars[0]
        
        dim1 = f.createDimension(dimensionnames[1], len(dimensionvars[1]))
        dimvar1 = f.createVariable(dimensionnames[1], 'f4', (dimensionnames[1],))    
        dimvar1[:] = dimensionvars[1]

        f.createVariable(varname, 'f4', dimensions=dimensionnames)
        f[varname][:,:] = array[:,:]
        
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

