import numpy as np
import pandas as pd
import os
import glob
import logging
import json
import typhon
import pickle
from time import sleep
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from moisture_space import utils

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_modelspecific_varnames(model):
    """ Returns dictionary with variable names for a specific model.
    
    Parameters:
        model (string): Name of model
    """
    
    if model == 'ICON':
        varname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P',
            'SURF_PRES': 'PS', 
            'RH': 'RH',
            'QI': 'QI_DIA',
            'OLR': 'ATHB_T',
            'IWV': 'TQV_DIA',
            'TQC': 'TQC_DIA',
            'TQI': 'TQI_DIA',
            'TQR': 'TQR',
            'TQS': 'TQS',
            'TQG': 'TQG',
            'QC': 'QC_DIA',#'param212.1.0',
            'W500': 'omega',
            'W': 'W',#'wz', 
            'U': 'U',
            'V': 'V',
            'WHL': 'W',
            'STOA': 'ASOB_T',
            'PREC': 'TOT_PREC',
            'OLRC': '-',
            'STOAC': '-',
            'OMEGA': '-'
        }
    elif model == 'ICON_coupled':
        varname = {
            'TEMP': 'ta',
            'QV': 'hus',
            'PRES': 'pfull',
            'SURF_PRES': 'ps', 
            'RH': 'RH',
            'U': 'ua',
            'V': 'va',
            'OMEGA': 'wap',
            'IWV': 'prw',
            'PR': 'pr'
        }
    elif model == 'NICAM':
        varname = {
            'TEMP': 'ms_tem',
            'QV': 'ms_qv',
            'PRES': 'ms_pres',
            'SURF_PRES': 'ss_slp',
            'RH': 'RH',
            'RHint': 'ms_rh',
            'QI': 'ms_qi',
            'IWV': 'sa_vap_atm',
            'OLR': 'sa_lwu_toa',
            'FTOASWD': 'ss_swd_toa',
            'FTOASWU': 'ss_swu_toa',
            'QC': 'ms_qc',
            'W': 'ms_w',
            'U': 'ms_u',
            'V': 'ms_v',
            'SUTOA': 'ss_swu_toa',
            'SDTOA': 'ss_swd_toa',
            'OLRC': 'ss_lwu_toa_c',
            'STOAC': '-',
            'OMEGA': '-',
            'QR': 'ms_qr',
            'QS': 'ms_qs',
            'QG': 'ms_qg'
        }
    elif model == 'GEOS':
        varname = {
            'TEMP': 'T',
            'QV': 'QV',
            'PRES': 'P',
            'SURF_PRES': 'SLP',
            'RH': 'RH',
            'H': 'H',
            'QI': 'QI',
            'OLR': 'OLR',
            'OLRC': 'OLRC',
            'STOA': 'SWTNET',
            'STOAC': 'SWTNETC',
            'IWV': 'TQV',
            'QC': 'QL',
            'W': 'W', 
            'V': 'V',
            'U': 'U',
            'W500': 'OMEGA',
            'OMEGA': '-'
        }    
    elif model == 'IFS':
        varname = {
            'TEMP': 'TEMP',
            'QV': 'QV',
            'QI': 'QI',
            'QC': 'param83.1.0',
            'W': 'param120.128.192',
            'U': 'U',
            'V': 'V',
            'PRES': 'PRES',
            'RH': 'RH',
            'OLR': 'OLR',
            'IWV': 'IWV',
            'H': 'H',
            'OLRC': 'OLRC',
            'STOA': 'STOA',
            'STOAC': 'STOAC',
            'OMEGA': '-',
            'GH': 'GH'
        }
    elif model == 'SAM':
        varname = {
            'TEMP': 'TABS',
            'QV': 'QV',
            'PRES': 'PRES',
            'SURF_PRES': 'PSFC',
            'RH': 'RH',
            'QI': 'QI',
            'IWV': 'PW',
            'OLR': 'LWNTA',
            'QC': 'QC',
            'W': 'W',
            'U': 'U',
            'V': 'V',
            'STOA': 'SWNTA',
            'OMEGA': '-'
        }
    elif model == 'UM':
        varname = {
            'TEMP': 'air_temperature',
            'QV': 'specific_humidity',
            'PRES': 'air_pressure',
            'SURF_PRES': 'surface_air_pressure',
            'RH': 'RH',
            'QI': 'mass_fraction_of_cloud_ice_in_air',
            'OLR': 'toa_outgoing_longwave_flux',
            'IWV': 'atmosphere_water_vapor_content',
            'ICI': 'atmosphere_mass_content_of_cloud_ice',
            'QC': 'mass_fraction_of_cloud_liquid_water_in_air',
            'W': 'upward_air_velocity',
            'U': 'eastward_wind',
            'V': 'northward_wind',
            'SDTOA': 'toa_incoming_shortwave_flux',
            'SUTOA': 'toa_outgoing_shortwave_flux',
            'STOA': 'STOA', 
            'OMEGA': '-'
        }
        
    elif model == 'FV3':
        varname = {
            'TEMP': 'temp',
            'QV': 'qv',
            'PRES': 'pres',
            'SURF_PRES': 'ps',
            'RH': 'RH',
            'QI': 'qi',
            'IWV': 'intqv',
            'OLR': 'flut',
            'H': 'H',
            'W': 'w',
            'U': 'u',
            'V': 'v',
            'QC': 'ql',
            'SDTOA': 'fsdt',
            'SUTOA': 'fsut',
            'STOA': 'STOA',
            'OMEGA': '-'
        }
        
    elif model == 'MPAS':
        varname = {
            'TEMP': 'temperature',
            'QV': 'qv',
            'QI': 'qi',
            'PRES': 'pressure',
            'SURF_PRES': 'mslp',
            'RH': 'RH',
            'IWV': 'vert_int_qv',
            'OLR': 'aclwnett',
            'W': 'w',
            'U': 'uReconstructZonal',
            'V': 'uReconstructMeridional',
            'QC': 'qc',
            'STOA': 'acswnett',
            'OMEGA': '-'
        }
        
    elif model == 'ARPEGE':
        varname = {
            'TEMP': 'T',
            'QV': 'QV',
            'QI': 'param84.1.0',
            'SURF_PRES': 'PS',
            'RH': 'RH',
            'IWV': 'TQV',
            'OLR': 'ACCTHB_T',
            'W': 'W',
            'U': 'U',
            'V': 'V',
            'QC': 'param83.1.0',
            'STOA': 'ACCSOB_T',
            'OMEGA': 'W',
            'PRES': 'PRES',
            'GH': 'FI'
        }

    else:
        print(f'Modelspecific variable names for Model {model} have not been implemented yet.')
        return
        
    return varname

def get_variable_units():
    """ Returns dictionary with units for different variables. 
    """
    
    varunit = {
        'TEMP': 'K',
        'QV': 'kg kg**-1',
        'PRES': 'Pa',
        'RH': '-',
        'QI': 'kg kg**-1',
        'QC': 'kg kg**-1',
        'OLR': 'W m**-2',
        'IWV': 'kg m**-2',
        'ICI': 'kg m**-2',
        'H': 'm',
        'W': 'm s**-1',
        'U': 'm s**-1',
        'V': 'm s**-1',
        'OMEGA': 'Pa s**-1',
        'STOA': 'W m**-2',
        'STOAC': 'W m**-2',
        'SUTOA': 'W m**-2',
        'SDTOA': 'W m**-2',
        'OLRC': 'W m**-2',
        'PREC': 'kg s**-1 m**-2'
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
        elif run == '2.5km' or run == '2.5km_winter':
            weights = 'ICON_R2B10_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "5.0km_1", "5.0km_2", "2.5km".')
            return None
        
    elif model == 'ICON_coupled':
        grid_dir = '/work/mh0066/m300752/DYAMOND++/data/weight/'
        weights = 'weight_dpp0016_01x01.nc'
        
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
            #grid_dir = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/UM'
            weights = 'UM-5km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "5.0km".')
            
    elif model == 'FV3':
        if run == '3.25km':
            weights = 'FV3-3.3km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported run is: "3.25km".')
            
    elif model == 'MPAS':
        if run == '3.75km':
            weights = 'MPAS-3.75km_0.10_grid_wghts.nc'
        elif run == '7.5km':
            weights = 'MPAS-7.5km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported runs are: "3.75km" and "7.5km".')
            
    elif model == 'ARPEGE':
        if run == '2.5km':
            weights = 'ARPEGE-2.5km_0.10_grid_wghts.nc'
        else:
            logger.error(f'Run {run} not supported for {model}.\nSupported run is: "2.5km".')
    
    elif model == 'ERA5':
        grid_dir = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/ERA5'
        weights = 'ERA5-31.0km_0.10_grid_wghts.nc'
    # other models...
    else:
        logger.error('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
        return None
    
    path2weights = os.path.join(grid_dir, weights)
    if not os.path.isfile(path2weights):
        logger.warning('Warning: There are no pre-calculated weights for this model run and target grid yet.')

    return path2weights

def get_path2heightfile(model, grid_res=0.1, **kwargs):
    """ Returns path to file containing geometrical heights corresponding to model levels for a given model.
   
    Parameters:
        model (str): name of model
        grid_res (num): resolution of grid in degrees
    """
    gridname = str(grid_res).ljust(4, '0')
    return f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{model}-{gridname}deg_heights.nc'

def get_path2targetheightfile(model, data_dir, **kwargs):
    """
    """

    targetheightfile = f'{data_dir}/{model}/target_height.nc'
    
    return targetheightfile

def get_path2z0file(model, run, **kwargs):
    """
    """
    if model == 'FV3':
        path2z0file = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{model}-{run}_OROG_sea_estimated_trop.nc'
    else:
        path2z0file = f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/{model}/{model}-{run}_OROG_trop.nc'
    
    return path2z0file

def get_path2landmask(lonlatbox, **kwargs):
    """ Returns path to file containing a land mask.
    """
    if lonlatbox == [-180, 180, -30, 30]:
        latlonstr = ''
    else:
        latlonstr = f'_{lonlatbox[0]}_{lonlatbox[1]}_{lonlatbox[2]}_{lonlatbox[3]}'
        
    return f'/mnt/lustre02/work/mh1126/m300773/DYAMOND/ICON/land_mask{latlonstr}.nc'

def get_interpolationfilelist(models, runs, variables, time_period, temp_dir, grid_res, **kwargs):
    """ Returns list of all raw output files from a specified DYAMOND model run corresponding to a given
    time period and given variables. A list of filenames for output files needed e.g. for
    horizontal interpolation is also returned.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period in the format YYYY-mm-dd
        temp_dir (str): path to directory for output files
        grid_res (float): resolution of target grid in degrees (e.g. 0.1)
    """
    dyamond_dir = '/work/ka1081/DYAMOND'
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    
    raw_files = []
    out_files = []
    options = []
    weights = []
    grids = []
    
    for model, run in zip(models, runs):
        print(model)
        print(run)
        varnames = get_modelspecific_varnames(model)

        if model == 'ICON':
            # variables with 15-minute output
            vars_15min = ['SURF_PRES', 'OLR', 'IWV', 'TQI', 'TQC', 'TQR', 'TQS', 'TQG', 'W500']
            # dictionary containing endings of filenames containing the variables
            var2suffix = {
                'TEMP': 'atm_3d_t_ml_',
                'QV': 'atm_3d_qv_ml_',
                'PRES': 'atm_3d_pres_ml_',
                'SURF_PRES': 'atm2_2d_ml_',
                'IWV': 'atm1_2d_ml_',
                'TQI': 'atm1_2d_ml_',
                'TQC': 'atm1_2d_ml_',
                'TQR': 'atm3_2d_ml_',
                'TQS': 'atm1_2d_ml_',
                'TQG': 'atm1_2d_ml_',
                'QI': 'atm_3d_tot_qi_dia_ml_',
                'OLR': 'atm_2d_avg_ml_',
                'QC': 'atm_3d_tot_qc_dia_ml_',
                'W500': 'atm_omega_3d_pl_',
                'W': 'atm_3d_w_ml_',
                'U': 'atm_3d_u_ml_',
                'V': 'atm_3d_v_ml_',
                'STOA': 'atm_2d_avg_ml_',
                'PREC': 'atm2_2d_ml_',
                'OLRC': '-',
                'STOAC': '-',
                'OMEGA': '-'
                    }

            for var in variables:
                # filenames of raw files, grid weights and the output file
                var_suffix = var2suffix[var]
                varname = varnames[var]
                if run == '5.0km_1':
                    stem = os.path.join(dyamond_dir, f'ICON-5km/nwp_R2B09_lkm1006_{var_suffix}')
                elif run == '5.0km_2':
                    stem = os.path.join(dyamond_dir, f'ICON-5km_2/nwp_R2B09_lkm1013_{var_suffix}')
                elif run == '2.5km':
                    stem = os.path.join(dyamond_dir, f'ICON-2.5km/nwp_R2B10_lkm1007_{var_suffix}')
                elif run == '2.5km_winter':
                    stem = f'/mnt/lustre02/work/mh1126/m300773/DYAMONDwinter/ICON-2.5km/nwp_R2B10_lkm1007_{var_suffix}'
                else:
                    print (f'Run {run} not supported for {model}.\nSupported runs are: "5.0km_1", "5.0km_2", "2.5km".')
                    return None
                
                # ICON output is one file per day
                for t in time:
                    # filenames
                    raw_file = stem + t.strftime("%Y%m%d") + 'T000000Z.grb'
                    time_str = t.strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(raw_file)
                    out_files.append(out_file)
                    weights.append(get_path2weights(model, run, grid_res))
                    grids.append(get_path2grid(grid_res))
                    
                    # options for remapping
                    opt = f'-chname,{varname},{var} -selvar,{varname}'
                    if var in vars_15min:
                        opt = opt + ' -seltimestep,1/96/12'
                    options.append(opt)
                    
        elif model == 'ICON_coupled':
            vars_1hr = ['IWV']
            var2suffix = {
                'TEMP': '_atm_3d_1_ml_',
                'PRES': '_atm_3d_1_ml_',
                'QV': '_atm_3d_4_ml_',
                'U': '_atm_3d_2_ml_',
                'V': '_atm_3d_2_ml_',
                'OMEGA': '_atm_3d_3_ml_',
                'IWV': '_atm1_2d_ml_',
                'PR': '_atm2_2d_ml_'
            }
            
            for var in variables:
                var_suffix = var2suffix[var]
                varname = varnames[var]
                # paths to raw output
                if run == 'dpp0016':
                    directory = '/work/mh0287/k203123/GIT/icon-aes-dyw_albW/experiments/dpp0016/'
                elif run == 'dpp0020':
                    directory = '/work/mh0287/k203123/GIT/icon-aes-dyw2/experiments/dpp0020/'
                elif run == 'dpp0029':
                    directory = '/work/mh0287/k203123/GIT/icon-aes-dyw2/experiments/dpp0029/'
            
                # ICON output is one file per day
                for t in time:
                    # filenames
                    raw_file = os.path.join(directory, f'{run}{var_suffix}{t.strftime("%Y%m%d")}T000000Z.nc')
                    time_str = t.strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(raw_file)
                    out_files.append(out_file)
                    weights.append(get_path2weights(model, run, grid_res))
                    grids.append(get_path2grid(grid_res))
                    
                    # options for remapping
                    opt = f'-chname,{varname},{var} -selvar,{varname}'
                    if var in vars_1hr:
                        opt = opt + ' -seltimestep,1/24/6'
                    options.append(opt)
                   
        elif model == 'NICAM':
            # variables with 15-minute output
            vars_15min = ['SURF_PRES', 'OLR', 'IWV', 'OLRC', 'SUTOA', 'SDTOA']
            # dictionary containing filenames containing the variables
            var2filename = {
                'TEMP': 'ms_tem.nc',
                'QV': 'ms_qv.nc',
                'PRES': 'ms_pres.nc',
                'SURF_PRES': 'ss_slp.nc',
                'IWV': 'sa_vap_atm.nc',
                'OLR': 'sa_lwu_toa.nc',
                'QC': 'ms_qc.nc',
                'QI': 'ms_qi.nc',
                'QR': 'ms_qr.nc',
                'QS': 'ms_qs.nc',
                'QG': 'ms_qg.nc',
                'RHint': 'ms_rh.nc',
                'W': 'ms_w.nc',
                'U': 'ms_u.nc',
                'V': 'ms_v.nc',
                'SUTOA': 'ss_swu_toa.nc',
                'SDTOA': 'ss_swd_toa.nc',
                'OLRC': 'ss_lwu_toa_c.nc',
                'STOA': '-',
                'STOAC': '-',
                'OMEGA': '-'
            }

            for var in variables:
                filename = var2filename[var]
                varname = varnames[var]
                if run == '3.5km':
                    stem = os.path.join(dyamond_dir, 'NICAM-3.5km')
                elif run == '7.0km':
                    stem = os.path.join(dyamond_dir, 'NICAM-7km')
                else:
                    print (f'Run {run} not supported for {model}.\nSupported runs are: "3.5km" and "7.0km".')
                    return None

                # options for remapping
                opt = f'-chname,{varname},{var} -selvar,{varname}'
                if var in vars_15min:
                    opt = opt + ' -seltimestep,12/96/12'
                # append filenames for this variable to raw_files and out_files
                # NICAM output is one file per day
                for t in time:
                    # filenames
                    dayfolder = glob.glob(os.path.join(stem, t.strftime("%Y%m%d")+'*'))[0]
                    raw_file = os.path.join(dayfolder, filename)
                    time_str = t.strftime("%m%d")
                    out_file = f'{model}-{run}_{var}_{time_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)

                    raw_files.append(raw_file)
                    out_files.append(out_file)
                    weights.append(get_path2weights(model, run, grid_res))
                    grids.append(get_path2grid(grid_res))
                    options.append(opt)
                    

        elif model == 'GEOS':
            vars_flux = ['OLR', 'STOA', 'OLRC', 'STOAC']
            vars_2D = ['IWV', 'SURF_PRES']
            
            var2dirname = {
                'TEMP': 'T',
                'QV': 'QV',
                'PRES': 'P',
                'SURF_PRES': 'asm',
                'H': 'H',
                'QI': 'QI',
                'IWV': 'asm',
                'OLR': 'flx',
                'OLRC': 'flx',
                'QC': 'QL',
                'W': 'W',
                'U': 'U',
                'V': 'V',
                'STOA': 'flx',
                'STOAC': 'flx',
                'OMEGA': '-'
            }
            # GEOS output is one file per time step (3-hourly)
            for var in variables:
                varname = var2dirname[var]
                varname_infile = varnames[var]
                if run == '3.0km':
                    if var in vars_flux:
                        stem = os.path.join(dyamond_dir, f'GEOS-3km/tavg/tavg_15mn_2d_{varname}_Mx') 
                    elif var in vars_2D:
                        stem = os.path.join(dyamond_dir, f'DYAMOND/GEOS-3km/inst/inst_15mn_2d_{varname}_Mx')
                    else:
                        stem = os.path.join(dyamond_dir, f'DYAMOND/GEOS-3km/inst/inst_03hr_3d_{varname}_Mv')
                else:
                    print (f'Run {run} not supported for {model}.\nSupported runs are: "3.0km".')  

                # options for remapping
                opt = f'-chname,{varname_infile},{var} -selvar,{varname_infile}'
                # filenames
                for t in time:
                    for h in np.arange(0, 24, 3):
                        date_str = t.strftime("%Y%m%d")
                        hour_str = f'{h:02d}'
                        if var in vars_flux:
                            hour_file = f'DYAMOND.tavg_15mn_2d_flx_Mx.{date_str}_{hour_str}00z.nc4'
                        elif var in vars_2D:
                            hour_file = f'DYAMOND.inst_15mn_2d_asm_Mx.{date_str}_{hour_str}00z.nc4'
                        else:
                            hour_file = f'DYAMOND.inst_03hr_3d_{varname}_Mv.{date_str}_{hour_str}00z.nc4'
                        hour_file = os.path.join(stem, hour_file)
                        date_str = time[i].strftime("%m%d")
                        out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                        out_file = os.path.join(temp_dir, out_file)
                        raw_files.append(hour_file)
                        out_files.append(out_file)
                        weights.append(get_path2weights(model, run, grid_res))
                        grids.append(get_path2grid(grid_res))
                        options.append(opt)

        elif model == 'IFS':
            var2varnumber = {
                'TEMP': 130,
                'QV': 133,
                'SURF_PRES': 152, 
                'QI': 247,
                'OLR': 179,
                'IWV': 137,
                'QC': 246,
                'W': 120,
                'STOA': 178,
                'OLRC': 209, 
                'STOAC': 208,
                'OMEGA': '-',
                'GH': 'FI'
            }
            var2variablename = {
                'TEMP': 'T',
                'QV': 'QV',
                'SURF_PRES': 'lnsp',
                'IWV': 'tcwv',#
                'QI': 'param84.1.0', 
                'QC': 'param83.1.0',
                'W': 'param120.128.192',
                'U': 'U',
                'V': 'V',
                'OLR': 'ttr',
                'OLRC': 'ttrc',
                'STOA': 'tsr',
                'STOAC': 'tsrc',
                'PRES': '-',
                'OMEGA': '-',
                'GH': 'FI'
            }

            for var in variables:
                # option for remapping 
                opt = f'-chname,{var2variablename[var]},{var}'
                for i in np.arange(time.size):
                    for h in np.arange(0, 24, 3):
                        date_str = time[i].strftime("%m%d")
                        hour_str = f'{h:02d}'
                        hour_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                        out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                        hour_file = os.path.join(temp_dir, hour_file)
                        out_file = os.path.join(temp_dir, out_file)
                        raw_files.append(hour_file)
                        out_files.append(out_file)
                        weights.append(get_path2weights(model, run, grid_res))
                        grids.append(get_path2grid(grid_res))
                        options.append(opt)

        elif model == 'SAM':
            ref_date = pd.Timestamp('2016-08-01-00')
            timeunit_option = '-settunits,days' 
            grid_option = '-setgrid,/mnt/lustre02/work/ka1081/DYAMOND/SAM-4km/OUT_2D/DYAMOND_9216x4608x74_7.5s_4km_4608_0000002400.CWP.2D.nc'
            vars_2D = ['OLR', 'STOA', 'IWV', 'SURF_PRES']
            
            var2filename = {
                'TEMP': '_TABS',
                'QV': '_QV',
                'PRES': '_PP',
                'SURF_PRES': '.PSFC.2D',
                'QI': '_QI',
                'IWV': '.PW.2D',
                'OLR': '.LWNTA.2D',
                'QC': '_QC',
                'W': '_W',
                'U': '_U',
                'V': '_V',
                'STOA': '.SWNTA.2D',
                'OLRC': '-',
                'STOAC': '-',
                'OMEGA': '-'
            }

            stem = os.path.join(dyamond_dir, 'SAM-4km')
            for var in variables:
                varname = varnames[var]
                if var in vars_2D:
                    stem = os.path.join(stem, 'OUT_2D')
                else:
                    stem = os.path.join(stem, 'OUT_3D')
                    
                chname_option = f'-chname,{varname},{var}'
                for i in np.arange(time.size):
                    for h in np.arange(0, 24, 3):
                        # filenames
                        date_str = time[i].strftime('%Y-%m-%d')
                        hour_str = f'{h:02d}'

                        timestamp = pd.Timestamp(f'{date_str}-{hour_str}')
                        secstr = int((timestamp - ref_date).total_seconds() / 7.5)
                        secstr = f'{secstr:010}'
                        hour_file = f'DYAMOND_9216x4608x74_7.5s_4km_4608_{secstr}{var2filename[var]}.nc'
                        hour_file = os.path.join(stem, hour_file)
                        date_str_short = time[i].strftime("%m%d")
                        out_file = f'{model}-{run}_{var}_{date_str_short}_{hour_str}_hinterp.nc'
                        
                        # combine options for remapping
                        timeaxis_option = f'-settaxis,{date_str},{hour_str}:00:00,3h'
                        option = ' '.join([chname_option, timeaxis_option, timeunit_option, grid_option])
                        
                        out_file = os.path.join(temp_dir, out_file)
                        raw_files.append(hour_file)
                        out_files.append(out_file)        
                        weights.append(get_path2weights(model, run, grid_res))
                        grids.append(get_path2grid(grid_res))
                        options.append(option)  

        elif model == 'UM':
            stem = '/mnt/lustre02/work/ka1081/DYAMOND/UM-5km'
            # variables with 15-min output
            vars_15min = ['IWV', 'SURF_PRES']
            # variables with 1h output
            vars_1h = ['OLR', 'SDTOA', 'SUTOA']

            var2dirname = {
                'TEMP': 'ta',
                'QV': 'hus',
                'PRES': 'phalf',
                'SURF_PRES': 'ps',
                'IWV': 'prw',
                'QI': 'cli',
                'OLR': 'rlut',
                'QC': 'clw',
                'W': 'wa',
                'U': 'ua',
                'V': 'va-NEW',
                'SDTOA': 'rsdt',
                'SUTOA': 'rsut',
                'STOA': '-',
                'OLRC': '-',
                'STOAC': '-',
                'OMEGA': '-'
            }

            opt = f'-chname,{varname},{var} -selvar,{varname}'
            for var in variables:
                varname = varnames[var]
                dirname = var2dirname[var]
                for i in np.arange(time.size):
                    date_str = time[i].strftime('%Y%m%d')
                    if var == 'V':
                        varstr = 'va'
                    else: 
                        varstr = var2dirname[var]

                    if var in vars_15min:
                        opt = opt + ' -seltimestep,12/96/12'
                        out_interv_str = '15min'
                    elif var in vars_1h:
                        opt = opt + ' -seltimestep,3/24/3'
                        out_interv_str = '1hr'
                    else:
                        out_interv_str = '3hr'
                        
                    day_file = f'{varstr}_{out_interv_str}_HadGEM3-GA71_N2560_{date_str}.nc'
                    day_file = os.path.join(stem, dirname, day_file)
                    date_str_short = time[i].strftime('%m%d')
                    out_file = f'{model}-{run}_{var}_{date_str_short}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)

                    raw_files.append(day_file)
                    out_files.append(out_file)
                    weights.append(get_path2weights(model, run, grid_res))
                    grids.append(get_path2grid(grid_res))
                    options.append(opt)

        # For the following models, raw output has to be prepared before the 
        # horizontal interpolation can be performed!

        elif model == 'FV3':
            if run == '3.25km':
                stem = '/mnt/lustre02/work/ka1081/DYAMOND/FV3-3.25km'
            else: 
                print (f'Run {run} not supported for {model}.\nSupported run is "3.25km".')
                
            vars_15min = ['IWV', 'OLR', 'SDTOA', 'SUTOA', 'SURF_PRES']

            var2filename = {
                'TEMP': 'temp',
                'QV': 'qv',
                'PRES': 'pres',
                'SURF_PRES': 'ps',
                'QI': 'qi',
                'IWV': 'intqv',
                'OLR': 'flut',
                'W': 'w',
                'U': 'u',
                'V': 'v',
                'QC': 'ql',
                'SDTOA': 'fsdt',
                'SUTOA': 'fsut',
                'STOA': '-',
                'OLRC': '-',
                'STOAC': '-',
                'OMEGA': '-'
            }

            target_time = pd.date_range(time_period[0]+' 3:00:00', pd.Timestamp(time_period[1]+' 0:00:00')+pd.DateOffset(1), freq='3h')
            time_15min = pd.date_range("2016-08-01 0:15:00", "2016-09-10 0:00:00", freq='15min')
            time_3h = pd.date_range("2016-08-01 3:00:00", "2016-09-10 0:00:00", freq='3h')
            
            for var in variables:
                varname = varnames[var]
                for t in target_time:
                    if t < pd.Timestamp('2016-08-11 3:00:00'):
                        dir_name = '2016080100'
                    elif t < pd.Timestamp('2016-08-21 3:00:00'):
                        dir_name = '2016081100'
                    elif t < pd.Timestamp('2016-08-31 3:00:00'):
                        dir_name = '2016082100'
                    else:
                        dir_name = '2016083100'
                    
                    if var in vars_15min:
                        timestep = np.mod(np.where(time_15min == t)[0][0], 960) + 1
                    else:
                        timestep = np.mod(np.where(time_3h == t)[0][0], 80) + 1
                        
                    if var in ['U', 'V']:
                            hour_file = f'{var.lower()}_3hr.nc'
                            hour_file = os.path.join(temp_dir, dir_name, hour_file)
                    else:
                        hour_file = var2filename[var] + '_C3072_12288x6144.fre.nc'
                        hour_file = os.path.join(stem, dir_name, hour_file)
                        
                    date_str = t.strftime('%m%d')
                    hour_str = t.strftime('%H')
                    out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    out_file = os.path.join(temp_dir, out_file)
                    option = f'-chname,{varname},{var} -selvar,{varname} -seltimestep,{timestep}'
                    print(timestep)
                    print(out_file)

                    raw_files.append(hour_file)
                    out_files.append(out_file)
                    grids.append(get_path2grid(grid_res))
                    if var in ['U', 'V']:
                        weights.append('/mnt/lustre02/work/mh1126/m300773/DYAMOND/FV3/FV3-3.25km_UV_0.10_grid_wghts.nc')
                    else:
                        weights.append(get_path2weights(model, run, grid_res))
                    
                    options.append(option)

        elif model == 'MPAS':
            for var in variables:
                varname = varnames[var]
                for i in np.arange(time.size):
                    for h in np.arange(0, 24, 3):
                        date_str = time[i].strftime("%m%d")
                        hour_str = f'{h:02d}'
                        hour_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                        hour_file = os.path.join(temp_dir, hour_file)
                        out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                        out_file = os.path.join(temp_dir, out_file)
                        
                        raw_files.append(hour_file)
                        out_files.append(out_file)
                        weights.append(get_path2weights(model, run, grid_res))
                        grids.append(get_path2grid(grid_res))
                        
                        option = f'-chname,{varname},{var} -selvar,{varname}'
                        options.append(option)

        elif model == 'ARPEGE':
            preprocessing_path = '/mnt/lustre02/work/um0878/users/tlang/work/dyamond/processing/prerocessing/preprocessing_ARPEGE'
            griddesfile = os.path.join(preprocessing_path, 'griddes.arpege1')
            for var in variables:
                varname = varnames[var]
                # options for remapping
                opt = f'-setgrid,{griddesfile} -setgridtype,regular -chname,{varname},{var}'
                
                # filenames
                for t in time:
                    # file are missing for 10 August
                    if t.day == 10 and t.month == 8:
                        hours = np.arange(3, 22, 3)
                    else:
                        hours = np.arange(3, 25, 3)
                        
                    for h in hours:
                        year_str = t.strftime("%Y")
                        date_str = t.strftime("%m%d")
                        hour_str = f'{h:02d}'
                        hour_file = f'{year_str}{date_str}{hour_str}_{var}.gp'
                        hour_file = os.path.join(temp_dir, hour_file)

                        # hr 24 in ARPEGE data corresponds to hr 00 on next day
                        if h == 24:
                            hour_str_out = '00'
                            time_out = time[i] + pd.Timedelta('1d')
                            date_str_out = time_out.strftime("%m%d")
                        else:
                            hour_str_out = hour_str
                            date_str_out = date_str
                        out_file = f'{model}-{run}_{var}_{date_str_out}_{hour_str_out}_hinterp.nc'
                        out_file = os.path.join(temp_dir, out_file)
                        
                        raw_files.append(hour_file)
                        out_files.append(out_file)
                        weights.append(get_path2weights(model, run, grid_res))
                        grids.append(get_path2grid(grid_res))
                        options.append(opt)
        
        elif model == 'ERA5':
            var2variablename = {
                'TEMP': 'T',
                'QV': 'QV',
                'SURF_PRES': 'var134',
                'QI': 'param84.1.0', 
                'QC': 'param83.1.0',
                'OMEGA': 'OMEGA',
                'U': 'U',
                'V': 'V'
            }
            for var in variables:
                opt = f'-seltimestep,1/24/3 -chname,{var2variablename[var]},{var}'
                for i in np.arange(time.size):
                    date_str = time[i].strftime("%m%d")
                    day_file = f'{model}-{run}_{var}_{date_str}.nc'
                    out_file = f'{model}-{run}_{var}_{date_str}_hinterp.nc'
                    day_file = os.path.join(temp_dir, day_file)
                    out_file = os.path.join(temp_dir, out_file)
                    raw_files.append(day_file)
                    out_files.append(out_file)
                    weights.append(get_path2weights(model, run, grid_res))
                    grids.append(get_path2grid(grid_res))
                    options.append(opt)
        
        else:
            logger.error('The model specified for horizontal interpolation does not exist or has not been implemented yet.') 
            return None
              
    return raw_files, out_files, weights, grids, options
                
def get_preprocessingfilelist(models, runs, variables, time_period, temp_dir, **kwargs):
    """ Returns list of all raw output files from a specified DYAMOND model run corresponding to a given
    time period and given variables. A list of filenames for output files needed for
    pre-processing of these models is also returned.
    
    Parameters:
        model (str): name of model
        run (str): name of model run
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period in the format YYYY-mm-dd
        temp_dir (str): path to directory for output files
    """
    
    models_list = []
    infile_list = []
    tempfile_list = []
    outfile_list = []
    option_1_list = []
    option_2_list = []
    
    time = pd.date_range(time_period[0], time_period[1], freq='1D')

    for model, run in zip(models, runs):
    
        if model == 'IFS':
            # Filenames of IFS raw files contain a number corresponding to the hours since the 
            # start of the simulation (2016-08-01 00:00:00)
            # define reference time vector starting at 2016-08-01 00:00:00
            ref_time = pd.date_range('2016-08-01 00:00:00', '2018-09-10 00:00:00', freq='3h')
            # find the index where the reference time equals the specified start time and 
            # multiply by 3 (because the output is 3-hourly) to get the filename-number of the
            # first relevant file
            hour_start = np.where(ref_time == time[0])[0][0] * 3
            var2filename = {
                'TEMP': 'gg_mars_out_ml_upper_sh',
                'QV': 'mars_out_ml_moist',
                'SURF_PRES': 'gg_mars_out_sfc_ps_orog',
                'QI': 'mars_out_ml_moist',
                'QC': 'mars_out_ml_moist',
                'W': 'gg_mars_out_ml_upper_sh',
                'U': 'gg_uv_mars_out_ml_vor_div_sh',
                'V': 'gg_uv_mars_out_ml_vor_div_sh',
                'OLR': 'mars_out',
                'IWV': 'mars_out',
                'OLRC': 'mars_out',
                'STOA': 'mars_out',
                'STOAC': 'mars_out',
                'PRES': '-',
                'OMEGA': '-',
                'GH': 'gg_mars_out_sfc_ps_orog'
            }
            if run == '9.0km':
                var2filename['SURF_PRES'] = 'mars_out_sfc_ps_orog_gg'
                
            var2variablename = {
                'TEMP': 'T',
                'QV': 'QV',
                'SURF_PRES': 'lnsp',
                'IWV': 'tcwv',#
                'QI': 'param84.1.0', 
                'QC': 'param83.1.0',
                'W': 'param120.128.192',
                'U': 'U',
                'V': 'V',
                'OLR': 'ttr',
                'OLRC': 'ttrc',
                'STOA': 'tsr',
                'STOAC': 'tsrc',
                'PRES': '-',
                'OMEGA': '-',
                'GH': 'FI'
            }
            var2numzlevels = {
                'TEMP': 113,
                'QV': 113,
                'QI': 113,
                'QC': 113,
                'W': 113,
                'U': 113,
                'V': 113,
                'SURF_PRES': 1,
                'OLR': 1,
                'IWV': 1,
                'STOA': 1,
                'OLRC': 1,
                'STOAC': 1,
                'PRES': '-',
                'OMEGA': '-',
                'GH': 1
            }


            if (run=='4.0km'):
                stem   = '/work/ka1081/DYAMOND/IFS-4km'
            elif (run=='9.0km'):
                stem   = '/work/ka1081/DYAMOND/IFS-9km'

            for var in variables:
                hour_con = hour_start - 3
                # option 1: variable name in file
                option_1 = f'{var2variablename[var]}'
                # option 2: number of vertical levels (not needed for OLR and IWV):
                if var in['OLR', 'STOA', 'OLRC', 'STOAC', 'IWV', 'GH']:
                    option_2 = ''
                else:
                    option_2 = f'{var2numzlevels[var]}'
                for i in np.arange(time.size):
                    for h in np.arange(0, 24, 3):
                        hour_con = hour_con + 3
                        filename = var2filename[var]
                        in_file = f'{filename}.{hour_con}'
                        in_file = os.path.join(stem, in_file)
                        if var in ['TEMP', 'W', 'GH', 'U', 'V'] or (var == 'SURF_PRES' and run == '4.0km'):
                            in_file = in_file+'.grib'

                        date_str = time[i].strftime("%m%d")
                        hour_str = f'{h:02d}'
                        temp_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.grb'
                        out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                        temp_file = os.path.join(temp_dir, temp_file)
                        out_file = os.path.join(temp_dir, out_file)

                        infile_list.append(in_file)
                        tempfile_list.append(temp_file)
                        outfile_list.append(out_file)
                        option_1_list.append(option_1)
                        option_2_list.append(option_2)
                        models_list.append('IFS')

        elif model == 'ERA5':
            stem = '/work/bk1099/data/'
            vars_2D = ['SURF_PRES']
            var2varid = {
                'TEMP': 130,
                'SURF_PRES': 134,
                'QV': 133,
                'QI': 247,
                'QC': 246,
                'OMEGA': 135,
                'U': 131,
                'V': 132,
                'OLR': 235040,
                'STOA': 235035
            }
            
            for var in variables:
                # option 1: variable given on spectral or Gaussian reduced grid
                if var in ['TEMP', 'OMEGA', 'U', 'V']:
                    option_1 = 'spectral'
                else:
                    option_1 = 'gaussian'

                # 2D variables: One file per month
                if var in vars_2D:
                    directory = 'sf00_1H'
                    option_2 = 'splitday'
                    
                    for i in np.arange(time.size):
                        if i == 0 or (i > 0 and time[i].month != time[i-1].month):
                            month_str = time[i].strftime("%m") 
                            filename = f'E5sf00_1H_{time[i].year}-{month_str}_{var2varid[var]}'
                            in_file = os.path.join(stem, directory, str(time[i].year), filename)
                            out_file = f'{model}-{run}_{var}_{month_str}'
                            out_file = os.path.join(temp_dir, out_file)
                            temp_file = ''
                            
                            infile_list.append(in_file)
                            tempfile_list.append(temp_file)
                            outfile_list.append(out_file)
                            option_1_list.append(option_1)
                            option_2_list.append(option_2)
                            models_list.append('ERA5')                            
                        
                # 3D variables: One file per day
                else:
                    directory = 'ml00_1H'
                    option_2 = 'copy'
                
                    for i in np.arange(time.size):
                        date_str = time[i].strftime("%m-%d")
                        filename = f'E5ml00_1H_{time[i].year}-{date_str}_{var2varid[var]}'
                        in_file = os.path.join(stem, directory, str(time[i].year), filename)
                        date_str = time[i].strftime("%m%d")
                        out_file = f'{model}-{run}_{var}_{date_str}.nc'
                        out_file = os.path.join(temp_dir, out_file)
                        temp_file = ''

                        infile_list.append(in_file)
                        tempfile_list.append(temp_file)
                        outfile_list.append(out_file)
                        option_1_list.append(option_1)
                        option_2_list.append(option_2)
                        models_list.append('ERA5')
                                
        elif model == 'MPAS':
            stem = '/work/ka1081/DYAMOND/MPAS-3.75km'
            #stem = '/mnt/lustre02/work/bk1040/DYAMOND_SWAP/MPAS-3.75km'
            var2filename = {
                'TEMP': 'history',
                'PRES': 'history',
                'SURF_PRES': 'diag',
                'QV': 'history',
                'QI': 'history',
                'QC': 'history',
                'W': 'history',
                'U': 'history',
                'V': 'history',
                'IWV': 'diag',
                'OLR': 'diag',
                'STOA': 'diag'
            }
            varnames = get_modelspecific_varnames(model)

            for var in variables:
                for i in np.arange(time.size):
                    for h in np.arange(0, 24, 3):
                        date_str = time[i].strftime("%Y-%m-%d")
                        hour_str = f'{h:02d}'
                        in_file = f'{var2filename[var]}.{date_str}_{hour_str}.00.00.nc'
                        date_str = time[i].strftime("%m%d")
                        temp_file = f'{model}-{run}_{var}_{date_str}_{hour_str}_1.nc'
                        out_file = f'{model}-{run}_{var}_{date_str}_{hour_str}.nc'
                        in_file = os.path.join(stem, in_file)
                        temp_file = os.path.join(temp_dir, temp_file)
                        out_file = os.path.join(temp_dir, out_file)
                        option_selvar = varnames[var]

                        infile_list.append(in_file)
                        tempfile_list.append(temp_file)
                        outfile_list.append(out_file)
                        option_1_list.append(option_selvar)
                        option_2_list.append('')
                        models_list.append('MPAS')
                        
        # only needed for variables U and V
        elif model == 'FV3':
            #stem = '/work/ka1081/DYAMOND/FV3-3.25km'
            stem = '/mnt/lustre02/work/bk1040/DYAMOND_SWAP/FV3-3.25km'
            target_time_3h = pd.date_range(time_period[0]+' 3:00:00', pd.Timestamp(time_period[1]+' 0:00:00')+pd.DateOffset(1), freq='3h')
            time_3h = pd.date_range("2016-08-01 3:00:00", "2016-09-10 0:00:00", freq='3h')
            dir_names = ['2016080100', '2016081100', '2016082100', '2016083100']
            gridspec_file = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/FV3/FV3-3.25km_gridspec.nc'
            
            var2filename = {
                'U': 'u',
                'V': 'v'
            }
            
            for var in variables:
                for d in dir_names:
                    models_list.append(model)
                    varname = var2filename[var]
                    infile = os.path.join(stem, d, f'{varname}_3hr.tile?.nc')
                    outfile = os.path.join(temp_dir, d, f'{varname}_3hr.nc')

                    infile_list.append(infile)
                    tempfile_list.append('')
                    outfile_list.append(outfile)
                    option_1_list.append(gridspec_file)
                    option_2_list.append('')
            
#        else:
#            logger.error('The model specified for preprocessing does not exist or has not been implemented yet.')
        
    return models_list, infile_list, tempfile_list, outfile_list, option_1_list, option_2_list
                    
def get_preprocessing_ARPEGE_filelist(variables, time_period, temp_dir, **kwargs):
    """ Returns filelist needed for preprocessing of ARPEGE raw output files.
    
    Parameters:
        preprocess_dir (str): path to directory where preprocessing is done 
            (output can only be written the same directory)
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
    """
    #TODO: different parameter codes for some days (2D variables)
    #TODO: not all model levels included in 3D variables for 20180810 24:00
    
    time = pd.date_range(time_period[0], time_period[1], freq='1D')

    infile_3D_list = []
    infile_2D_list = []
    outfileprefix_list = []
    merge_list_3D = [] #verschachtelte Liste   
    tempfile_list_3D = []    
    filelist_2D = []    
    tempfile_list_2D = []
        
    dyamond_dir = '/work/ka1081/DYAMOND/ARPEGE-NH-2.5km'
    
    variables_2D = ['SURF_PRES', 'OLR', 'STOA']
    levels = np.arange(16, 76).astype(int)
    var2code = {
        'TEMP': '0.0.0',
        'QV': '0.1.0',
        'QI': '0.1.84',
        'QC': '0.1.83',
        'W': '0.2.9',
        'U': '0.2.2',
        'V': '0.2.3',
        'OLR': '0.5.5',
        'STOA': '0.4.9',
        'SURF_PRES': '0.3.0',
        'GH': '0.3.4'
    }
    
    var2t = {
        'TEMP': '119',
        'QV': '119',
        'QI': '119',
        'QC': '119',
        'W': '119',
        'U': '119',
        'V': '119',
        'OLR': '8',
        'STOA': '8',
        'SURF_PRES': '1',
        'GH': '119'
    }
    
    for i in np.arange(time.size):
        if time[i].day == 10:
            hours = np.arange(3, 22, 3)
        else:
            hours = np.arange(3, 25, 3)
          
        date_str = time[i].strftime('%Y%m%d')
        
        for h in hours:
            hour_str = f'{h:02d}'
            infile_3D = f'ARPNH3D{date_str}{hour_str}00'
            infile_2D = f'ARPNH2D{date_str}{hour_str}00'
            infile_3D = os.path.join(dyamond_dir, date_str, infile_3D)
            infile_2D = os.path.join(dyamond_dir, date_str, infile_2D)
            infile_3D_list.append(infile_3D)
            infile_2D_list.append(infile_2D)
            outfileprefix = date_str+hour_str
            outfileprefix = os.path.join(temp_dir, outfileprefix)
            outfileprefix_list.append(outfileprefix)
            merge_sublist = []
            tempfile_sublist_3D = []
            tempfile_sublist_2D = []
            file_sublist_2D = []
            for var in variables:
                code = var2code[var]
                tcode = var2t[var]
                if var not in variables_2D:
                    if var == 'GH':
                        mergefiles = [os.path.join(temp_dir, f'{date_str}{hour_str}.t{tcode}.l7500.grb.{code}.gp')]
                    else:
                        mergefiles = [os.path.join(temp_dir, f'{date_str}{hour_str}.t{tcode}.l{l}00.grb.{code}.gp')\
                                      for l in levels]
                    merge_sublist.append(mergefiles)
                    tempfile = os.path.join(temp_dir, f'{date_str}{hour_str}_{var}.gp')
                    tempfile_sublist_3D.append(tempfile)
                else:
                    file = os.path.join(temp_dir, f'{date_str}{hour_str}.t{tcode}.l000.grb.{code}.gp')
                    file_sublist_2D.append(file)
                    tempfile = os.path.join(temp_dir, f'{date_str}{hour_str}_{var}.gp')
                    tempfile_sublist_2D.append(tempfile)
            merge_list_3D.append(merge_sublist)
            tempfile_list_3D.append(tempfile_sublist_3D)
            tempfile_list_2D.append(tempfile_sublist_2D) 
            filelist_2D.append(file_sublist_2D)
        
    return infile_2D_list, infile_3D_list, outfileprefix_list, merge_list_3D, tempfile_list_3D, filelist_2D, tempfile_list_2D

    
def get_mergingfilelist(models, runs, variables, time_period, vinterp, temp_dir, data_dir, **kwargs):
    """ Returns a list of filenames of horizontally interpolated DYAMOND output needed for time merging.
    For each variable, the list contains a list of filenames. 
    
    Parameters:
        models (str): names of models
        runs (str): names of model runs
        variables (list of str): list of variables
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        vinterp (boolean): True if fields have already bin vertically interpolated
        temp_dir (str): path to directory with temporary files
        data_dir (str): path to directory to save files        
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    infile_list = []
    outfile_list = []
    
    v = 0
    for model, run in zip(models, runs):
        print(model)
        for var in variables:
            infile_list.append([])
            if vinterp == 1 and model in ['GEOS', 'IFS', 'FV3', 'ARPEGE', 'ERA5']:
                outfile_name = f'{model}-{run}_{var}_hinterp_vinterp_merged_{start_date}-{end_date}.nc'
            else:
                outfile_name = f'{model}-{run}_{var}_hinterp_merged_{start_date}-{end_date}.nc'
            
            if model == 'ERA5':
                outfile_name = os.path.join(data_dir, outfile_name)
            else:
                outfile_name = os.path.join(data_dir, model, outfile_name)
            outfile_list.append(outfile_name)

            if model == 'FV3' and not vinterp and var not in ['RH', 'H']:
                target_time = pd.date_range(time_period[0]+' 3:00:00', pd.Timestamp(time_period[1]+' 0:00:00')+pd.DateOffset(1), freq='3h')
                for t in target_time:
                    date_str = t.strftime('%m%d')
                    hour_str = t.strftime('%H')
                    infile_name = f'{model}-{run}_{var}_{date_str}_{hour_str}_hinterp.nc'
                    infile_name = os.path.join(temp_dir, infile_name)
                    infile_list[v].append(infile_name)
            else:
                for i in np.arange(time.size):
                    if not (model == 'ARPEGE' and time[i].month == 8 and time[i].day == 11):
                        date_str = time[i].strftime("%m%d")
                        if model == 'ARPEGE':
                            if vinterp and var not in ['OLR', 'STOA']:
                                for h in np.arange(0, 24, 3):
                                    if not (time[i].day == 12 and time[i].month == 8 and h == 0)\
                                    and not (time[i].day == 10 and time[i].month == 8 and h in [0, 15]):
                                        infile_name = f'{model}-{run}_{var}_hinterp_vinterp_{date_str}_{h:02d}.nc'
                                        infile_name = os.path.join(temp_dir, infile_name)
                                        infile_list[v].append(infile_name)                                    
                            else:
                                for h in np.arange(0, 24, 3):
                                    if not (time[i].day == 12 and time[i].month == 8 and h == 0)\
                                    and not (time[i].day == 10 and time[i].month == 8 and h in [0, 15]):
                                        infile_name = f'{model}-{run}_{var}_{date_str}_{h:02d}_hinterp.nc'
                                        infile_name = os.path.join(temp_dir, infile_name)
                                        infile_list[v].append(infile_name)

                        else:
                            if model in ['GEOS', 'IFS', 'FV3', 'ERA5'] and vinterp:
                                for h in np.arange(0, 24, 3):
                                    infile_name = f'{model}-{run}_{var}_hinterp_vinterp_{date_str}_{h:02d}.nc'
                                    infile_name = os.path.join(temp_dir, infile_name)
                                    infile_list[v].append(infile_name)

                            elif model in ['GEOS', 'IFS', 'MPAS', 'SAM'] or var in ['RH', 'WHL', 'H']:
                                for h in np.arange(0, 24, 3):
                                    infile_name = f'{model}-{run}_{var}_{date_str}_{h:02d}_hinterp.nc'
                                    infile_name = os.path.join(temp_dir, infile_name)
                                    infile_list[v].append(infile_name)
                            else:
                                infile_name = f'{model}-{run}_{var}_{date_str}_hinterp.nc'
                                infile_name = os.path.join(temp_dir, infile_name)
                                infile_list[v].append(infile_name)
            v += 1
     
    return infile_list, outfile_list

def get_averagingfilelist(models, runs, variables, data_dir, time_period, vinterp, **kwargs):
    """ Returns list of filenames of horizontally interpolated and merged as well as 
    list of filenames of horizontally interpolated, merged and temporally averaged 
    DYAMOND output files needed for time averaging. 
    
    Parameters:
        models (str): names of models
        runs (str): names of model runs
        variables (list of str): list of variables
        data_dir (str): path to directory to save files 
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    infile_list = []
    outfile_list = []
    option_list = []
    
    for model, run in zip(models, runs):
        for var in variables:
            option = ('')
            if vinterp == 1 and (model == 'GEOS' or model == 'IFS' or model == 'FV3'):
                infile_name = f'{model}-{run}_{var}_hinterp_vinterp_merged_{start_date}-{end_date}.nc'
                outfile_name = f'{model}-{run}_{var}_hinterp_vinterp_timeaverage_{start_date}-{end_date}.nc'
            else:
                infile_name = f'{model}-{run}_{var}_hinterp_merged_{start_date}-{end_date}.nc'
                outfile_name = f'{model}-{run}_{var}_hinterp_timeaverage_{start_date}-{end_date}.nc'

            infile_name = os.path.join(data_dir, model.upper(), infile_name)
            outfile_name = os.path.join(data_dir, model.upper(), outfile_name)
            infile_list.append(infile_name)
            outfile_list.append(outfile_name)
            option_list.append(option)
    
    return infile_list, outfile_list, option_list
 
    
def get_sellonlatboxfilelist(models, runs, variables, data_dir, time_period, lonlatbox, vinterp, **kwargs):
    """ Returns list of filenames needed for the selection of a lat-lon box. Input files are horizontally interpolated
    temporally averaged fields. Output filenames contain boundaries of the lat-lon box.
    
    Parameters:
        models (str): list of model names
        runs (str): list of runs
        variables (list of str): list of variables
        data_dir (str): path to directory to save files 
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        lonlatbox (list): Boundaries of lon-lat box in the following order: [lonmin lonmax latmin latmax]
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    infile_list = []
    outfile_list = []
    lonlatstr = f'{lonlatbox[0]}_{lonlatbox[1]}_{lonlatbox[2]}_{lonlatbox[3]}'
    
    for model, run in zip(models, runs):
        for var in variables:
            option = ('')
            if (model == 'GEOS' or model == 'IFS' or model == 'FV3') and vinterp == 1:
                infile_name = f'{model}-{run}_{var}_hinterp_vinterp_timeaverage_{start_date}-{end_date}.nc'
                outfile_name = f'{model}-{run}_{var}_hinterp_vinterp_timeaverage_{start_date}-{end_date}_{lonlatstr}.nc'
            else:
                infile_name = f'{model}-{run}_{var}_hinterp_timeaverage_{start_date}-{end_date}.nc'
                outfile_name = f'{model}-{run}_{var}_hinterp_timeaverage_{start_date}-{end_date}_{lonlatstr}.nc'
            infile_name = os.path.join(data_dir, model.upper(), infile_name)
            outfile_name = os.path.join(data_dir, model.upper(), outfile_name)
            infile_list.append(infile_name)
            outfile_list.append(outfile_name)
        
    return infile_list, outfile_list
       

def get_vinterpolation_per_timestep_filelist(models, runs, variables, time_period, data_dir, num_timesteps, temp_dir, **kwargs):
    """ Returns list of filenames needed for the vertical interpolation which is done separately for 
    every timestep. The input files consist of one file per horizontally interpolated and merged model
    variable. Output files consist of one file per model variable and per timestep.
    
    Parameters:
        models (str): list of model names
        runs (str): list of runs
        variables (list of str): list of variables
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        data_dir (str): path to directory to save files
        temp_dir (str): path to directory with temporary files   
    """

    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    infile_list = []
    model_list = []
    run_list = []
    variable_list = []
    heightfile_list = []
    targetheightfile_list = []
    
    for model, run in zip(models, runs):
        heightfile = f'{model}-{run}_H_hinterp_merged_{start_date}-{end_date}.nc'
        heightfile = os.path.join(data_dir, model, heightfile)
        targetheightfile = get_path2targetheightfile(model, data_dir)

        for var in variables:
            infile = f'{model}-{run}_{var}_hinterp_merged_{start_date}-{end_date}.nc'
            infile = os.path.join(data_dir, model, infile)
            #infile_list.append(infile)
            
            for timestep in range(num_timesteps):
                model_list.append(model)
                run_list.append(run)
                variable_list.append(var)
                infile_list.append(infile)
                heightfile_list.append(heightfile)
                targetheightfile_list.append(targetheightfile)
            
    #models, runs, variables, infiles, heightfiles, targetheightfiles
    return model_list, run_list, variable_list, infile_list, heightfile_list, targetheightfile_list


def get_height_calculation_filelist(models, runs, time_period, data_dir, num_timesteps, temp_dir, lonlatbox, **kwargs):
    """ Returns list of filenames needed for the calculation of model level heights from model level
    pressure and temperature. Output files consist of one file for every timestep.
    
    Parameters:
        models (list of str): names of models
        runs (list of str): names of model runs
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        data_dir (str): path to directory to save files   
        num_timesteps (num): number of timesteps in input files
        temp_dir (str): path to directory for temporary output files
        lonlatbox (list of num): borders of lon-lat box to sample from [lonmin, lonmax, latmin, latmax]
         
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    
    presfile_list = []
    tempfile_list = []
    z0file_list = []
    model_list = []
    run_list = []
    
    for model, run in zip(models, runs):
        tempfile = f'{model}-{run}_TEMP_hinterp_merged_{start_date}-{end_date}.nc'
        tempfile = os.path.join(data_dir, model, tempfile)
        presfile = f'{model}-{run}_PRES_hinterp_merged_{start_date}-{end_date}.nc'
        presfile = os.path.join(data_dir, model, presfile)
        z0file = get_path2z0file(model, run)
        for t in range(num_timesteps):
            tempfile_list.append(tempfile)
            presfile_list.append(presfile)
            z0file_list.append(z0file)
            model_list.append(model)
            run_list.append(run)
   
    return model_list, run_list, presfile_list, tempfile_list, z0file_list
    
def get_rhcalculation_filelist(models, runs, time_period, num_timesteps, data_dir, **kwargs):
    """ Returns filelist needed to calculate the relative humidity for every timestep. Input files
    consist of three files containing horizontally interpolated and merged temperature, specific 
    humidity and pressure.
    
    Parameters:
        models (list of str): names of models
        runs (list of str): names of model runs
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        num_timesteps (num): number of timesteps in input files
        data_dir (str): path to directory to save files    
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    temp_files = []
    qv_files = []
    pres_files = []
    model_list = []
    run_list = []
    
    for model, run in zip(models, runs):
        temp_file = f'{model}-{run}_TEMP_hinterp_merged_{start_date}-{end_date}.nc'
        temp_file = os.path.join(data_dir, model, temp_file)
        pres_file = f'{model}-{run}_PRES_hinterp_merged_{start_date}-{end_date}.nc'
        pres_file = os.path.join(data_dir, model, pres_file)
        qv_file = f'{model}-{run}_QV_hinterp_merged_{start_date}-{end_date}.nc'
        qv_file = os.path.join(data_dir, model, qv_file)
        for timestep in range(num_timesteps):
            temp_files.append(temp_file)
            pres_files.append(pres_file)
            qv_files.append(qv_file)
            model_list.append(model)
            run_list.append(run)
    
    return model_list, run_list, temp_files, qv_files, pres_files

def get_wcalculation_filelist(models, runs, time_period, num_timesteps, data_dir, **kwargs):
    """ Returns filelist needed to calculate vertical velocity from pressure velocity for every timestep. 
    Input files consist of three files containing horizontally interpolated and merged temperature, 
    specific humidity and pressure.
    
    NOT NEEDED ANY MORE!!!
    
    Parameters:
        models (list of str): names of models
        runs (list of str): names of model runs
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        num_timesteps (num): number of timesteps in input files
        data_dir (str): path to directory to save files    
    """
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    omega_files = []
    temp_files = []
    qv_files = []
    pres_files = []
    height_files = []
    model_list = []
    run_list = []
    
    for model, run in zip(models, runs):
        omega_file = f'{model}-{run}_OMEGA_hinterp_merged_{start_date}-{end_date}.nc'
        omega_file = os.path.join(data_dir, model, omega_file)        
        temp_file = f'{model}-{run}_TEMP_hinterp_merged_{start_date}-{end_date}.nc'
        temp_file = os.path.join(data_dir, model, temp_file)
        pres_file = f'{model}-{run}_PRES_hinterp_merged_{start_date}-{end_date}.nc'
        pres_file = os.path.join(data_dir, model, pres_file)
        qv_file = f'{model}-{run}_QV_hinterp_merged_{start_date}-{end_date}.nc'
        qv_file = os.path.join(data_dir, model, qv_file)
        height_file = f'{model}-{run}_H_hinterp_merged_{start_date}-{end_date}.nc'
        height_file = os.path.join(data_dir, model, height_file)
        for timestep in range(num_timesteps):
            omega_files.append(omega_file)
            temp_files.append(temp_file)
            pres_files.append(pres_file)
            height_files.append(height_file)
            qv_files.append(qv_file)
            model_list.append(model)
            run_list.append(run)
    
    return model_list, run_list, omega_files, temp_files, qv_files, pres_files, height_files

def get_interpolationtohalflevels_filelist(models, runs, time_period, variables, num_timesteps, data_dir, temp_dir, **kwargs):
    """ Return filelist needed for interpolation from full model levels to half levels.
    
    Parameters:
        models (list of str): names of models
        runs (list of str): names of model runs
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        variables (list of str): list of variable names
        data_dir (str): path to directory of input file
        temp_dir (str): path to directory for output files
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    infile_list = []
    model_list = []
    run_list = []
    variable_list = []
    
    #outfiles = []
    for model, run in zip(models, runs):

        for var in variables:
            infile = f'{model}-{run}_{var}_hinterp_merged_{start_date}-{end_date}.nc'
            infile = os.path.join(data_dir, model, infile)
            for timestep in range(num_timesteps):
                model_list.append(model)
                run_list.append(run)
                variable_list.append(var)
                infile_list.append(infile)
                

                #infiles.append(infile)
                #outfile = f'{model}-{run}_{var}HL_{date_str}_{hour_str}_hinterp.nc'
                #outfile = os.path.join(temp_dir, outfile)
                #outfiles.append(outfile)
        
    return model_list, run_list, variable_list, infile_list

def get_netfluxcalculationfilelist(fluxes, models, runs, time_period, data_dir, **kwargs):
    """ Returns filelist to calculate net fluxes from upward and downward fluxes 
    
    Parameters:
        fluxes (list of str): Name of variables containing 1. downward, 2. upward and 3. net radiation fluxes
            ['SDTOA', 'SUTOA', 'STOA']
        models (list of str): names of models
        runs (list of str): names of model runs
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        data_dir (str): path to directory of input file
    """
    
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    
    infile_list = []
    tempfile_list = []
    outfile_list = []

    for model, run in zip(models, runs):
        infile_1 = f'{model}-{run}_{fluxes[0]}_hinterp_merged_{start_date}-{end_date}.nc'
        infile_2 = f'{model}-{run}_{fluxes[1]}_hinterp_merged_{start_date}-{end_date}.nc'       
        tempfile = f'{model}-{run}_{fluxes[0]}_{fluxes[1]}_hinterp_merged_{start_date}-{end_date}.nc'
        outfile = f'{model}-{run}_{fluxes[2]}_hinterp_merged_{start_date}-{end_date}.nc'
        infile_1 = os.path.join(data_dir, model, infile_1)
        infile_2 = os.path.join(data_dir, model, infile_2)
        infile_sublist = [infile_1, infile_2]
        tempfile = os.path.join(data_dir, model, tempfile)
        outfile = os.path.join(data_dir, model, outfile)
        infile_list.append(infile_sublist)
        tempfile_list.append(tempfile)
        outfile_list.append(outfile)
        
    return infile_list, tempfile_list, outfile_list    

def get_deaccumulationfilelist(models, runs, variables, time_period, data_dir, **kwargs):
    """ Returns filelist needed to deaccumulate radiation fields given as energy. 
    
    Parameters:
        models (list of str): names of models
        runs (list of str): names of model runs
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        data_dir (str): path to directory of input file
    """
    time = pd.date_range(time_period[0], time_period[1]+'-21', freq='3h')
    start_date = time[0].strftime("%m%d")
    end_date = time[-1].strftime("%m%d")
    
    model_list = []
    infile_list = []
    variable_list = []
    
    for model, run in zip(models, runs):
        for var in variables:
            infile = f'{model}-{run}_{var}_hinterp_merged_{start_date}-{end_date}.nc'
            infile = os.path.join(data_dir, model.upper(), infile)
        
            infile_list.append(infile)
            model_list.append(model)
            variable_list.append(var)
    
    return model_list, infile_list, variable_list

# def get_samplefilelist(num_samples_tot, models, runs, variables, time_period, lonlatbox, data_dir, sample_days, experiment=None, day=None, **kwargs):
#     """ Returns filelist needed to perform Monte Carlo Sampling of Profiles.
    
#     Parameters:
#         num_samples_tot (num): number of samples to draw
#         models (list of str): names of models
#         runs (list of str): names of model runs
#         variables (list of str): list of variable names
#         time_period (list of str): list containing start and end of time period as string 
#             in the format YYYY-mm-dd 
#         lonlatbox (list of ints): borders of lon-lat box to sample from [lonmin, lonmax, latmin, latmax]
#         data_dir (str): path to directory of input file
#         experiment (num): experiment number (if several experiments are performed), needed for output filename
#     """  
#     #model = models[0]
#     #run = runs[0]
#     variables_2D = ['OLR', 'OLRC', 'STOA', 'STOAC', 'IWV', 'CRH', 'SURF_PRES', 'SST']
#     time = pd.date_range(time_period[0], time_period[1], freq='1D')
#     start_date_in = time[0].strftime("%m%d")
#     end_date_in = time[-1].strftime("%m%d")
#     if day is None:
#         start_date_out = start_date_in
#         end_date_out = end_date_in
#     else:
#         start_date_out = time[day].strftime("%m%d")
#         end_date_out = time[day].strftime("%m%d")
        
#     if lonlatbox[:2] == [-180, 180]:
#         latlonstr = ''
#     else:
#         latlonstr = f'_{lonlatbox[0]}_{lonlatbox[1]}_{lonlatbox[2]}_{lonlatbox[3]}'
#     if experiment is not None:
#         expstr = f'_{str(experiment)}'
#     else:
#         expstr = ''
#     if sample_days == 'all':
#         sample_day_str = ''
#     else:
#         sample_day_str = '_'+sample_days
    
#     model_list = []
#     run_list = []
#     infile_list = []
#     outfile_list = []
    
#     for model, run in zip(models, runs):
#         model_list.append(model)
#         run_list.append(run)
#         infile_sublist = []
#         outfile_sublist = []
#         for var in variables:
#             if model in ['ICON', 'MPAS'] and var == 'W':
#                 var = 'WHL'
#             if var == 'SST':
#                 file = f'Initialization/DYAMOND_SST_{start_date_in}-{end_date_in}_hinterp_merged.nc'
#             elif model in ['IFS', 'GEOS', 'FV3', 'ARPEGE', 'ERA5'] and var not in variables_2D:
#                 file = f'{model}/{model}-{run}_{var}_hinterp_vinterp_merged_{start_date_in}-{end_date_in}.nc'
#             else: 
#                 file = f'{model}/{model}-{run}_{var}_hinterp_merged_{start_date_in}-{end_date_in}.nc'
#             file = os.path.join(data_dir, file)
#             infile_sublist.append(file) 

#         for var in variables + ['IWV', 'CRH', 'lon', 'lat', 'ind_lon', 'ind_lat', 'sort_ind']:
#             outfile = f'{model}-{run}_{var}_sample_{num_samples_tot}_{start_date_out}-{end_date_out}{sample_day_str}{latlonstr}{expstr}.nc'
#             outfile = os.path.join(data_dir, model, 'random_samples', outfile)
#             outfile_sublist.append(outfile)
#         infile_list.append(infile_sublist)
#         outfile_list.append(outfile_sublist)
    
#     return model_list, run_list, infile_list, outfile_list

# def get_IWV_calc_filelist(models, runs, time_period, num_timesteps, data_dir, **kwargs):
#     """
#     Returns filelist for IWV calculation
#     """
#     time = pd.date_range(time_period[0], time_period[1], freq='1D')
#     start_date = time[0].strftime("%m%d")
#     end_date = time[-1].strftime("%m%d")
#     temp_files = []
#     qv_files = []
#     pres_files = []
#     height_files = []
#     model_list = []
#     run_list = []
    
#     for model, run in zip(models, runs):  
#         if model in ['ARPEGE', 'FV3', 'IFS', 'ERA5', 'GEOS']:
#             temp_file = f'{model}-{run}_TEMP_hinterp_vinterp_merged_{start_date}-{end_date}.nc'
#             pres_file = f'{model}-{run}_PRES_hinterp_vinterp_merged_{start_date}-{end_date}.nc'
#             qv_file = f'{model}-{run}_QV_hinterp_vinterp_merged_{start_date}-{end_date}.nc'
#         else:
#             temp_file = f'{model}-{run}_TEMP_hinterp_merged_{start_date}-{end_date}.nc'
#             pres_file = f'{model}-{run}_PRES_hinterp_merged_{start_date}-{end_date}.nc'
#             qv_file = f'{model}-{run}_QV_hinterp_merged_{start_date}-{end_date}.nc'
            
#         height_file = 'target_height.nc'
#         height_file = os.path.join(data_dir, model, height_file)
#         temp_file = os.path.join(data_dir, model, temp_file)
#         pres_file = os.path.join(data_dir, model, pres_file)
#         qv_file = os.path.join(data_dir, model, qv_file)
        
#         for timestep in range(num_timesteps):
#             temp_files.append(temp_file)
#             pres_files.append(pres_file)
#             height_files.append(height_file)
#             qv_files.append(qv_file)
#             model_list.append(model)
#             run_list.append(run)
    
#     return model_list, run_list, temp_files, qv_files, pres_files, height_files
    

