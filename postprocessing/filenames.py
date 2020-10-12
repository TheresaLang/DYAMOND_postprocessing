from os.path import join
from datetime import datetime
import postprocessing_tools as ptools

def models_vinterp():
    return ['GEOS', 'FV3', 'ARPEGE', 'IFS', 'ERA5']

def time_period_str(time_period):
    """
    """
    format_str_in = '%Y-%m-%d'
    format_str_out = '%m%d'
    start_time_str = datetime.strptime(time_period[0], format_str_in).strftime(format_str_out)
    end_time_str = datetime.strptime(time_period[1], format_str_in).strftime(format_str_out)
    
    return start_time_str, end_time_str

def preprocessed_output(data_dir, model, run, variable, num_samples, time_period):
    """
    """
    start_time_str, end_time_str = time_period_str(time_period)
    if model in models_vinterp() and variable in ptools.varlist_3D():
        vinterp_str = 'vinterp_' 
    else:
        vinterp_str = ''
        
    if model in ['ICON', 'MPAS'] and variable == 'W':
        variable = 'WHL'
        
    filename = f'{model}-{run}_{variable}_hinterp_{vinterp_str}merged_{start_time_str}-{end_time_str}.nc'
    
    if variable == 'SST':
        filename = f'DYAMOND_SST_{start_time_str}-{end_time_str}_hinterp_merged.nc'
        filename = join(data_dir, 'Initialization', filename)
    else:
        filename = join(data_dir, model, filename)
    
    return filename

def random_ind(data_dir, model, run, num_samples, time_period):
    """
    """
    start_time_str, end_time_str = time_period_str(time_period)
    filename = f'{model}-{run}_random_ind_sample_{num_samples}_{start_time_str}-{end_time_str}.nc'
    filename = join(data_dir, model, 'random_samples', filename)
    
    return filename

def selected_profiles(data_dir, model, run, variable, num_samples, time_period, info_timesteps):
    """
    """
    if info_timesteps:
        info_timestep = '_'+info_timestep
        
    start_time_str, end_time_str = time_period_str(time_period)
    filename = f'{model}-{run}_{variable}_sample_{num_samples}_{start_time_str}-{end_time_str}.nc'
    filename = join(data_dir, model, 'random_samples', filename)
    return filename
    

def averaged_profiles(data_dir, model, run, variable, num_samples, num_percentiles, time_period, info_timesteps):
    """
    """
    start_time_str, end_time_str = time_period_str(time_period)
    filename = f'{model}-{run}_{variable}_{num_percentiles}_averages_{num_samples}_{start_time_str}-{end_time_str}.nc'
    filename = join(data_dir, model, 'random_samples', filename)
    return filename

def heightfile(data_dir, model):
    """
    """
    filename = f'{data_dir}/{model}/target_height.nc'
    return filename

def landmaskfile(data_dir):
    """
    """
    filename = f'{data_dir}/ICON/land_mask.nc'
    return filename