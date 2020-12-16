#!/usr/bin/env python
#SBATCH --job-name=pressure_to_height_calc
#SBATCH --output=logs/pressure_to_height_calc-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate model level heights from model level pressure and temperature
# separately for every time step assuming hydrostatic equilibirum (Needed for the IFS model).
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one timestep contained in the input files (which contains
# horizontally interpolated fields for one variable and several timesteps.)
# Script has to be submitted with option --array=0-<Number_of_timesteps*number_of_models>

import numpy as np
import os
import typhon
import preprocessing_tools as ptools
from netCDF4 import Dataset
import filelists

# import config
config = ptools.config()

# paths to files containing pressure and temperature (for all timesteps) and topography
#pres_file, temp_file, out_files = ptools.get_height_calculation_filelist(**config)
models, runs, pres_files, temp_files, z0_files = filelists.get_height_calculation_filelist(**config)
#z0_file = ptools.get_path2z0file(**config)

timesteps = config['num_timesteps']
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
timestep_ID = np.mod(ID, timesteps) # ID corresponds to timestep

# Perform height calculation
ptools.calc_height_from_pressure(pres_files[ID], temp_files[ID], z0_files[ID], timestep_ID, models[ID], runs[timestep_ID], **config)
