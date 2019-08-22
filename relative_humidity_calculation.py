#!/usr/bin/env python
#SBATCH --job-name=rh_calc
#SBATCH --output=logs/rh_calc-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate relative humidity from specific humidity, temperature and pressure
# separately for every timestep contained in the input files.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one timestep in the input files (which contain
# horizontally interpolated fields for one variable and several timesteps).
# Script has to be submitted with option --array=0-<Number_of_timesteps>
import numpy as np
import os
import typhon
import analysis_tools as atools
import processing_tools as ptools
from netCDF4 import Dataset

# load config
config = ptools.config()

# paths to files containing temperature, pressure and specific humidity
temp_file, qv_file, pres_file = ptools.get_rhcalculation_filelist(**config)

timestep_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to timestep

# perform relative humidity calculation (and save to one file per timestep)
ptools.calc_relative_humidity(temp_file, qv_file, pres_file, timestep_ID, **config)