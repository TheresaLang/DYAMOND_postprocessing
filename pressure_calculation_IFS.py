#!/usr/bin/env python
#SBATCH --job-name=pressure_calc
#SBATCH --output=logs/pressure_calc-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate pressure at model levels from the surface pressure (Needed only 
# for the IFS model).
# Model, run and time period have to be specified before in config.json.
# One job corresponds to the processing of one timestep contained in the input files 
# (which contain horizontally interpolated fields for one variable and several timesteps)
# Script has to be submitted with option --array=0-<Number_of_timesteps>
import numpy as np
import os
import typhon
import analysis_tools as atools
import processing_tools as ptools
from netCDF4 import Dataset

# load config
config = ptools.config()
model = config['model']
run = config['run']

# path to file containing surface pressure
surf_pres_file = f'{model}-{run}_SURF_PRES_hinterp_merged.nc'
surf_pres_file = os.path.join(config['data_dir'], model, surf_pres_file)

timestep_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to timestep

# perform pressure calculation
ptools.calc_level_pressure_from_surface_pressure_IFS(surf_pres_file, timestep_ID, **config)