#!/usr/bin/env python
#SBATCH --job-name=vinterp_per_timestep
#SBATCH --output=logs/vinterp_per_timestep-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to vertically interpolate fields separately for each timestep contained
# in the input files.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one timestep in the input files (which contain
# horizontally interpolated fields for one variable and several timesteps).
# Script has to be submitted with option --array=0-<Number_of_timesteps>

import numpy as np
import os
import typhon
import preprocessing_tools as ptools
from netCDF4 import Dataset
import filelists

# load config
config = ptools.config()

# paths to input files, output files and file containing model level heights 
models, runs, variables, infiles, heightfiles, targetheightfiles = filelists.get_vinterpolation_per_timestep_filelist(**config)

timesteps = config['num_timesteps']
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
timestep_ID = np.mod(ID, timesteps) # ID corresponds to timestep

# perform vertical interpolation
ptools.interpolate_vertically_per_timestep(infiles[ID], heightfiles[ID], targetheightfiles[ID], timestep_ID, models[ID], runs[ID], variables[ID], **config)