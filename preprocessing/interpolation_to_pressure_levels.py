#!/usr/bin/env python
#SBATCH --job-name=pressure_interpolation
#SBATCH --output=logs/model_interpolation-%j.out
##SBATCH --error=model_interpolation-%j.err

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to vertically interpolate raw 3D DYAMOND output to a uniform pressure grid.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one DYAMOND output file (which contains
# one variable and either one timestep or 8 timesteps.)
# Script has to be submitted with option --array=0-<Number_of_inputfiles>
import os
import numpy as np
from os.path import join
from netCDF4 import Dataset
import preprocessing_tools as ptools
import filelists

# load configuration
config = ptools.config()

# get list of input and output files (infiles contains files with variables to interpolate, 
# presfiles and surfpresfiles contain files with pressure and surface pressure)
infiles, presfiles, surfpresfiles, outfiles, variables = filelists.get_pinterpolation_filelist(**config)
# (new) pressure levels to interpolate on
plev_file = filelists.get_path2targetpresfile(**config)
with Dataset(plev_file) as ds:
    pressure_levels = ds.variables['PRES'][:].filled(np.nan)
# ID of this job
# Each job gets an own ID from 0 to N, where N is the number of jobs in the job array
# N is specified when job is submitted with 
# $ sbatch --array=0-N horizontal_interpolation.py
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to infile

# interpolation for the file with index ID in the list infiles
ptools.interpolate_to_pressure_levels(infiles[ID], presfiles[ID], surfpresfiles[ID], outfiles[ID], pressure_levels, variables[ID], **config)
