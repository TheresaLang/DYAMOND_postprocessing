#!/usr/bin/env python
#SBATCH --job-name=model_vinterpolation
#SBATCH --output=logs/model_vinterpolation-%j.out
#SBATCH --account=um0878         # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48       # Specify number of CPUs per task
#SBATCH --time=08:00:00          # Set a limit on the total run time
#SBATCH --mem=0                  # All mem on node
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL        # Notify user by email in case of job failure

import os
import numpy as np
from os.path import join
import processing_tools as ptools

# load config
config = ptools.config()
# load paths to files containing heights
heightfile = ptools.get_path2heightfile(**config)
targetheightfile = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/ICON/target_height.nc'
# get list of input and output files (one entry per vriable)
infiles, outfiles = ptools.get_vinterpolationfilelist(**config)
# get variable names for this model (needed to read file in function for vertical interpolation)
varnames = ptools.get_modelspecific_varnames(config['model'])
varunits = ptools.get_variable_units()

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to variable
# vertical interpolation of file with index ID in l0-2 ist infiles
variable = config['variables'][ID]
ptools.interpolate_vertically(
    infiles[ID], varnames[variable], varunits[variable], heightfile, targetheightfile, outfiles[ID], variable, **config)

