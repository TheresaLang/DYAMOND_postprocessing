#!/usr/bin/env python
#SBATCH --job-name=model_vinterpolation
#SBATCH --output=logs/model_vinterpolation-%j.out
#SBATCH --account=mh1126         # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48       # Specify number of CPUs per task
#SBATCH --time=08:00:00          # Set a limit on the total run time
#SBATCH --mem=0                  # All mem on node
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL        # Notify user by email in case of job failure

# Script to vertically interpolate one field (temporally averaged field) from 
# given model level heights to a target height vector.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one variable.
# Script has to be submitted with option --array=0-<Number_of_variables>
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

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to variable

# vertical interpolation of file with index ID in list infiles
variable = config['variables'][ID]
ptools.interpolate_vertically(
    infiles[ID], outfiles[ID], heightfile, targetheightfile, variable, **config)

