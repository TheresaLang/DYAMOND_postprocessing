#!/usr/bin/env python
#SBATCH --job-name=deaccumulation
#SBATCH --output=logs/deaccumulation-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to deaccumulate radiation fields that are given in accumulated energy [J]
# rather than power [W/m**2]. 
# Model, run, time period and variables have to be specified in config.json.
# One job corresponds to the processing of one file.
# Script has to be submitted with option --array=0-<Number_of_models * Number_of_variables>
import os
import numpy as np
from importlib import reload
from os.path import join
import processing_tools as ptools
import analysis_tools as atools
from netCDF4 import Dataset
import filelists

# load configuration
config = ptools.config()

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to time step

models, infiles, variables = filelists.get_deaccumulationfilelist(**config)

ptools.deaccumulate_fields(models[ID], infiles[ID], variables[ID])